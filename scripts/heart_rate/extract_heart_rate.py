"""Batch heart-rate extraction from video using tracked or static ROIs."""

import sys
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kilifish_paths import HR_RESULTS

import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.stats import mannwhitneyu

SUPPORTED_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}
EXPECTED_BODYPARTS = ("snout", "dorsal_midline", "tail_base", "tail_tip")
TRACKED_ROI_KEY = "tracked_body_patch"
COLORS = [
    (255, 255, 255),
    (0, 255, 255),
    (0, 255, 0),
    (0, 128, 255),
    (255, 128, 0),
    (255, 0, 255),
]


@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int
    key: str = ""


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = max(1e-6, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    return butter(order, [low, high], btype="band")


def load_roi_map(roi_json: Path):
    if roi_json and roi_json.exists():
        with open(roi_json, "r") as f:
            return json.load(f)
    return {}


def infer_specimen_and_state(p: Path):
    name = p.stem
    low = name.lower()
    state = "unknown"
    if "before" in low:
        state = "before"
    elif "after" in low:
        state = "after"
    parent = p.parent.name
    parent_low = parent.lower()
    looks_like_specimen = any(k in parent_low for k in ["old", "young", "middle"]) or any(
        ch.isdigit() for ch in parent_low
    )
    specimen_id = (
        parent
        if looks_like_specimen and parent_low not in ["old", "young", "middle"]
        else p.stem.split("-before")[0]
        .split("_before")[0]
        .split("-after")[0]
        .split("_after")[0]
    )
    cohort = "unknown"
    for part in p.parts:
        pl = part.lower()
        if "young" in pl:
            cohort = "young"
        elif "old" in pl:
            cohort = "old"
        elif "middle" in pl:
            cohort = "middle"
    return specimen_id, state, cohort


def get_roi_for_key(frame_shape, roi_map, roi_key):
    h, w = frame_shape[:2]
    if roi_key in roi_map:
        r = roi_map[roi_key]
        return ROI(int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]), key=roi_key)
    cw, ch = int(w * 0.5), int(h * 0.5)
    x = (w - cw) // 2
    y = (h - ch) // 2
    return ROI(x, y, cw, ch, key=roi_key + "(fallback)")


def find_videos(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix in SUPPORTED_EXTS)


def safe_nanmean(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return np.nan
    return float(np.nanmean(arr))


def safe_nanmedian(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return np.nan
    return float(np.nanmedian(arr))


def interpolate_short_gaps(values, valid_mask, max_gap_frames):
    series = pd.Series(
        np.where(valid_mask & np.isfinite(values), values, np.nan),
        dtype=np.float64,
    )
    if max_gap_frames <= 0:
        return series.to_numpy()
    series = series.interpolate(limit=max_gap_frames, limit_direction="both", limit_area="inside")
    series = series.ffill(limit=max_gap_frames).bfill(limit=max_gap_frames)
    return series.to_numpy()


def smooth_finite(values, span=5):
    series = pd.Series(values, dtype=np.float64)
    finite = np.isfinite(series.to_numpy())
    if not finite.any():
        return series.to_numpy()
    smoothed = series.ffill().bfill().ewm(span=span, adjust=False).mean()
    out = smoothed.to_numpy(dtype=np.float64)
    out[~finite] = np.nan
    return out


def clip_roi(center_x, center_y, roi_w, roi_h, frame_w, frame_h):
    roi_w = int(np.clip(round(roi_w), 4, frame_w))
    roi_h = int(np.clip(round(roi_h), 4, frame_h))
    x = int(round(center_x - roi_w / 2))
    y = int(round(center_y - roi_h / 2))
    x = max(0, min(x, frame_w - roi_w))
    y = max(0, min(y, frame_h - roi_h))
    return x, y, roi_w, roi_h


def scaled_roi_from_row(row, frame_shape, roi_scale=1.0):
    if not (
        pd.notna(row["roi_x"])
        and pd.notna(row["roi_y"])
        and pd.notna(row["roi_w"])
        and pd.notna(row["roi_h"])
    ):
        return None
    frame_h, frame_w = frame_shape[:2]
    x0 = float(row["roi_x"])
    y0 = float(row["roi_y"])
    w0 = float(row["roi_w"])
    h0 = float(row["roi_h"])
    center_x = x0 + w0 / 2.0
    center_y = y0 + h0 / 2.0
    return clip_roi(center_x, center_y, w0 * roi_scale, h0 * roi_scale, frame_w, frame_h)


def resolve_track_file(video_path: Path, explicit_track_file=""):
    if explicit_track_file:
        track_path = Path(explicit_track_file)
        if track_path.exists():
            return track_path
        raise FileNotFoundError(f"Track file not found: {track_path}")

    stem = video_path.stem
    candidates = []
    for ext in (".csv", ".h5"):
        candidates.extend(sorted(video_path.parent.glob(f"{stem}*DLC*{ext}")))
        candidates.extend(sorted(video_path.parent.glob(f"{stem}*DeepCut*{ext}")))
    if not candidates:
        raise FileNotFoundError(f"No DLC track sidecar found for {video_path}")

    candidates = sorted(set(candidates), key=lambda p: (p.suffix.lower() != ".csv", str(p)))
    return candidates[0]


def load_track_dataframe(track_file: Path):
    suffix = track_file.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(track_file, header=[0, 1, 2], index_col=0)
    elif suffix in {".h5", ".hdf", ".hdf5"}:
        df = pd.read_hdf(track_file)
    else:
        raise ValueError(f"Unsupported track file: {track_file}")
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels < 3:
        raise ValueError(f"Unexpected DLC column format in {track_file}")
    return df


def extract_bodypart_columns(df, bodypart):
    matches = {}
    for col in df.columns:
        _, bp, coord = (str(part) for part in col[:3])
        if bp == bodypart and coord in {"x", "y", "likelihood"}:
            matches[coord] = np.asarray(df[col], dtype=np.float64)
    if not {"x", "y", "likelihood"}.issubset(matches):
        raise KeyError(f"Bodypart '{bodypart}' missing x/y/likelihood columns")
    return matches


def load_dlc_tracks(track_file: Path):
    df = load_track_dataframe(track_file)
    tracks = {}
    for bodypart in EXPECTED_BODYPARTS:
        tracks[bodypart] = extract_bodypart_columns(df, bodypart)
    frame_count = len(df)
    frame_index = np.arange(frame_count, dtype=int)
    return tracks, frame_index


def build_tracked_roi_qc(track_file: Path, frame_shape, track_pcutoff, max_gap_frames):
    frame_h, frame_w = frame_shape[:2]
    tracks, frame_index = load_dlc_tracks(track_file)
    n = len(frame_index)

    interp = {}
    likelihood_cols = []
    for bodypart, cols in tracks.items():
        point_valid = (
            np.isfinite(cols["x"])
            & np.isfinite(cols["y"])
            & np.isfinite(cols["likelihood"])
            & (cols["likelihood"] >= track_pcutoff)
        )
        interp[bodypart] = {
            "x": interpolate_short_gaps(cols["x"], point_valid, max_gap_frames),
            "y": interpolate_short_gaps(cols["y"], point_valid, max_gap_frames),
            "orig_valid": point_valid,
            "likelihood": np.asarray(cols["likelihood"], dtype=np.float64),
        }
        likelihood_cols.append(np.where(np.isfinite(cols["likelihood"]), cols["likelihood"], np.nan))

    snout_x = interp["snout"]["x"]
    snout_y = interp["snout"]["y"]
    dm_x = interp["dorsal_midline"]["x"]
    dm_y = interp["dorsal_midline"]["y"]
    tail_x = interp["tail_base"]["x"]
    tail_y = interp["tail_base"]["y"]

    axis_from_snout = (
        np.isfinite(snout_x)
        & np.isfinite(snout_y)
        & np.isfinite(tail_x)
        & np.isfinite(tail_y)
    )
    axis_from_dm = (
        np.isfinite(dm_x)
        & np.isfinite(dm_y)
        & np.isfinite(tail_x)
        & np.isfinite(tail_y)
    )

    axis_x = np.full(n, np.nan, dtype=np.float64)
    axis_y = np.full(n, np.nan, dtype=np.float64)
    body_len = np.full(n, np.nan, dtype=np.float64)

    axis_x[axis_from_snout] = snout_x[axis_from_snout] - tail_x[axis_from_snout]
    axis_y[axis_from_snout] = snout_y[axis_from_snout] - tail_y[axis_from_snout]
    body_len[axis_from_snout] = np.sqrt(axis_x[axis_from_snout] ** 2 + axis_y[axis_from_snout] ** 2)

    fallback_mask = (~axis_from_snout) & axis_from_dm
    axis_x[fallback_mask] = dm_x[fallback_mask] - tail_x[fallback_mask]
    axis_y[fallback_mask] = dm_y[fallback_mask] - tail_y[fallback_mask]
    body_len[fallback_mask] = 1.7 * np.sqrt(axis_x[fallback_mask] ** 2 + axis_y[fallback_mask] ** 2)

    axis_norm = np.sqrt(axis_x**2 + axis_y**2)
    unit_x = np.divide(axis_x, axis_norm, out=np.full(n, np.nan), where=axis_norm > 1e-6)
    unit_y = np.divide(axis_y, axis_norm, out=np.full(n, np.nan), where=axis_norm > 1e-6)

    anchor_x = np.where(np.isfinite(dm_x), dm_x, (snout_x + tail_x) / 2.0)
    anchor_y = np.where(np.isfinite(dm_y), dm_y, (snout_y + tail_y) / 2.0)

    center_x = anchor_x + 0.18 * body_len * unit_x
    center_y = anchor_y + 0.18 * body_len * unit_y
    center_x = smooth_finite(center_x, span=5)
    center_y = smooth_finite(center_y, span=5)
    body_len = smooth_finite(body_len, span=5)

    roi_w = np.clip(0.22 * body_len, 18, max(18, int(frame_w * 0.35)))
    roi_h = np.clip(0.18 * body_len, 14, max(14, int(frame_h * 0.35)))

    frame_track_conf = np.nanmean(np.vstack(likelihood_cols), axis=0)
    roi_valid = (
        np.isfinite(center_x)
        & np.isfinite(center_y)
        & np.isfinite(roi_w)
        & np.isfinite(roi_h)
        & np.isfinite(frame_track_conf)
    )

    roi_x = np.full(n, np.nan)
    roi_y = np.full(n, np.nan)
    roi_w_px = np.full(n, np.nan)
    roi_h_px = np.full(n, np.nan)

    last_valid = None
    for i in range(n):
        if roi_valid[i]:
            x, y, w, h = clip_roi(center_x[i], center_y[i], roi_w[i], roi_h[i], frame_w, frame_h)
            roi_x[i], roi_y[i], roi_w_px[i], roi_h_px[i] = x, y, w, h
            last_valid = (x, y, w, h)
        elif last_valid is not None:
            roi_x[i], roi_y[i], roi_w_px[i], roi_h_px[i] = last_valid

    qc = pd.DataFrame(
        {
            "frame_idx": frame_index,
            "roi_x": roi_x,
            "roi_y": roi_y,
            "roi_w": roi_w_px,
            "roi_h": roi_h_px,
            "roi_valid": roi_valid.astype(bool),
            "track_conf": frame_track_conf,
            "snout_x": snout_x,
            "snout_y": snout_y,
            "snout_valid": interp["snout"]["orig_valid"].astype(bool),
            "dorsal_midline_x": dm_x,
            "dorsal_midline_y": dm_y,
            "dorsal_midline_valid": interp["dorsal_midline"]["orig_valid"].astype(bool),
            "tail_base_x": tail_x,
            "tail_base_y": tail_y,
            "tail_base_valid": interp["tail_base"]["orig_valid"].astype(bool),
            "tail_tip_x": interp["tail_tip"]["x"],
            "tail_tip_y": interp["tail_tip"]["y"],
            "tail_tip_valid": interp["tail_tip"]["orig_valid"].astype(bool),
            "anchor_x": anchor_x,
            "anchor_y": anchor_y,
            "body_length_px": body_len,
        }
    )
    return qc


def measure_patch_intensity(patch, use_channel):
    if patch.size == 0:
        return np.nan
    if isinstance(use_channel, str):
        stripped = use_channel.strip().lower()
        if stripped in {"0", "1", "2"}:
            use_channel = int(stripped)
        else:
            use_channel = stripped
    if use_channel == "gray":
        return float(np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)))
    if use_channel in [0, 1, 2]:
        return float(np.mean(patch[:, :, use_channel]))
    return float(np.mean(np.median(patch, axis=2)))


def prepare_patch_frame(patch, use_channel, resize_to=None):
    if patch.size == 0:
        return None
    if isinstance(use_channel, str):
        stripped = use_channel.strip().lower()
        if stripped in {"0", "1", "2"}:
            use_channel = int(stripped)
        else:
            use_channel = stripped
    if use_channel == "gray":
        arr = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    elif use_channel in [0, 1, 2]:
        arr = patch[:, :, use_channel]
    else:
        arr = np.median(patch, axis=2)
    if resize_to is not None:
        arr = cv2.resize(arr, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
    return np.asarray(arr, dtype=np.float32)


def decimate_signal_and_qc(x, qc, fps, decimate):
    if decimate <= 1:
        return x, qc.reset_index(drop=True), fps
    x_dec = np.asarray(x, dtype=np.float64)[::decimate]
    qc_dec = qc.iloc[::decimate].reset_index(drop=True)
    return x_dec, qc_dec, fps / decimate


def extract_signal_static(video_path: Path, roi: ROI, use_channel="gray", roi_scale=1.0, return_patch_stack=False, patch_resize=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    xs = []
    frame_rows = []
    patch_frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        row = {
            "frame_idx": frame_idx,
            "roi_x": roi.x,
            "roi_y": roi.y,
            "roi_w": roi.w,
            "roi_h": roi.h,
            "roi_valid": True,
            "track_conf": np.nan,
        }
        scaled = scaled_roi_from_row(row, frame.shape, roi_scale=roi_scale)
        if scaled is None:
            patch = np.empty((0, 0, 3), dtype=frame.dtype)
        else:
            x, y, w, h = scaled
            patch = frame[y : y + h, x : x + w]
        xs.append(measure_patch_intensity(patch, use_channel))
        if return_patch_stack:
            patch_frame = prepare_patch_frame(patch, use_channel, resize_to=patch_resize)
            if patch_frame is None:
                patch_frames.append(np.full((patch_resize, patch_resize), np.nan, dtype=np.float32))
            else:
                patch_frames.append(patch_frame)
        frame_rows.append(row)
        frame_idx += 1
    cap.release()

    x = np.array(xs, dtype=np.float64)
    qc = pd.DataFrame(frame_rows)
    patch_stack = np.stack(patch_frames, axis=0) if return_patch_stack and patch_frames else None
    return x, fps, qc, patch_stack


def extract_signal_tracked(video_path: Path, qc_df: pd.DataFrame, use_channel="gray", roi_scale=1.0, return_patch_stack=False, patch_resize=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)

    xs = []
    frame_idx = 0
    qc_rows = []
    patch_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < len(qc_df):
            row = qc_df.iloc[frame_idx].copy()
        else:
            row = pd.Series(
                {
                    "frame_idx": frame_idx,
                    "roi_x": np.nan,
                    "roi_y": np.nan,
                    "roi_w": np.nan,
                    "roi_h": np.nan,
                    "roi_valid": False,
                    "track_conf": np.nan,
                }
            )

        roi_draw_valid = (
            pd.notna(row["roi_x"])
            and pd.notna(row["roi_y"])
            and pd.notna(row["roi_w"])
            and pd.notna(row["roi_h"])
        )
        signal_valid = bool(row["roi_valid"]) and roi_draw_valid
        scaled = scaled_roi_from_row(row, frame.shape, roi_scale=roi_scale) if roi_draw_valid else None
        if scaled is not None:
            x0, y0, w, h = scaled
            patch = frame[y0 : y0 + h, x0 : x0 + w]
            intensity = measure_patch_intensity(patch, use_channel) if signal_valid else np.nan
        else:
            patch = np.empty((0, 0, 3), dtype=frame.dtype)
            intensity = np.nan

        row["signal"] = intensity
        row["signal_valid"] = signal_valid
        row["roi_draw_valid"] = scaled is not None
        if return_patch_stack:
            patch_frame = prepare_patch_frame(patch, use_channel, resize_to=patch_resize)
            if patch_frame is None:
                patch_frames.append(np.full((patch_resize, patch_resize), np.nan, dtype=np.float32))
            else:
                patch_frames.append(patch_frame)
        qc_rows.append(row.to_dict())
        xs.append(intensity)
        frame_idx += 1

    cap.release()

    x = np.array(xs, dtype=np.float64)
    qc = pd.DataFrame(qc_rows)
    patch_stack = np.stack(patch_frames, axis=0) if return_patch_stack and patch_frames else None
    return x, fps, qc, patch_stack


def estimate_bpm_psd(x, fps, bpm_min, bpm_max, notch_bpm=None):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < max(32, int(round(fps * 2))):
        return np.nan, np.nan
    x = x - np.mean(x)
    lo_hz = max(0.05, bpm_min / 60.0)
    hi_hz = min(8.0, bpm_max / 60.0)
    b, a = butter_bandpass(lo_hz, hi_hz, fs=fps, order=3)
    xf = filtfilt(b, a, x)
    if notch_bpm is not None and notch_bpm > 0:
        w0 = (notch_bpm / 60.0) / (0.5 * fps)
        if 0 < w0 < 1:
            bn, an = iirnotch(w0, 20)
            xf = filtfilt(bn, an, xf)
    nperseg = min(len(xf) // 2, 4096)
    nperseg = max(nperseg, 256)
    f, pxx = welch(xf, fs=fps, nperseg=nperseg, nfft=4 * nperseg, noverlap=nperseg // 2)
    bpm_axis = f * 60.0
    mask = (bpm_axis >= bpm_min) & (bpm_axis <= bpm_max)
    if not np.any(mask):
        idx = int(np.argmax(pxx))
        denom = np.sum(pxx) + 1e-12
    else:
        idx_local = int(np.argmax(pxx[mask]))
        idx = np.arange(len(pxx))[mask][idx_local]
        denom = np.sum(pxx[mask]) + 1e-12
    return float(bpm_axis[idx]), float(pxx[idx] / denom)


def interpolate_patch_stack(flat_stack):
    n_frames, n_features = flat_stack.shape
    filled = np.asarray(flat_stack, dtype=np.float64).copy()
    idx = np.arange(n_frames, dtype=np.float64)
    for feature_idx in range(n_features):
        col = filled[:, feature_idx]
        valid = np.isfinite(col)
        if not np.any(valid):
            filled[:, feature_idx] = 0.0
            continue
        if np.sum(valid) == 1:
            filled[:, feature_idx] = col[valid][0]
            continue
        if not np.all(valid):
            filled[:, feature_idx] = np.interp(idx, idx[valid], col[valid])
    return filled


def extract_evm_signal(patch_stack, fps, low_bpm, high_bpm, alpha):
    if patch_stack is None or patch_stack.size == 0:
        return None
    n_frames = patch_stack.shape[0]
    flat = np.asarray(patch_stack, dtype=np.float64).reshape(n_frames, -1)
    flat = interpolate_patch_stack(flat)
    low_hz = max(0.05, low_bpm / 60.0)
    high_hz = min(8.0, high_bpm / 60.0)
    b, a = butter_bandpass(low_hz, high_hz, fs=fps, order=2)
    filtered = filtfilt(b, a, flat, axis=0)
    magnified = flat + alpha * filtered
    return np.mean(magnified, axis=1)


def moving_window_bpm(
    x,
    fps,
    win_s,
    hop_s,
    bpm_min,
    bpm_max,
    valid_mask=None,
    track_conf=None,
    notch_bpm=None,
    min_valid_fraction=0.75,
):
    n = len(x)
    win = int(round(win_s * fps))
    hop = int(round(hop_s * fps))
    if win < 16:
        raise ValueError("Window too small")
    starts = np.arange(0, n - win + 1, hop, dtype=int)
    centers_t = (starts + win // 2) / fps

    if valid_mask is None:
        valid_mask = np.isfinite(x)
    if track_conf is None:
        track_conf = np.full(n, np.nan)

    bpms = []
    confs = []
    valid_fracs = []
    mean_track_confs = []
    for s in starts:
        seg = np.asarray(x[s : s + win], dtype=np.float64)
        seg_valid = np.asarray(valid_mask[s : s + win], dtype=bool) & np.isfinite(seg)
        valid_fraction = float(np.mean(seg_valid)) if len(seg_valid) else 0.0
        mean_track_conf = safe_nanmean(track_conf[s : s + win])
        valid_fracs.append(valid_fraction)
        mean_track_confs.append(mean_track_conf)
        if valid_fraction < min_valid_fraction or np.sum(seg_valid) < max(32, int(round(fps * 2))):
            bpms.append(np.nan)
            confs.append(np.nan)
            continue

        seg = pd.Series(np.where(seg_valid, seg, np.nan), dtype=np.float64)
        seg = seg.interpolate(limit_direction="both")
        bpm, conf = estimate_bpm_psd(seg.to_numpy(), fps, bpm_min, bpm_max, notch_bpm=notch_bpm)
        bpms.append(bpm)
        confs.append(conf)

    return (
        centers_t,
        np.asarray(bpms, dtype=np.float64),
        np.asarray(confs, dtype=np.float64),
        np.asarray(valid_fracs, dtype=np.float64),
        np.asarray(mean_track_confs, dtype=np.float64),
    )


def build_label(roi_key, t_s, timeline_df):
    if timeline_df.empty:
        return f"{roi_key}: n/a"
    idx = int(np.argmin(np.abs(timeline_df["t_center_s"].to_numpy(dtype=np.float64) - t_s)))
    row = timeline_df.iloc[idx]
    if np.isfinite(row["bpm"]):
        return (
            f"{roi_key}: {row['bpm']:.1f} BPM "
            f"(c={row['confidence']:.2f}, valid={row['valid_fraction']:.2f})"
        )
    return f"{roi_key}: n/a (valid={row['valid_fraction']:.2f})"


def annotate_video(video_path: Path, overlay_specs, out_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_s = frame_idx / fps
        for spec_idx, spec in enumerate(overlay_specs):
            color = COLORS[spec_idx % len(COLORS)]
            qc_df = spec["qc_df"]
            timeline_df = spec["timeline_df"]
            if frame_idx < len(qc_df):
                row = qc_df.iloc[frame_idx]
                if bool(row.get("roi_draw_valid", row.get("roi_valid", False))):
                    x = int(row["roi_x"])
                    y = int(row["roi_y"])
                    w = int(row["roi_w"])
                    h = int(row["roi_h"])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                for bodypart in EXPECTED_BODYPARTS:
                    x_col = f"{bodypart}_x"
                    y_col = f"{bodypart}_y"
                    if x_col in row.index and y_col in row.index:
                        px = row[x_col]
                        py = row[y_col]
                        if np.isfinite(px) and np.isfinite(py):
                            cv2.circle(frame, (int(round(px)), int(round(py))), 3, color, -1)
                label_x = int(row["roi_x"]) if pd.notna(row["roi_x"]) else 15
                label_y = max(18, int(row["roi_y"]) - 8) if pd.notna(row["roi_y"]) else 18
            else:
                label_x, label_y = 15, 18 + 24 * spec_idx
            label = build_label(spec["roi_key"], t_s, timeline_df)
            cv2.putText(
                frame,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()


def summarize_windows(bpms, confs, conf_min):
    finite = np.isfinite(bpms)
    if conf_min > 0:
        finite = finite & np.isfinite(confs) & (confs >= conf_min)
    return safe_nanmedian(bpms[finite]), safe_nanmean(confs[finite])


def run_tracked_roi_pipeline(vp, args, outdir):
    cap = cv2.VideoCapture(str(vp))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read first frame from {vp}")

    track_file = resolve_track_file(vp, args.track_file)
    base_qc = build_tracked_roi_qc(track_file, frame.shape, args.track_pcutoff, args.max_gap_frames)
    use_evm = args.evm_alpha > 0
    x_full, fps_full, qc_df, patch_stack_full = extract_signal_tracked(
        vp,
        base_qc,
        use_channel=args.channel,
        roi_scale=args.roi_scale,
        return_patch_stack=use_evm,
        patch_resize=args.evm_resize,
    )
    x, qc_est, fps = decimate_signal_and_qc(x_full, qc_df, fps_full, args.decimate)
    if use_evm and patch_stack_full is not None:
        patch_stack = patch_stack_full[:: args.decimate]
        x = extract_evm_signal(
            patch_stack,
            fps,
            low_bpm=args.evm_low_bpm if args.evm_low_bpm > 0 else args.bpm_min,
            high_bpm=args.evm_high_bpm if args.evm_high_bpm > 0 else args.bpm_max,
            alpha=args.evm_alpha,
        )
    notch = args.breath_notch_bpm if args.breath_notch_bpm > 0 else None
    centers_t, bpms, confs, valid_fracs, mean_track_confs = moving_window_bpm(
        x,
        fps,
        args.win_s,
        args.hop_s,
        args.bpm_min,
        args.bpm_max,
        valid_mask=qc_est["signal_valid"].to_numpy(dtype=bool),
        track_conf=qc_est["track_conf"].to_numpy(dtype=np.float64),
        notch_bpm=notch,
        min_valid_fraction=args.min_valid_fraction,
    )
    timeline_df = pd.DataFrame(
        {
            "t_center_s": centers_t,
            "bpm": bpms,
            "confidence": confs,
            "valid_fraction": valid_fracs,
            "mean_track_conf": mean_track_confs,
        }
    )
    timeline_csv = outdir / f"{vp.stem}_{TRACKED_ROI_KEY}_moving_bpm.csv"
    qc_csv = outdir / f"{vp.stem}_{TRACKED_ROI_KEY}_qc.csv"
    timeline_df.to_csv(timeline_csv, index=False)
    qc_df.to_csv(qc_csv, index=False)
    video_out = outdir / f"{vp.stem}_{TRACKED_ROI_KEY}_overlay.mp4"
    if not args.skip_overlay:
        annotate_video(
            vp,
            [{"roi_key": TRACKED_ROI_KEY, "qc_df": qc_df, "timeline_df": timeline_df}],
            video_out,
        )
    median_bpm, mean_conf = summarize_windows(bpms, confs, args.conf_min)
    return [
        {
            "roi_key": TRACKED_ROI_KEY,
            "fps": fps,
            "median_bpm": median_bpm,
            "mean_conf": mean_conf,
            "timeline_csv": str(timeline_csv),
            "qc_csv": str(qc_csv),
            "overlay_video": str(video_out) if not args.skip_overlay else "",
            "track_file": str(track_file),
            "valid_window_fraction": safe_nanmean(valid_fracs),
        }
    ]


def run_static_roi_pipeline(vp, args, outdir, roi_map):
    cap = cv2.VideoCapture(str(vp))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read first frame from {vp}")

    sid, state, _ = infer_specimen_and_state(vp)
    extra_keys = [k.strip() for k in args.roi_keys.split(",") if k.strip()]
    default_key = vp.name if vp.name in roi_map else f"{sid}:{state}"
    roi_keys = [default_key] + [k for k in extra_keys if k in roi_map]
    rois = [get_roi_for_key(frame.shape, roi_map, key) for key in roi_keys]

    overlay_specs = []
    rows = []
    for roi in rois:
        use_evm = args.evm_alpha > 0
        x_full, fps_full, qc_df, patch_stack_full = extract_signal_static(
            vp,
            roi,
            use_channel=args.channel,
            roi_scale=args.roi_scale,
            return_patch_stack=use_evm,
            patch_resize=args.evm_resize,
        )
        x, qc_est, fps = decimate_signal_and_qc(x_full, qc_df, fps_full, args.decimate)
        if use_evm and patch_stack_full is not None:
            patch_stack = patch_stack_full[:: args.decimate]
            x = extract_evm_signal(
                patch_stack,
                fps,
                low_bpm=args.evm_low_bpm if args.evm_low_bpm > 0 else args.bpm_min,
                high_bpm=args.evm_high_bpm if args.evm_high_bpm > 0 else args.bpm_max,
                alpha=args.evm_alpha,
            )
        notch = args.breath_notch_bpm if args.breath_notch_bpm > 0 else None
        centers_t, bpms, confs, valid_fracs, mean_track_confs = moving_window_bpm(
            x,
            fps,
            args.win_s,
            args.hop_s,
            args.bpm_min,
            args.bpm_max,
            valid_mask=np.isfinite(x),
            notch_bpm=notch,
            min_valid_fraction=args.min_valid_fraction,
        )
        timeline_df = pd.DataFrame(
            {
                "t_center_s": centers_t,
                "bpm": bpms,
                "confidence": confs,
                "valid_fraction": valid_fracs,
                "mean_track_conf": mean_track_confs,
            }
        )
        timeline_csv = outdir / f"{vp.stem}_{roi.key.replace(':', '-')}_moving_bpm.csv"
        qc_csv = outdir / f"{vp.stem}_{roi.key.replace(':', '-')}_qc.csv"
        timeline_df.to_csv(timeline_csv, index=False)
        qc_df.to_csv(qc_csv, index=False)
        median_bpm, mean_conf = summarize_windows(bpms, confs, args.conf_min)
        rows.append(
            {
                "roi_key": roi.key,
                "fps": fps,
                "median_bpm": median_bpm,
                "mean_conf": mean_conf,
                "timeline_csv": str(timeline_csv),
                "qc_csv": str(qc_csv),
                "overlay_video": "",
                "track_file": "",
                "valid_window_fraction": safe_nanmean(valid_fracs),
            }
        )
        overlay_specs.append({"roi_key": roi.key, "qc_df": qc_df, "timeline_df": timeline_df})

    video_out = outdir / f"{vp.stem}_overlay_multi.mp4"
    if not args.skip_overlay:
        annotate_video(vp, overlay_specs, video_out)
    for row in rows:
        row["overlay_video"] = str(video_out) if not args.skip_overlay else ""
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Folder to scan recursively")
    ap.add_argument("--track_file", default="", help="Optional explicit DLC CSV/H5 track file")
    ap.add_argument(
        "--roi_mode",
        default="tracked_body_patch",
        choices=["tracked_body_patch", "static_roi"],
        help="Use tracked fish-relative ROI or legacy static ROI",
    )
    ap.add_argument(
        "--roi_json",
        default="",
        help="ROI JSON for static mode. Keys: video basename or 'Specimen:state'",
    )
    ap.add_argument(
        "--roi_keys",
        default="",
        help="Optional extra ROI keys for static mode (comma-separated)",
    )
    ap.add_argument("--outdir", default=str(HR_RESULTS), help="Output folder")
    ap.add_argument("--win_s", type=float, default=20.0)
    ap.add_argument("--hop_s", type=float, default=2.0)
    ap.add_argument("--bpm_min", type=float, default=10.0)
    ap.add_argument("--bpm_max", type=float, default=50.0)
    ap.add_argument("--channel", default="gray")
    ap.add_argument("--decimate", type=int, default=3)
    ap.add_argument("--roi_scale", type=float, default=1.0, help="Scale factor applied to the tracked ROI for signal extraction")
    ap.add_argument("--skip_overlay", action="store_true", help="Skip rendering overlay videos")
    ap.add_argument("--include_free_swim", action="store_true", help="Also process videos without before/after in the name")
    ap.add_argument("--conf_min", type=float, default=0.0, help="Minimum PSD confidence to keep a window in summary stats")
    ap.add_argument("--track_pcutoff", type=float, default=0.6, help="Minimum DLC likelihood for using a keypoint")
    ap.add_argument("--max_gap_frames", type=int, default=5, help="Maximum short gap to interpolate in tracks")
    ap.add_argument("--evm_alpha", type=float, default=0.0, help="Temporal amplification factor for ROI-local EVM (0 disables EVM)")
    ap.add_argument("--evm_resize", type=int, default=24, help="Resize tracked ROI patches to NxN before EVM")
    ap.add_argument("--evm_low_bpm", type=float, default=0.0, help="Low BPM bound for ROI-local EVM (0 uses bpm_min)")
    ap.add_argument("--evm_high_bpm", type=float, default=0.0, help="High BPM bound for ROI-local EVM (0 uses bpm_max)")
    ap.add_argument(
        "--min_valid_fraction",
        type=float,
        default=0.75,
        help="Minimum valid-frame fraction required for an HR window",
    )
    ap.add_argument("--breath_notch_bpm", type=float, default=0.0, help="Optional notch at opercular BPM for HR estimation (0=off)")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    roi_map = load_roi_map(Path(args.roi_json)) if args.roi_json else {}

    per_video_rows = []
    for vp in find_videos(root):
        sid, state, cohort = infer_specimen_and_state(vp)
        if state == "unknown" and not args.include_free_swim:
            continue

        try:
            if args.roi_mode == "tracked_body_patch":
                result_rows = run_tracked_roi_pipeline(vp, args, outdir)
            else:
                result_rows = run_static_roi_pipeline(vp, args, outdir, roi_map)
        except Exception as exc:
            print(f"Skip ({type(exc).__name__}): {vp} :: {exc}")
            continue

        for result in result_rows:
            per_video_rows.append(
                {
                    "video": str(vp),
                    "basename": vp.name,
                    "specimen_id": sid,
                    "cohort": cohort,
                    "state": state,
                    "roi_mode": args.roi_mode,
                    "roi_key": result["roi_key"],
                    "track_file": result["track_file"],
                    "fps": result["fps"],
                    "win_s": args.win_s,
                    "hop_s": args.hop_s,
                    "median_bpm": result["median_bpm"],
                    "mean_conf": result["mean_conf"],
                    "valid_window_fraction": result["valid_window_fraction"],
                    "timeline_csv": result["timeline_csv"],
                    "qc_csv": result["qc_csv"],
                    "overlay_video": result["overlay_video"],
                }
            )

    df = pd.DataFrame(per_video_rows)
    summary_csv = outdir / "moving_window_multiROI_summary.csv"
    df.to_csv(summary_csv, index=False)

    if df.empty:
        print("No videos processed.")
        print(f"Saved empty summary: {summary_csv}")
        return

    if args.roi_mode == "tracked_body_patch":
        core_df = df[df["roi_key"] == TRACKED_ROI_KEY].copy()
    else:
        core_df = df[
            df.apply(
                lambda r: r["roi_key"]
                == (Path(r["video"]).name if Path(r["video"]).name in roi_map else f"{r['specimen_id']}:{r['state']}"),
                axis=1,
            )
        ].copy()

    core_csv = outdir / "moving_window_core_summary.csv"
    core_df.to_csv(core_csv, index=False)

    agg_rows = []
    for sid, g in core_df.groupby("specimen_id"):
        cohort_vals = g["cohort"].dropna().unique().tolist()
        cohort = cohort_vals[0] if cohort_vals else "unknown"
        before_vals = g[g["state"] == "before"]["median_bpm"].dropna().values
        after_vals = g[g["state"] == "after"]["median_bpm"].dropna().values
        before = safe_nanmedian(before_vals)
        after = safe_nanmedian(after_vals)
        delta = after - before if np.isfinite(after) and np.isfinite(before) else np.nan
        agg_rows.append(
            {
                "specimen_id": sid,
                "cohort": cohort,
                "before_bpm_med": before,
                "after_bpm_med": after,
                "delta_bpm": delta,
            }
        )
    agg_df = pd.DataFrame(agg_rows)
    agg_csv = outdir / "moving_window_core_by_specimen.csv"
    agg_df.to_csv(agg_csv, index=False)

    cohort_stats = []
    for state_col in ["before_bpm_med", "after_bpm_med", "delta_bpm"]:
        young = agg_df[agg_df["cohort"] == "young"][state_col].dropna().values
        old = agg_df[agg_df["cohort"] == "old"][state_col].dropna().values
        if len(young) >= 2 and len(old) >= 2:
            u, p = mannwhitneyu(young, old, alternative="two-sided")
        else:
            u, p = np.nan, np.nan
        cohort_stats.append(
            {
                "metric": state_col,
                "young_n": len(young),
                "old_n": len(old),
                "young_med": safe_nanmedian(young),
                "old_med": safe_nanmedian(old),
                "mannwhitney_u": float(u) if np.isfinite(u) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
            }
        )
    cohort_df = pd.DataFrame(cohort_stats)
    cohort_csv = outdir / "cohort_stats.csv"
    cohort_df.to_csv(cohort_csv, index=False)

    print(f"\nSaved per-video summary: {summary_csv}")
    print(f"Saved core per-video summary: {core_csv}")
    print(f"Saved per-specimen table: {agg_csv}")
    print(f"Saved cohort stats: {cohort_csv}")
    print("\nTip: inspect *_overlay*.mp4 and *_qc.csv for moving-ROI quality control.")


if __name__ == "__main__":
    main()
