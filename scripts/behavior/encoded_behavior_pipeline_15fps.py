from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
except Exception:
    smf = None

try:
    from kilifish_paths import KILLIFISH_V2_ENCODED_ROOT, OUT_V6_OLD_15FPS, PROJECT_ROOT
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from kilifish_paths import KILLIFISH_V2_ENCODED_ROOT, OUT_V6_OLD_15FPS, PROJECT_ROOT


ROOT = PROJECT_ROOT
DATA_ROOT = KILLIFISH_V2_ENCODED_ROOT
OUTDIR = OUT_V6_OLD_15FPS

VIDEO_EXT = ".mp4"
BP_HEAD = "snout"
BP_TAIL = "tail_tip"

PCUTOFF = 0.90
ROBUST_WINDOWS_S = (1 / 15, 0.8, 1.6, 3.2)
SPEED_THRESH_BL_S = 0.5
MORPH_RADIUS_FRAMES = 1
MIN_BOUT_SEC = 0.5
DEFAULT_FPS = 30000 / 1001
FORCE_FPS = 15.0
ARENA_GRID_SIZE = 8
ARENA_CENTER_BOX_FRAC = 0.50
ARENA_WALL_MARGIN_FRAC = 0.10
HABITUATION_EDGE_WINDOW_S = 300.0
HABITUATION_BIN_S = 60.0

CORE_METRICS = [
    "total_path_bl",
    "avg_speed_bl_s",
    "bout_freq_per_min",
    "bout_avg_speed_bl_s",
    "bout_peak_speed_bl_s",
    "bout_avg_path_bl",
    "bout_avg_duration_s",
]

NEW_METRICS = [
    "v_max_bl_s",
    "v_mean",
    "v_int",
    "v2_max",
    "v2_mean",
    "v2_int",
    "v3_max",
    "v3_mean",
    "v3_int",
    "v_int_bouts",
    "v2_int_bouts",
    "v3_int_bouts",
    "v_max_bouts",
    "v2_max_bouts",
    "v3_max_bouts",
    "bout_v_int_mean",
    "bout_v_int_max",
    "bout_v_int_sum",
    "bout_v2_int_mean",
    "bout_v2_int_max",
    "bout_v2_int_sum",
    "bout_v3_int_mean",
    "bout_v3_int_max",
    "bout_v3_int_sum",
]

ARENA_METRICS = [
    "center_frac",
    "near_wall_frac",
    "mean_center_dist_norm",
    "occupancy_bins_frac",
    "occupancy_entropy_norm",
]

HABITUATION_METRICS = [
    "speed_delta_late_minus_early",
    "bout_freq_delta_late_minus_early",
    "center_frac_delta_late_minus_early",
    "near_wall_frac_delta_late_minus_early",
    "speed_slope_per_min",
]

METRICS = CORE_METRICS + [
    m
    for m in (NEW_METRICS + ARENA_METRICS + HABITUATION_METRICS)
    if m not in CORE_METRICS
]


def zscore_series(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - mu) / sd


def parse_age_months(path: str) -> float:
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*mon", os.path.basename(path))
    if not m:
        raise ValueError(f"Could not parse age from filename: {path}")
    return float(m.group(1))


def parse_sex(path: str) -> str:
    low = path.lower()
    if f"{os.sep}male{os.sep}" in low:
        return "male"
    if f"{os.sep}female{os.sep}" in low:
        return "female"
    b = os.path.basename(low)
    if "female" in b:
        return "female"
    if "male" in b:
        return "male"
    raise ValueError(f"Could not infer sex from path: {path}")


def parse_subject_id(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


def guess_matching_dlc_csv(video_path: str) -> str:
    d = os.path.dirname(video_path)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    cands = []
    for fn in os.listdir(d):
        low = fn.lower()
        if low.endswith(".csv") and fn.startswith(stem) and "dlc" in low:
            cands.append(os.path.join(d, fn))
    if not cands:
        raise FileNotFoundError(f"No DLC CSV found next to {video_path}")
    cands.sort(key=lambda p: len(os.path.basename(p)))
    return cands[0]


def list_videos(root: Path) -> List[str]:
    vids = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(VIDEO_EXT):
                vids.append(os.path.join(dirpath, fn))
    vids.sort()
    return vids


def read_dlc_multiindex(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)


def extract_part(df: pd.DataFrame, bodypart: str):
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 3:
        raise ValueError("Expected DLC CSV with 3-level MultiIndex columns.")
    scorer = df.columns.levels[0][0]
    x = df[(scorer, bodypart, "x")].to_numpy(dtype=float)
    y = df[(scorer, bodypart, "y")].to_numpy(dtype=float)
    try:
        p = df[(scorer, bodypart, "likelihood")].to_numpy(dtype=float)
    except KeyError:
        p = np.ones_like(x, dtype=float)
    return x, y, p


@dataclass
class Track:
    head_xy: np.ndarray
    head_p: np.ndarray
    tail_xy: np.ndarray
    tail_p: np.ndarray


def load_track(csv_path: str, head: str, tail: str) -> Track:
    df = read_dlc_multiindex(csv_path)
    hx, hy, hp = extract_part(df, head)
    tx, ty, tp = extract_part(df, tail)
    return Track(
        head_xy=np.column_stack([hx, hy]),
        head_p=hp,
        tail_xy=np.column_stack([tx, ty]),
        tail_p=tp,
    )


def compute_body_length_px(track: Track, pcutoff: float = 0.9) -> float:
    ok = (track.head_p >= pcutoff) & (track.tail_p >= pcutoff)
    if not np.any(ok):
        raise ValueError("No frames pass pcutoff for body length.")
    d = np.linalg.norm(track.head_xy[ok] - track.tail_xy[ok], axis=1)
    bl = float(np.nanmedian(d))
    if not np.isfinite(bl) or bl <= 0:
        raise ValueError("Invalid body length.")
    return bl


def fill_xy(xy: np.ndarray, p: np.ndarray, pcutoff: float) -> Tuple[np.ndarray, float]:
    pos = xy.astype(float).copy()
    missing = (p < pcutoff) | ~np.isfinite(pos).all(axis=1)
    pos[p < pcutoff] = np.nan
    df = pd.DataFrame(pos, columns=["x", "y"])
    filled = df.interpolate(limit_direction="both").to_numpy()
    return filled, float(np.mean(missing))


def robust_ks_for_fps(fps: float, windows_s: Sequence[float]) -> Tuple[int, ...]:
    return tuple(sorted({max(1, int(round(w * fps))) for w in windows_s}))


def robust_speed_px_s(pos_xy: np.ndarray, fps: float, ks: Sequence[int]) -> np.ndarray:
    T = pos_xy.shape[0]
    speeds = []
    for k in ks:
        k = int(k)
        if k <= 0:
            continue
        v = np.full(T, np.nan, dtype=float)
        if T > k:
            disp = np.linalg.norm(pos_xy[k:] - pos_xy[:-k], axis=1)
            v[:-k] = disp / (k / fps)
        speeds.append(v)
    stacked = np.vstack(speeds)
    all_nan = np.all(np.isnan(stacked), axis=0)
    safe = stacked.copy()
    safe[:, all_nan] = np.inf
    out = np.min(safe, axis=0)
    out[all_nan] = np.nan
    return out


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    w = 2 * radius + 1
    c = np.convolve(mask.astype(np.int32), np.ones(w, dtype=np.int32), mode="same")
    return c > 0


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    w = 2 * radius + 1
    c = np.convolve(mask.astype(np.int32), np.ones(w, dtype=np.int32), mode="same")
    return c == w


def morph_open_close(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    opened = _dilate(_erode(mask, radius), radius)
    closed = _erode(_dilate(opened, radius), radius)
    return closed


def enforce_min_len(mask: np.ndarray, min_len_frames: int) -> np.ndarray:
    if min_len_frames <= 1:
        return mask.copy()
    out = mask.copy()
    T = len(out)
    i = 0
    while i < T:
        if not out[i]:
            i += 1
            continue
        j = i
        while j < T and out[j]:
            j += 1
        if (j - i) < min_len_frames:
            out[i:j] = False
        i = j
    return out


def mask_to_intervals(mask: np.ndarray) -> List[Tuple[int, int]]:
    bouts = []
    T = len(mask)
    i = 0
    while i < T:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < T and mask[j]:
            j += 1
        bouts.append((i, j))
        i = j
    return bouts


def detect_bouts_from_speed(speed_bl_s: np.ndarray, fps: float) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    raw = speed_bl_s > SPEED_THRESH_BL_S
    cleaned = morph_open_close(raw, radius=MORPH_RADIUS_FRAMES)
    cleaned = enforce_min_len(cleaned, int(round(MIN_BOUT_SEC * fps)))
    return cleaned, mask_to_intervals(cleaned)


def read_video_meta(video_path: str) -> Tuple[float, float, float]:
    fps = float(FORCE_FPS) if FORCE_FPS is not None else float(DEFAULT_FPS)
    frame_width_px = np.nan
    frame_height_px = np.nan
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if FORCE_FPS is None:
            raw_fps = cap.get(cv2.CAP_PROP_FPS)
            if raw_fps is not None and raw_fps > 0:
                fps = float(raw_fps)
        raw_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        raw_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if raw_width is not None and raw_width > 0:
            frame_width_px = float(raw_width)
        if raw_height is not None and raw_height > 0:
            frame_height_px = float(raw_height)
    except Exception:
        pass
    finally:
        if cap is not None:
            cap.release()
    return fps, frame_width_px, frame_height_px


def safe_nanmean(arr: np.ndarray) -> float:
    return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else np.nan


def safe_linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return np.nan
    try:
        coef = np.polyfit(x[ok], y[ok], 1)
        return float(coef[0])
    except Exception:
        return np.nan


def sanitize_frame_dims(head_pos: np.ndarray, frame_width_px: float, frame_height_px: float) -> Tuple[float, float]:
    frame_w = float(frame_width_px) if np.isfinite(frame_width_px) and frame_width_px > 0 else np.nan
    frame_h = float(frame_height_px) if np.isfinite(frame_height_px) and frame_height_px > 0 else np.nan

    finite_x = head_pos[:, 0][np.isfinite(head_pos[:, 0])]
    finite_y = head_pos[:, 1][np.isfinite(head_pos[:, 1])]
    if not np.isfinite(frame_w):
        frame_w = float(np.nanmax(finite_x) + 1.0) if finite_x.size else 1.0
    if not np.isfinite(frame_h):
        frame_h = float(np.nanmax(finite_y) + 1.0) if finite_y.size else 1.0
    return max(frame_w, 1.0), max(frame_h, 1.0)


def compute_arena_metrics(head_pos: np.ndarray, frame_width_px: float, frame_height_px: float) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    frame_w, frame_h = sanitize_frame_dims(head_pos, frame_width_px, frame_height_px)
    x = np.clip(head_pos[:, 0], 0.0, frame_w - 1e-6)
    y = np.clip(head_pos[:, 1], 0.0, frame_h - 1e-6)

    cx = 0.5 * frame_w
    cy = 0.5 * frame_h
    dx = (x - cx) / max(cx, 1.0)
    dy = (y - cy) / max(cy, 1.0)
    center_dist_norm = np.sqrt(dx * dx + dy * dy) / np.sqrt(2.0)

    center_half_w = 0.5 * ARENA_CENTER_BOX_FRAC * frame_w
    center_half_h = 0.5 * ARENA_CENTER_BOX_FRAC * frame_h
    center_mask = (np.abs(x - cx) <= center_half_w) & (np.abs(y - cy) <= center_half_h)

    wall_margin_px = ARENA_WALL_MARGIN_FRAC * min(frame_w, frame_h)
    near_wall_mask = (
        (x <= wall_margin_px)
        | (x >= frame_w - wall_margin_px)
        | (y <= wall_margin_px)
        | (y >= frame_h - wall_margin_px)
    )

    gx = np.clip((x / frame_w * ARENA_GRID_SIZE).astype(int), 0, ARENA_GRID_SIZE - 1)
    gy = np.clip((y / frame_h * ARENA_GRID_SIZE).astype(int), 0, ARENA_GRID_SIZE - 1)
    grid_idx = gy * ARENA_GRID_SIZE + gx
    counts = np.bincount(grid_idx, minlength=ARENA_GRID_SIZE * ARENA_GRID_SIZE).astype(float)
    total = counts.sum()
    if total > 0:
        p = counts[counts > 0] / total
        occupancy_entropy_norm = float(-(p * np.log(p)).sum() / np.log(len(counts)))
        occupancy_bins_frac = float(np.mean(counts > 0))
    else:
        occupancy_entropy_norm = np.nan
        occupancy_bins_frac = np.nan

    metrics = {
        "center_frac": float(np.mean(center_mask)),
        "near_wall_frac": float(np.mean(near_wall_mask)),
        "mean_center_dist_norm": safe_nanmean(center_dist_norm),
        "occupancy_bins_frac": occupancy_bins_frac,
        "occupancy_entropy_norm": occupancy_entropy_norm,
    }
    return metrics, center_mask, near_wall_mask


def compute_habituation_metrics(
    speed_bl_s: np.ndarray,
    bouts: List[Tuple[int, int]],
    fps: float,
    center_mask: np.ndarray,
    near_wall_mask: np.ndarray,
) -> Dict[str, float]:
    out = {k: np.nan for k in HABITUATION_METRICS}
    T = len(speed_bl_s)
    if T < 2 or not np.isfinite(fps) or fps <= 0:
        return out

    edge_frames = min(int(round(HABITUATION_EDGE_WINDOW_S * fps)), T // 2)
    if edge_frames < 1:
        return out

    def bout_freq_in_window(start: int, end: int) -> float:
        dur_min = (end - start) / float(fps) / 60.0
        if dur_min <= 0:
            return np.nan
        n_bouts = 0
        for a, b in bouts:
            if b <= start or a >= end:
                continue
            n_bouts += 1
        return float(n_bouts / dur_min)

    early = slice(0, edge_frames)
    late = slice(T - edge_frames, T)

    speed_early = safe_nanmean(speed_bl_s[early])
    speed_late = safe_nanmean(speed_bl_s[late])
    center_early = float(np.mean(center_mask[early]))
    center_late = float(np.mean(center_mask[late]))
    wall_early = float(np.mean(near_wall_mask[early]))
    wall_late = float(np.mean(near_wall_mask[late]))
    bout_freq_early = bout_freq_in_window(0, edge_frames)
    bout_freq_late = bout_freq_in_window(T - edge_frames, T)

    if np.isfinite(speed_early) and np.isfinite(speed_late):
        out["speed_delta_late_minus_early"] = float(speed_late - speed_early)
    if np.isfinite(bout_freq_early) and np.isfinite(bout_freq_late):
        out["bout_freq_delta_late_minus_early"] = float(bout_freq_late - bout_freq_early)
    out["center_frac_delta_late_minus_early"] = float(center_late - center_early)
    out["near_wall_frac_delta_late_minus_early"] = float(wall_late - wall_early)

    bin_frames = max(1, int(round(HABITUATION_BIN_S * fps)))
    xs = []
    ys = []
    for start in range(0, T, bin_frames):
        end = min(T, start + bin_frames)
        y = safe_nanmean(speed_bl_s[start:end])
        if np.isfinite(y):
            xs.append(((start + end) * 0.5) / float(fps) / 60.0)
            ys.append(y)
    out["speed_slope_per_min"] = safe_linear_slope(np.asarray(xs, float), np.asarray(ys, float))
    return out


def compute_video_metrics(
    video_path: str,
    track: Track,
    fps: float,
    frame_width_px: float,
    frame_height_px: float,
) -> Dict:
    bl_px = compute_body_length_px(track, pcutoff=PCUTOFF)
    head_pos, frac_missing_head = fill_xy(track.head_xy, track.head_p, pcutoff=PCUTOFF)
    frame_w, frame_h = sanitize_frame_dims(head_pos, frame_width_px, frame_height_px)
    ks = robust_ks_for_fps(fps, ROBUST_WINDOWS_S)
    sp_px_s = robust_speed_px_s(head_pos, fps=fps, ks=ks)
    v = sp_px_s / bl_px

    T = len(v)
    dt = 1.0 / float(fps)
    duration_s = T * dt
    duration_min = duration_s / 60.0

    disp_px = np.full(T, np.nan)
    disp_px[1:] = np.linalg.norm(head_pos[1:] - head_pos[:-1], axis=1)
    disp_bl = disp_px / bl_px
    total_path_bl = float(np.nansum(disp_bl))

    arena_metrics, center_mask, near_wall_mask = compute_arena_metrics(head_pos, frame_w, frame_h)
    bout_mask, bouts = detect_bouts_from_speed(v, fps=fps)
    n_bouts = len(bouts)
    bout_freq_per_min = (n_bouts / duration_min) if duration_min > 0 else np.nan
    habituation_metrics = compute_habituation_metrics(v, bouts, fps, center_mask, near_wall_mask)

    avg_speed_bl_s = float(np.nanmean(v))
    v_max = float(np.nanmax(v)) if np.any(np.isfinite(v)) else np.nan

    v2 = v * v
    v3 = v2 * v

    def _safe_mean(x):
        return float(np.nanmean(x)) if np.any(np.isfinite(x)) else np.nan

    def _safe_max(x):
        return float(np.nanmax(x)) if np.any(np.isfinite(x)) else np.nan

    def _safe_int(x):
        return float(np.nansum(x) * dt) if len(x) else np.nan

    v_mean, v2_mean, v3_mean = _safe_mean(v), _safe_mean(v2), _safe_mean(v3)
    v2_max, v3_max = _safe_max(v2), _safe_max(v3)
    v_int, v2_int, v3_int = _safe_int(v), _safe_int(v2), _safe_int(v3)

    if T > 0 and np.any(bout_mask):
        vb = v[bout_mask]
        v2b = v2[bout_mask]
        v3b = v3[bout_mask]
        v_int_bouts = float(np.nansum(vb) * dt)
        v2_int_bouts = float(np.nansum(v2b) * dt)
        v3_int_bouts = float(np.nansum(v3b) * dt)
        v_max_bouts = float(np.nanmax(vb)) if np.any(np.isfinite(vb)) else np.nan
        v2_max_bouts = float(np.nanmax(v2b)) if np.any(np.isfinite(v2b)) else np.nan
        v3_max_bouts = float(np.nanmax(v3b)) if np.any(np.isfinite(v3b)) else np.nan
    else:
        v_int_bouts = v2_int_bouts = v3_int_bouts = np.nan
        v_max_bouts = v2_max_bouts = v3_max_bouts = np.nan

    bout_durations_s = []
    bout_path_bl = []
    bout_mean_speed = []
    bout_max_speed = []
    bout_v_int = []
    bout_v2_int = []
    bout_v3_int = []

    for a, b in bouts:
        if b <= a:
            continue
        seg_v = v[a:b]
        seg_v2 = v2[a:b]
        seg_v3 = v3[a:b]
        bout_durations_s.append((b - a) * dt)
        if b - a >= 2:
            bout_path_bl.append(float(np.nansum(disp_bl[a + 1 : b])))
        else:
            bout_path_bl.append(0.0)
        bout_mean_speed.append(float(np.nanmean(seg_v)) if np.any(np.isfinite(seg_v)) else np.nan)
        bout_max_speed.append(float(np.nanmax(seg_v)) if np.any(np.isfinite(seg_v)) else np.nan)
        bout_v_int.append(float(np.nansum(seg_v) * dt))
        bout_v2_int.append(float(np.nansum(seg_v2) * dt))
        bout_v3_int.append(float(np.nansum(seg_v3) * dt))

    def _agg(x):
        if len(x) == 0:
            return (np.nan, np.nan, np.nan)
        arr = np.asarray(x, float)
        return (float(np.nanmean(arr)), float(np.nanmax(arr)), float(np.nansum(arr)))

    bout_v_int_mean, bout_v_int_max, bout_v_int_sum = _agg(bout_v_int)
    bout_v2_int_mean, bout_v2_int_max, bout_v2_int_sum = _agg(bout_v2_int)
    bout_v3_int_mean, bout_v3_int_max, bout_v3_int_sum = _agg(bout_v3_int)

    return dict(
        video_path=video_path,
        fps=float(fps),
        n_frames=int(T),
        duration_s=float(duration_s),
        frac_missing_head=float(frac_missing_head),
        body_length_px=float(bl_px),
        frame_width_px=float(frame_w),
        frame_height_px=float(frame_h),
        total_path_bl=total_path_bl,
        avg_speed_bl_s=avg_speed_bl_s,
        n_bouts=int(n_bouts),
        bout_freq_per_min=float(bout_freq_per_min),
        bout_avg_duration_s=float(np.nanmean(bout_durations_s)) if bout_durations_s else np.nan,
        bout_avg_path_bl=float(np.nanmean(bout_path_bl)) if bout_path_bl else np.nan,
        bout_avg_speed_bl_s=float(np.nanmean(bout_mean_speed)) if bout_mean_speed else np.nan,
        bout_max_speed_bl_s=float(np.nanmean(bout_max_speed)) if bout_max_speed else np.nan,
        bout_peak_speed_bl_s=float(np.nanmax(bout_max_speed)) if bout_max_speed else np.nan,
        v_max_bl_s=v_max,
        v_mean=v_mean,
        v_int=v_int,
        v2_max=v2_max,
        v2_mean=v2_mean,
        v2_int=v2_int,
        v3_max=v3_max,
        v3_mean=v3_mean,
        v3_int=v3_int,
        v_int_bouts=v_int_bouts,
        v2_int_bouts=v2_int_bouts,
        v3_int_bouts=v3_int_bouts,
        v_max_bouts=v_max_bouts,
        v2_max_bouts=v2_max_bouts,
        v3_max_bouts=v3_max_bouts,
        bout_v_int_mean=bout_v_int_mean,
        bout_v_int_max=bout_v_int_max,
        bout_v_int_sum=bout_v_int_sum,
        bout_v2_int_mean=bout_v2_int_mean,
        bout_v2_int_max=bout_v2_int_max,
        bout_v2_int_sum=bout_v2_int_sum,
        bout_v3_int_mean=bout_v3_int_mean,
        bout_v3_int_max=bout_v3_int_max,
        bout_v3_int_sum=bout_v3_int_sum,
        **arena_metrics,
        **habituation_metrics,
    )


def summarize_subject_age(per_video: pd.DataFrame) -> pd.DataFrame:
    agg = {}
    if "video_path" in per_video.columns:
        agg["video_path"] = "count"
    for col in [
        "fps",
        "duration_s",
        "frac_missing_head",
        "body_length_px",
        "frame_width_px",
        "frame_height_px",
    ] + METRICS:
        agg[col] = "median"
    out = (
        per_video.groupby(["sex", "subject", "age_months"], as_index=False)
        .agg(agg)
        .rename(columns={"video_path": "n_videos"})
        .sort_values(["sex", "subject", "age_months"])
        .reset_index(drop=True)
    )
    return out


def fit_mixed_models(subject_age: pd.DataFrame) -> pd.DataFrame:
    if smf is None:
        return pd.DataFrame([{"metric": "all", "note": "statsmodels_not_available"}])
    rows = []
    for metric in METRICS:
        if metric not in subject_age.columns:
            continue
        d = subject_age[["subject", "sex", "age_months", metric]].dropna().copy()
        if d["subject"].nunique() < 3:
            rows.append({"metric": metric, "note": "too_few_subjects"})
            continue
        d["y"] = zscore_series(d[metric].astype(float))
        d["age_c"] = d["age_months"] - d["age_months"].mean()
        try:
            res = smf.mixedlm("y ~ age_c * sex", d, groups=d["subject"]).fit(reml=False)
            params = res.params.to_dict()
            pvals = res.pvalues.to_dict()
            rows.append(
                {
                    "metric": metric,
                    "n_rows": len(d),
                    "n_subjects": d["subject"].nunique(),
                    "coef_age_c": params.get("age_c", np.nan),
                    "p_age_c": pvals.get("age_c", np.nan),
                    "coef_sex_male": params.get("sex[T.male]", np.nan),
                    "p_sex_male": pvals.get("sex[T.male]", np.nan),
                    "coef_age_sex": params.get("age_c:sex[T.male]", np.nan),
                    "p_age_sex": pvals.get("age_c:sex[T.male]", np.nan),
                    "converged": getattr(res, "converged", np.nan),
                }
            )
        except Exception as e:
            rows.append({"metric": metric, "note": f"mixedlm_failed: {e}"})
    return pd.DataFrame(rows)


def save_basic_plots(per_video: pd.DataFrame, subject_age: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(per_video["body_length_px"].dropna(), bins=30)
    ax.set_title("Body length (px)")
    ax.set_xlabel("body_length_px")
    fig.tight_layout()
    fig.savefig(OUTDIR / "qc_body_length_hist.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(per_video["frac_missing_head"].dropna(), bins=30)
    ax.set_title("Missing head fraction")
    ax.set_xlabel("frac_missing_head")
    fig.tight_layout()
    fig.savefig(OUTDIR / "qc_missing_head_hist.png")
    plt.close(fig)

    for metric in CORE_METRICS:
        if metric not in subject_age.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for sex, g in subject_age.groupby("sex"):
            gg = g.groupby("age_months", as_index=False)[metric].median().sort_values("age_months")
            ax.plot(gg["age_months"], gg[metric], marker="o", label=sex)
        ax.set_title(metric)
        ax.set_xlabel("age_months")
        ax.set_ylabel(metric)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTDIR / f"traj_subject_age_{metric}.png")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run robust behavior metrics over the encoded legacy killifish videos."
    )
    parser.add_argument("--data-root", default=str(DATA_ROOT), help="Folder to scan for encoded .mp4 files.")
    parser.add_argument("--outdir", default=str(OUTDIR), help="Output directory for old 15 fps results.")
    return parser.parse_args()


def main() -> None:
    global DATA_ROOT, OUTDIR

    args = parse_args()
    DATA_ROOT = Path(args.data_root)
    OUTDIR = Path(args.outdir)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"OUTDIR: {OUTDIR}")
    print(f"FORCE_FPS: {FORCE_FPS}")

    videos = list_videos(DATA_ROOT)
    print(f"Found videos: {len(videos)}")

    rows = []
    errors = []
    for vp in videos:
        try:
            sex = parse_sex(vp)
            subject = parse_subject_id(vp)
            age_months = parse_age_months(vp)
            dlc = guess_matching_dlc_csv(vp)
            track = load_track(dlc, BP_HEAD, BP_TAIL)
            fps, frame_width_px, frame_height_px = read_video_meta(vp)
            m = compute_video_metrics(vp, track, fps, frame_width_px, frame_height_px)
            m.update(sex=sex, subject=subject, age_months=age_months, dlc_csv=dlc)
            rows.append(m)
        except Exception as e:
            errors.append({"video_path": vp, "error": str(e)})

    per_video = pd.DataFrame(rows).sort_values(["sex", "subject", "age_months"])
    err_df = pd.DataFrame(errors)

    per_video.to_csv(OUTDIR / "per_video_metrics_robust.csv", index=False)
    err_df.to_csv(OUTDIR / "errors.csv", index=False)

    qc_cols = [
        "sex",
        "subject",
        "age_months",
        "fps",
        "n_frames",
        "duration_s",
        "frac_missing_head",
        "body_length_px",
        "frame_width_px",
        "frame_height_px",
    ]
    qc = per_video[qc_cols].copy()
    qc.to_csv(OUTDIR / "qc_per_video.csv", index=False)

    overview = pd.DataFrame(
        {
            "n_videos": [len(per_video)],
            "n_subjects": [per_video[["sex", "subject"]].drop_duplicates().shape[0]],
            "n_subjects_female": [per_video.loc[per_video["sex"] == "female", "subject"].nunique()],
            "n_subjects_male": [per_video.loc[per_video["sex"] == "male", "subject"].nunique()],
            "ages": [",".join(map(str, sorted(per_video["age_months"].dropna().unique().tolist())))],
            "force_fps": [FORCE_FPS],
        }
    )
    overview.to_csv(OUTDIR / "dataset_overview.csv", index=False)

    subject_age = summarize_subject_age(per_video)
    subject_age.to_csv(OUTDIR / "subject_age_summary.csv", index=False)

    mixed = fit_mixed_models(subject_age)
    mixed.to_csv(OUTDIR / "mixed_model_results_subject_age.csv", index=False)

    save_basic_plots(per_video, subject_age)

    print(f"Saved per-video metrics: {OUTDIR / 'per_video_metrics_robust.csv'}")
    print(f"Saved subject-age summary: {OUTDIR / 'subject_age_summary.csv'}")
    print(f"Saved mixed model summary: {OUTDIR / 'mixed_model_results_subject_age.csv'}")
    if len(err_df):
        print(f"Errors on {len(err_df)} videos; see {OUTDIR / 'errors.csv'}")


if __name__ == "__main__":
    main()
