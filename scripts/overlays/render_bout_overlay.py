#!/usr/bin/env python3
"""
render_bout_overlay.py

Create an overlay video to sanity-check "bout/burst" detection from DLC tracks.

This script can draw BOTH:
  1) Robust bout detector (Sci Rep-like): robust speed = min across forward windows k
     + morphology cleanup (open+close) + min bout duration.
  2) Simple bout detector, with two modes:
      - "instant": instantaneous (frame-to-frame) speed threshold
      - "v3": V3-like detector: (optional) smoothed positions + speed via np.gradient + hysteresis (high/low) thresholds

Overlay encoding (when --compare_simple is enabled):
  - Robust bout: red OUTER border
  - Simple bout: blue INNER border

Text shows ROBUST vs SIMPLE state and both speeds (BL/s).

No pixel-to-mm conversion. Everything is normalized by per-video Body Length (px),
computed as median distance between head bodypart and tail bodypart on high-confidence frames.

Requirements:
  - numpy, pandas, opencv-python

Typical use:
  python scripts/overlays/render_bout_overlay.py --input video.mp4 --dlc pose.csv --outdir ~/bout_overlays \
      --head snout --tail tail_tip --compare_simple --simple_mode v3

Directory mode:
  python scripts/overlays/render_bout_overlay.py --input /path/to/videos --outdir ~/bout_overlays \
      --compare_simple --simple_mode v3

Notes:
  - If you want it to behave closer to your old V3 notebook, use --simple_mode v3 and tune:
      --simple_high_bl_s and --simple_low_bl_s (hysteresis thresholds)
      --simple_smooth_win (smoothing on x/y before derivative)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import cv2  # type: ignore
except Exception as e:
    raise SystemExit("OpenCV (cv2) is required. Install opencv-python.") from e


# -----------------------------
# DLC loading
# -----------------------------

def _try_read_dlc_multiindex_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
        if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 3:
            return df
    except Exception:
        return None
    return None


def _read_dlc_csv(path: str) -> pd.DataFrame:
    df_mi = _try_read_dlc_multiindex_csv(path)
    if df_mi is not None:
        return df_mi
    return pd.read_csv(path)


def _extract_part(df: pd.DataFrame, bodypart: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract x,y,likelihood arrays for a bodypart from a DLC dataframe.

    Supports:
      - MultiIndex columns (scorer, bodypart, coord)
      - Flat columns: bodypart_x, bodypart_y, bodypart_likelihood
    """
    if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 3:
        scorer = df.columns.levels[0][0]
        try:
            x = df[(scorer, bodypart, "x")].to_numpy(dtype=float)
            y = df[(scorer, bodypart, "y")].to_numpy(dtype=float)
        except KeyError as e:
            raise KeyError(f"Bodypart '{bodypart}' not found in DLC file (multiindex).") from e
        try:
            p = df[(scorer, bodypart, "likelihood")].to_numpy(dtype=float)
        except KeyError:
            p = np.ones_like(x, dtype=float)
        return x, y, p

    # Flat format fallback
    cols = df.columns

    def pick(cands: List[str]) -> str:
        for c in cands:
            if c in cols:
                return c
        raise KeyError(f"Could not find columns for bodypart '{bodypart}'. Tried: {cands}")

    cx = pick([f"{bodypart}_x", f"{bodypart}.x", f"{bodypart} x", f"{bodypart}X"])
    cy = pick([f"{bodypart}_y", f"{bodypart}.y", f"{bodypart} y", f"{bodypart}Y"])
    try:
        cp = pick([f"{bodypart}_likelihood", f"{bodypart}.likelihood", f"{bodypart}_p", f"{bodypart} p"])
    except KeyError:
        cp = None

    x = df[cx].to_numpy(dtype=float)
    y = df[cy].to_numpy(dtype=float)
    if cp is None:
        p = np.ones_like(x, dtype=float)
    else:
        p = df[cp].to_numpy(dtype=float)
    return x, y, p


@dataclass
class Track:
    head_xy: np.ndarray
    head_p: np.ndarray
    tail_xy: np.ndarray
    tail_p: np.ndarray


def load_track(dlc_path: str, head: str, tail: str) -> Track:
    df = _read_dlc_csv(dlc_path)
    hx, hy, hp = _extract_part(df, head)
    tx, ty, tp = _extract_part(df, tail)
    head_xy = np.column_stack([hx, hy])
    tail_xy = np.column_stack([tx, ty])
    return Track(head_xy=head_xy, head_p=hp, tail_xy=tail_xy, tail_p=tp)


# -----------------------------
# Bout detection helpers
# -----------------------------

def compute_body_length_px(track: Track, pcutoff: float = 0.9) -> float:
    ok = (track.head_p >= pcutoff) & (track.tail_p >= pcutoff)
    if not np.any(ok):
        raise ValueError(f"No frames pass pcutoff={pcutoff}. Lower pcutoff or check DLC.")
    d = np.linalg.norm(track.head_xy[ok] - track.tail_xy[ok], axis=1)
    bl = float(np.nanmedian(d))
    if not np.isfinite(bl) or bl <= 0:
        raise ValueError("Body length estimate is invalid. Check tracking.")
    return bl


def _fill_xy(track_xy: np.ndarray, track_p: np.ndarray, pcutoff: float) -> np.ndarray:
    pos = track_xy.astype(float).copy()
    pos[track_p < pcutoff] = np.nan
    dfpos = pd.DataFrame(pos, columns=["x", "y"])
    dfpos = dfpos.interpolate(limit_direction="both")
    return dfpos.to_numpy()


def _smooth_xy(pos_xy: np.ndarray, win: int) -> np.ndarray:
    """Cheap smoothing (no SciPy): rolling mean, centered."""
    if win is None or win <= 1:
        return pos_xy
    df = pd.DataFrame(pos_xy, columns=["x", "y"])
    sm = df.rolling(win, center=True, min_periods=1).mean()
    return sm.to_numpy()


def robust_speed_px_s(pos_xy: np.ndarray, fps: float, ks: Sequence[int]) -> np.ndarray:
    T = pos_xy.shape[0]
    speeds = []
    for k in ks:
        if k <= 0:
            continue
        v = np.full(T, np.nan, dtype=float)
        if T > k:
            disp = np.linalg.norm(pos_xy[k:] - pos_xy[:-k], axis=1)
            v[:-k] = disp / (k / fps)
        speeds.append(v)
    if not speeds:
        raise ValueError("No valid k windows for robust speed.")
    S = np.vstack(speeds)
    return np.nanmin(S, axis=0)


def simple_speed_px_s_instant(pos_xy: np.ndarray, fps: float) -> np.ndarray:
    """Forward difference between adjacent frames."""
    T = pos_xy.shape[0]
    v = np.full(T, np.nan, dtype=float)
    if T >= 2:
        disp = np.linalg.norm(pos_xy[1:] - pos_xy[:-1], axis=1)
        v[1:] = disp * fps
    return v


def simple_speed_px_s_v3(pos_xy: np.ndarray, fps: float) -> np.ndarray:
    """V3-style: derivative via np.gradient (central difference)."""
    x = pos_xy[:, 0]
    y = pos_xy[:, 1]
    vx = np.gradient(x) * fps
    vy = np.gradient(y) * fps
    return np.hypot(vx, vy)


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
    opened = _dilate(_erode(mask, radius), radius)
    closed = _erode(_dilate(opened, radius), radius)
    return closed


def enforce_min_bout_len(mask: np.ndarray, min_len_frames: int) -> np.ndarray:
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


def hysteresis_mask(speed_bl_s: np.ndarray, low: float, high: float) -> np.ndarray:
    """V3-like hysteresis: enter bout at >high, exit at <low."""
    if not (np.isfinite(low) and np.isfinite(high) and low < high):
        raise ValueError(f"Invalid hysteresis thresholds: low={low}, high={high}")
    on = False
    out = np.zeros_like(speed_bl_s, dtype=bool)
    for i, s in enumerate(speed_bl_s):
        if not np.isfinite(s):
            out[i] = on
            continue
        if (not on) and (s > high):
            on = True
        elif on and (s < low):
            on = False
        out[i] = on
    return out


@dataclass
class BoutResult:
    mask: np.ndarray
    speed_bl_s: np.ndarray
    body_length_px: float
    bouts: List[Tuple[int, int]]


def mask_to_intervals(mask: np.ndarray) -> List[Tuple[int, int]]:
    bouts: List[Tuple[int, int]] = []
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


def detect_bouts_from_speed(
    speed_bl_s: np.ndarray,
    fps: float,
    speed_thresh_bl_s: float,
    min_bout_sec: float,
    morph_radius: int,
) -> np.ndarray:
    raw = speed_bl_s > speed_thresh_bl_s
    cleaned = morph_open_close(raw, radius=morph_radius)
    min_len_frames = int(round(min_bout_sec * fps))
    cleaned = enforce_min_bout_len(cleaned, min_len_frames=min_len_frames)
    return cleaned


def detect_bouts_robust(
    track: Track,
    fps: float,
    ks: Sequence[int],
    speed_thresh_bl_s: float,
    min_bout_sec: float,
    morph_radius: int,
    pcutoff: float,
) -> BoutResult:
    bl_px = compute_body_length_px(track, pcutoff=pcutoff)
    pos = _fill_xy(track.head_xy, track.head_p, pcutoff=pcutoff)
    sp_px_s = robust_speed_px_s(pos, fps=fps, ks=ks)
    sp_bl_s = sp_px_s / bl_px
    mask = detect_bouts_from_speed(sp_bl_s, fps, speed_thresh_bl_s, min_bout_sec, morph_radius)
    return BoutResult(mask=mask, speed_bl_s=sp_bl_s, body_length_px=bl_px, bouts=mask_to_intervals(mask))


def detect_bouts_simple(
    track: Track,
    fps: float,
    speed_thresh_bl_s: float,
    min_bout_sec: float,
    morph_radius: int,
    pcutoff: float,
    simple_mode: str,
    simple_smooth_win: int,
    simple_low_bl_s: Optional[float],
    simple_high_bl_s: Optional[float],
) -> BoutResult:
    bl_px = compute_body_length_px(track, pcutoff=pcutoff)
    pos = _fill_xy(track.head_xy, track.head_p, pcutoff=pcutoff)
    pos = _smooth_xy(pos, win=simple_smooth_win)

    if simple_mode == "instant":
        sp_px_s = simple_speed_px_s_instant(pos, fps=fps)
        sp_bl_s = sp_px_s / bl_px
        mask = detect_bouts_from_speed(sp_bl_s, fps, speed_thresh_bl_s, min_bout_sec, morph_radius)

    elif simple_mode == "v3":
        sp_px_s = simple_speed_px_s_v3(pos, fps=fps)
        sp_bl_s = sp_px_s / bl_px

        high = float(speed_thresh_bl_s if simple_high_bl_s is None else simple_high_bl_s)
        low = float((0.5 * high) if simple_low_bl_s is None else simple_low_bl_s)

        # V3-style hysteresis first, then same cleanup/min duration to keep things comparable.
        raw = hysteresis_mask(sp_bl_s, low=low, high=high)
        cleaned = morph_open_close(raw, radius=morph_radius)
        min_len_frames = int(round(min_bout_sec * fps))
        mask = enforce_min_bout_len(cleaned, min_len_frames=min_len_frames)

    else:
        raise ValueError(f"Unknown simple_mode={simple_mode}. Use 'instant' or 'v3'.")

    return BoutResult(mask=mask, speed_bl_s=sp_bl_s, body_length_px=bl_px, bouts=mask_to_intervals(mask))


# -----------------------------
# Overlay rendering
# -----------------------------

def draw_red_border(frame: np.ndarray, thickness: int = 12) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), thickness)


def draw_blue_inner_border(frame: np.ndarray, thickness: int = 10, inset: int = 14) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (inset, inset), (w - 1 - inset, h - 1 - inset), (255, 0, 0), thickness)


def overlay_bouts_compare(
    video_path: str,
    track: Track,
    robust: BoutResult,
    out_path: str,
    simple: Optional[BoutResult] = None,
    fps_override: Optional[float] = None,
    max_frames: Optional[int] = None,
    label_simple: str = "SIMPLE",
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = fps_override if fps_override is not None else cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for: {out_path}")

    T = len(robust.mask)
    use_T = T
    if max_frames is not None:
        use_T = min(use_T, max_frames)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count > 0:
        use_T = min(use_T, frame_count)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(use_T):
        ok, frame = cap.read()
        if not ok:
            break

        # Draw head point (snout, usually)
        x, y = track.head_xy[i]
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(frame, (int(round(x)), int(round(y))), 6, (0, 255, 255), -1)

        r_on = bool(robust.mask[i])
        s_on = bool(simple.mask[i]) if simple is not None else False

        if r_on:
            draw_red_border(frame)
        if s_on:
            draw_blue_inner_border(frame)

        # HUD text
        txt1 = f"ROBUST: {'ON' if r_on else 'off'}"
        if simple is not None:
            txt2 = f"{label_simple}: {'ON' if s_on else 'off'}"
        else:
            txt2 = ""

        r_sp = robust.speed_bl_s[i] if i < len(robust.speed_bl_s) else np.nan
        s_sp = (simple.speed_bl_s[i] if (simple is not None and i < len(simple.speed_bl_s)) else np.nan)

        txt3 = f"speed BL/s: robust={r_sp:.3f}  {label_simple.lower()}={s_sp:.3f}" if simple is not None else f"speed BL/s: robust={r_sp:.3f}"
        txt4 = f"BL_px={robust.body_length_px:.1f}"

        y0 = 40
        for t in [txt1, txt2, txt3, txt4]:
            if not t:
                continue
            cv2.putText(frame, t, (20, y0), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            y0 += 35

        writer.write(frame)

    writer.release()
    cap.release()


# -----------------------------
# CLI
# -----------------------------

def parse_int_list(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def guess_matching_dlc(video_path: str) -> Optional[str]:
    video_dir = os.path.dirname(video_path) or "."
    stem = os.path.splitext(os.path.basename(video_path))[0]
    for filename in sorted(os.listdir(video_dir)):
        if not filename.lower().endswith(".csv"):
            continue
        if filename.startswith(stem) and "dlc" in filename.lower():
            return os.path.join(video_dir, filename)
    return None


def iter_videos(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        return [
            os.path.join(input_path, filename)
            for filename in sorted(os.listdir(input_path))
            if os.path.splitext(filename)[1].lower() in VIDEO_EXTS
        ]
    return [input_path]


def normalize_bodypart(name: str) -> str:
    n = name.strip().lower()
    aliases = {
        "tailtip": "tail_tip",
        "tail-tip": "tail_tip",
        "tail tip": "tail_tip",
        "tailbase": "tail_base",
        "tail-base": "tail_base",
        "tail base": "tail_base",
        "dorsalmidline": "dorsal_midline",
        "dorsal-midline": "dorsal_midline",
        "dorsal midline": "dorsal_midline",
    }
    return aliases.get(n, name)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create bout overlay videos from DLC tracks.")
    ap.add_argument("--input", required=True, help="Path to a video or a directory of videos")
    ap.add_argument("--dlc", default=None, help="DLC CSV for single-video mode. If omitted, tries to guess.")
    ap.add_argument("--outdir", default=None, help="Output directory. Defaults beside the input video or inside input directory.")
    ap.add_argument("--head", default="snout", help="Bodypart for head")
    ap.add_argument("--tail", default="tail_tip", help="Bodypart for tail (for BL)")
    ap.add_argument("--pcutoff", type=float, default=0.9, help="Likelihood cutoff for BL and filling gaps")
    ap.add_argument("--fps", type=float, default=None, help="Override FPS (otherwise read from video)")
    ap.add_argument("--max_frames", type=int, default=None, help="Max frames to render")

    ap.add_argument("--ks", default="1,12,24,48", help="Comma list of k windows for robust speed (frames)")
    ap.add_argument(
        "--robust_thresh_bl_s",
        "--speed_thresh",
        dest="robust_thresh_bl_s",
        type=float,
        default=0.35,
        help="Robust speed threshold (BL/s)",
    )
    ap.add_argument("--min_bout_sec", type=float, default=0.05, help="Min bout duration (sec)")
    ap.add_argument("--morph_radius", type=int, default=2, help="Morph open/close radius (frames)")

    ap.add_argument("--compare_simple", action="store_true", help="Also compute simple detector and draw it (blue border)")
    ap.add_argument(
        "--simple_mode",
        choices=["instant", "v3"],
        default="instant",
        help="Simple detector mode: 'instant' (frame-to-frame) or 'v3' (smoothed + np.gradient + hysteresis)",
    )
    ap.add_argument("--simple_thresh_bl_s", type=float, default=0.35, help="Simple threshold (BL/s) used as high threshold in v3 mode")
    ap.add_argument("--simple_low_bl_s", type=float, default=None, help="V3 hysteresis LOW threshold (BL/s). Default = 0.5 * high")
    ap.add_argument("--simple_high_bl_s", type=float, default=None, help="V3 hysteresis HIGH threshold (BL/s). Default = simple_thresh_bl_s")
    ap.add_argument("--simple_smooth_win", type=int, default=11, help="Smoothing window (frames) for simple_mode=v3 (<=1 disables)")
    return ap.parse_args()


def process_video(video_path: str, dlc_path: str, outdir: str, args: argparse.Namespace) -> None:
    track = load_track(dlc_path, head=normalize_bodypart(args.head), tail=normalize_bodypart(args.tail))

    fps_override = args.fps
    if fps_override is None:
        cap = cv2.VideoCapture(video_path)
        fps_override = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        if fps_override <= 0:
            fps_override = 30.0

    ks = parse_int_list(args.ks)
    robust = detect_bouts_robust(
        track=track,
        fps=fps_override,
        ks=ks,
        speed_thresh_bl_s=args.robust_thresh_bl_s,
        min_bout_sec=args.min_bout_sec,
        morph_radius=args.morph_radius,
        pcutoff=args.pcutoff,
    )

    simple = None
    if args.compare_simple:
        simple = detect_bouts_simple(
            track=track,
            fps=fps_override,
            speed_thresh_bl_s=args.simple_thresh_bl_s,
            min_bout_sec=args.min_bout_sec,
            morph_radius=args.morph_radius,
            pcutoff=args.pcutoff,
            simple_mode=args.simple_mode,
            simple_smooth_win=args.simple_smooth_win,
            simple_low_bl_s=args.simple_low_bl_s,
            simple_high_bl_s=args.simple_high_bl_s,
        )

    stem = os.path.splitext(os.path.basename(video_path))[0]
    suffix = f"compare_{args.simple_mode}" if args.compare_simple else "robust"
    out_path = os.path.join(outdir, f"{stem}_bouts_overlay_{suffix}.mp4")

    overlay_bouts_compare(
        video_path=video_path,
        track=track,
        robust=robust,
        simple=simple,
        out_path=out_path,
        fps_override=fps_override,
        max_frames=args.max_frames,
        label_simple=("SIMPLE_V3" if args.simple_mode == "v3" else "SIMPLE"),
    )

    robust_csv = os.path.join(outdir, f"{stem}_bouts_intervals_robust.csv")
    pd.DataFrame(robust.bouts, columns=["start_frame", "end_frame_exclusive"]).to_csv(robust_csv, index=False)

    print("[ok] overlay:", out_path)
    print("[ok] robust intervals:", robust_csv)
    if args.compare_simple and simple is not None:
        simple_csv = os.path.join(outdir, f"{stem}_bouts_intervals_{args.simple_mode}.csv")
        pd.DataFrame(simple.bouts, columns=["start_frame", "end_frame_exclusive"]).to_csv(simple_csv, index=False)
        print("[ok] simple intervals:", simple_csv)
        print("[summary] Robust bouts:", len(robust.bouts), "Simple bouts:", len(simple.bouts))
        print("[summary] Robust BL_px:", robust.body_length_px, "Simple BL_px:", simple.body_length_px)


def main() -> None:
    args = parse_args()
    videos = iter_videos(args.input)
    if not videos:
        raise SystemExit("No videos found.")

    if args.outdir is None:
        outdir = args.input if os.path.isdir(args.input) else (os.path.dirname(args.input) or ".")
    else:
        outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    for video_path in videos:
        if os.path.isdir(args.input):
            dlc_path = guess_matching_dlc(video_path)
            if dlc_path is None:
                print(f"[skip] no DLC CSV found for {video_path}")
                continue
        else:
            dlc_path = args.dlc or guess_matching_dlc(video_path)
            if dlc_path is None:
                raise SystemExit("No --dlc provided and could not guess a matching DLC CSV.")

        print(f"[video] {video_path}")
        print(f"[dlc]   {dlc_path}")
        process_video(video_path, dlc_path, outdir, args)


if __name__ == "__main__":
    main()
