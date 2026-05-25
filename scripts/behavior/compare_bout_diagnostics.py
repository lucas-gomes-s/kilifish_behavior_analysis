from __future__ import annotations

import importlib.util
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from kilifish_paths import OUT_BOUT_DIAGNOSTICS, PROJECT_ROOT
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from kilifish_paths import OUT_BOUT_DIAGNOSTICS, PROJECT_ROOT


ROOT = PROJECT_ROOT
OUTDIR = OUT_BOUT_DIAGNOSTICS

OLD_SCRIPT = Path(__file__).resolve().parent / "encoded_behavior_pipeline_15fps.py"
NEW_SCRIPT = Path(__file__).resolve().parent / "yoshidak_behavior_pipeline.py"

OVERLAP_AGES = {2.0, 2.5}
THRESHOLDS = (0.35, 0.50, 0.75, 1.00)
DEFAULT_THRESHOLD = 0.50
N_EXAMPLES_PER_SEX_AGE = 2
PLOT_WINDOW_S = 120.0

COMPARE_METRICS = [
    "bout_avg_path_bl",
    "bout_avg_duration_s",
    "bout_avg_speed_bl_s",
    "bout_peak_speed_bl_s",
    "bout_freq_per_min",
    "total_path_bl",
    "avg_speed_bl_s",
]


def load_module(script_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OLD = load_module(OLD_SCRIPT, "encoded_behavior_15fps")
NEW = load_module(NEW_SCRIPT, "yoshidak_behavior")


def robust_speed_noisy_safe(module, pos_xy: np.ndarray, fps: float, ks: Iterable[int]) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return module.robust_speed_px_s(pos_xy, fps=fps, ks=ks)


def detect_bouts(module, speed_bl_s: np.ndarray, fps: float, threshold: float) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    raw = speed_bl_s > threshold
    cleaned = module.morph_open_close(raw, radius=module.MORPH_RADIUS_FRAMES)
    cleaned = module.enforce_min_len(cleaned, int(round(module.MIN_BOUT_SEC * fps)))
    return cleaned, module.mask_to_intervals(cleaned)


def collect_video_cache(module, dataset_label: str) -> Tuple[List[Dict], List[Dict]]:
    videos = module.list_videos(module.DATA_ROOT)
    cache = []
    errors = []
    for vp in videos:
        try:
            sex = module.parse_sex(vp)
            subject = module.parse_subject_id(vp)
            age_months = module.parse_age_months(vp)
            if age_months not in OVERLAP_AGES:
                continue
            dlc = module.guess_matching_dlc_csv(vp)
            track = module.load_track(dlc, module.BP_HEAD, module.BP_TAIL)
            fps = float(module.read_fps_from_video(vp))
            bl_px = float(module.compute_body_length_px(track, pcutoff=module.PCUTOFF))
            head_pos, frac_missing_head = module.fill_xy(track.head_xy, track.head_p, pcutoff=module.PCUTOFF)
            ks = module.robust_ks_for_fps(fps, module.ROBUST_WINDOWS_S)
            speed_px_s = robust_speed_noisy_safe(module, head_pos, fps=fps, ks=ks)
            speed_bl_s = speed_px_s / bl_px

            disp_px = np.full(len(speed_bl_s), np.nan, dtype=float)
            disp_px[1:] = np.linalg.norm(head_pos[1:] - head_pos[:-1], axis=1)
            disp_bl = disp_px / bl_px

            extra = {}
            if hasattr(module, "parse_time_of_day"):
                extra["am_pm"] = module.parse_time_of_day(vp)
            if hasattr(module, "parse_camera"):
                extra["camera"] = module.parse_camera(vp)

            cache.append(
                {
                    "dataset": dataset_label,
                    "video_path": vp,
                    "dlc_csv": dlc,
                    "sex": sex,
                    "subject": subject,
                    "age_months": float(age_months),
                    "fps": fps,
                    "body_length_px": bl_px,
                    "frac_missing_head": float(frac_missing_head),
                    "speed_bl_s": speed_bl_s,
                    "disp_bl": disp_bl,
                    **extra,
                }
            )
        except Exception as e:
            errors.append({"dataset": dataset_label, "video_path": vp, "error": str(e)})
    return cache, errors


def summarize_entry_at_threshold(module, entry: Dict, threshold: float) -> Tuple[Dict, List[Dict]]:
    v = entry["speed_bl_s"]
    disp_bl = entry["disp_bl"]
    fps = float(entry["fps"])
    dt = 1.0 / fps
    duration_s = len(v) * dt
    duration_min = duration_s / 60.0
    total_path_bl = float(np.nansum(disp_bl))
    avg_speed_bl_s = float(np.nanmean(v)) if np.any(np.isfinite(v)) else np.nan

    bout_mask, bouts = detect_bouts(module, v, fps, threshold)
    n_bouts = len(bouts)
    bout_freq_per_min = (n_bouts / duration_min) if duration_min > 0 else np.nan

    bout_rows = []
    bout_durations_s = []
    bout_paths = []
    bout_mean_speeds = []
    bout_peak_speeds = []

    for idx, (a, b) in enumerate(bouts):
        if b <= a:
            continue
        seg_v = v[a:b]
        duration = (b - a) * dt
        if b - a >= 2:
            path = float(np.nansum(disp_bl[a + 1 : b]))
        else:
            path = 0.0
        mean_speed = float(np.nanmean(seg_v)) if np.any(np.isfinite(seg_v)) else np.nan
        peak_speed = float(np.nanmax(seg_v)) if np.any(np.isfinite(seg_v)) else np.nan

        bout_durations_s.append(duration)
        bout_paths.append(path)
        bout_mean_speeds.append(mean_speed)
        bout_peak_speeds.append(peak_speed)

        bout_rows.append(
            {
                "dataset": entry["dataset"],
                "threshold_bl_s": threshold,
                "sex": entry["sex"],
                "subject": entry["subject"],
                "age_months": entry["age_months"],
                "video_path": entry["video_path"],
                "bout_index": idx,
                "start_frame": a,
                "end_frame": b,
                "duration_s": duration,
                "path_bl": path,
                "mean_speed_bl_s": mean_speed,
                "peak_speed_bl_s": peak_speed,
            }
        )

    row = {
        "dataset": entry["dataset"],
        "threshold_bl_s": threshold,
        "sex": entry["sex"],
        "subject": entry["subject"],
        "age_months": entry["age_months"],
        "video_path": entry["video_path"],
        "fps": entry["fps"],
        "body_length_px": entry["body_length_px"],
        "frac_missing_head": entry["frac_missing_head"],
        "total_path_bl": total_path_bl,
        "avg_speed_bl_s": avg_speed_bl_s,
        "n_bouts": n_bouts,
        "bout_freq_per_min": float(bout_freq_per_min),
        "bout_avg_duration_s": float(np.nanmean(bout_durations_s)) if bout_durations_s else np.nan,
        "bout_avg_path_bl": float(np.nanmean(bout_paths)) if bout_paths else np.nan,
        "bout_avg_speed_bl_s": float(np.nanmean(bout_mean_speeds)) if bout_mean_speeds else np.nan,
        "bout_peak_speed_bl_s": float(np.nanmax(bout_peak_speeds)) if bout_peak_speeds else np.nan,
    }
    return row, bout_rows


def summarize_subject_age(per_video_threshold: pd.DataFrame) -> pd.DataFrame:
    agg = {"video_path": "count"}
    for col in ["fps", "body_length_px", "frac_missing_head"] + COMPARE_METRICS:
        agg[col] = "median"
    out = (
        per_video_threshold.groupby(["dataset", "threshold_bl_s", "sex", "subject", "age_months"], as_index=False)
        .agg(agg)
        .rename(columns={"video_path": "n_videos"})
        .sort_values(["dataset", "threshold_bl_s", "sex", "subject", "age_months"])
        .reset_index(drop=True)
    )
    return out


def build_threshold_comparison(subject_age_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grouped = (
        subject_age_df.groupby(["dataset", "threshold_bl_s", "sex", "age_months"], as_index=False)
        .agg({**{m: "median" for m in COMPARE_METRICS}, "subject": "nunique"})
        .rename(columns={"subject": "n_subjects"})
    )

    old = grouped[grouped["dataset"] == "old_15fps"].copy()
    new = grouped[grouped["dataset"] == "yoshidak"].copy()
    merged = new.merge(old, on=["threshold_bl_s", "sex", "age_months"], suffixes=("_new", "_old"), how="outer")

    rows = []
    for _, row in merged.iterrows():
        base = {
            "threshold_bl_s": row["threshold_bl_s"],
            "sex": row["sex"],
            "age_months": row["age_months"],
            "n_subjects_new": row.get("n_subjects_new", np.nan),
            "n_subjects_old": row.get("n_subjects_old", np.nan),
        }
        for metric in COMPARE_METRICS:
            new_val = row.get(f"{metric}_new", np.nan)
            old_val = row.get(f"{metric}_old", np.nan)
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "median_new": new_val,
                    "median_old": old_val,
                    "delta_new_minus_old": new_val - old_val if np.isfinite(new_val) and np.isfinite(old_val) else np.nan,
                    "ratio_new_over_old": new_val / old_val if np.isfinite(new_val) and np.isfinite(old_val) and old_val != 0 else np.nan,
                }
            )
    return grouped, pd.DataFrame(rows)


def select_representative_video(per_video_default: pd.DataFrame, dataset: str, sex: str, subject: str, age: float) -> str:
    sub = per_video_default[
        (per_video_default["dataset"] == dataset)
        & (per_video_default["sex"] == sex)
        & (per_video_default["subject"] == subject)
        & (np.isclose(per_video_default["age_months"], age))
    ].copy()
    if sub.empty:
        raise ValueError("No videos available for representative selection.")
    target = sub["bout_avg_path_bl"].median()
    idx = (sub["bout_avg_path_bl"] - target).abs().idxmin()
    return str(sub.loc[idx, "video_path"])


def choose_examples(per_video_default: pd.DataFrame) -> pd.DataFrame:
    old_keys = set(
        tuple(x)
        for x in per_video_default.loc[per_video_default["dataset"] == "old_15fps", ["sex", "subject", "age_months"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    new_keys = set(
        tuple(x)
        for x in per_video_default.loc[per_video_default["dataset"] == "yoshidak", ["sex", "subject", "age_months"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    common = sorted(old_keys & new_keys)

    rows = []
    for sex in ["female", "male"]:
        for age in sorted(OVERLAP_AGES):
            subjects = sorted([subject for s, subject, a in common if s == sex and np.isclose(a, age)])
            for subject in subjects[:N_EXAMPLES_PER_SEX_AGE]:
                rows.append(
                    {
                        "sex": sex,
                        "subject": subject,
                        "age_months": age,
                        "old_video_path": select_representative_video(per_video_default, "old_15fps", sex, subject, age),
                        "new_video_path": select_representative_video(per_video_default, "yoshidak", sex, subject, age),
                    }
                )
    return pd.DataFrame(rows)


def choose_plot_window(module, entry: Dict, threshold: float) -> Tuple[int, int, List[Tuple[int, int]]]:
    v = entry["speed_bl_s"]
    fps = float(entry["fps"])
    _, bouts = detect_bouts(module, v, fps, threshold)
    if bouts:
        durations = [b - a for a, b in bouts]
        a, b = bouts[int(np.argmax(durations))]
        center = (a + b) // 2
    else:
        finite = np.where(np.isfinite(v))[0]
        center = int(finite[len(finite) // 2]) if len(finite) else len(v) // 2
    half = int(round((PLOT_WINDOW_S * fps) / 2.0))
    start = max(0, center - half)
    end = min(len(v), center + half)
    return start, end, bouts


def plot_trace_pair(old_entry: Dict, new_entry: Dict, threshold: float, outpath: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharey=True)
    for ax, module, entry, label in [
        (axes[0], OLD, old_entry, "old_15fps"),
        (axes[1], NEW, new_entry, "yoshidak"),
    ]:
        start, end, bouts = choose_plot_window(module, entry, threshold)
        fps = float(entry["fps"])
        t = np.arange(start, end) / fps
        v = entry["speed_bl_s"][start:end]
        ax.plot(t, v, lw=0.8, color="black")
        ax.axhline(threshold, color="crimson", ls="--", lw=1.0)
        for a, b in bouts:
            if b <= start or a >= end:
                continue
            aa = max(a, start) / fps
            bb = min(b, end) / fps
            ax.axvspan(aa, bb, color="goldenrod", alpha=0.25)
        ax.set_ylabel("speed (BL/s)")
        ax.set_title(
            f"{label}: {entry['sex']} {entry['subject']} age {entry['age_months']} "
            f"fps={entry['fps']:.2f} video={Path(entry['video_path']).name}"
        )
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_per_bout_relationships(per_bout_default: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"old_15fps": "tab:blue", "yoshidak": "tab:orange"}
    for dataset, g in per_bout_default.groupby("dataset"):
        axes[0].scatter(g["duration_s"], g["path_bl"], s=10, alpha=0.18, color=colors.get(dataset, "gray"), label=dataset)
        axes[1].scatter(g["duration_s"], g["mean_speed_bl_s"], s=10, alpha=0.18, color=colors.get(dataset, "gray"), label=dataset)
    axes[0].set_xlabel("bout duration (s)")
    axes[0].set_ylabel("bout path (BL)")
    axes[0].set_title("Per-bout Duration vs Path")
    axes[1].set_xlabel("bout duration (s)")
    axes[1].set_ylabel("bout mean speed (BL/s)")
    axes[1].set_title("Per-bout Duration vs Mean Speed")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def summarize_within_bout_speed_bands(entry: Dict, threshold: float) -> Dict:
    module = OLD if entry["dataset"] == "old_15fps" else NEW
    bout_mask, _ = detect_bouts(module, entry["speed_bl_s"], float(entry["fps"]), threshold)
    in_bout = entry["speed_bl_s"][bout_mask]
    if len(in_bout) == 0:
        return {
            "dataset": entry["dataset"],
            "sex": entry["sex"],
            "subject": entry["subject"],
            "age_months": entry["age_months"],
            "video_path": entry["video_path"],
            "frames_in_bout": 0,
            "frac_0_50_to_0_75": np.nan,
            "frac_0_75_to_1_00": np.nan,
            "frac_ge_1_00": np.nan,
            "median_speed_in_bout": np.nan,
            "p90_speed_in_bout": np.nan,
        }
    return {
        "dataset": entry["dataset"],
        "sex": entry["sex"],
        "subject": entry["subject"],
        "age_months": entry["age_months"],
        "video_path": entry["video_path"],
        "frames_in_bout": int(len(in_bout)),
        "frac_0_50_to_0_75": float(np.mean((in_bout >= 0.50) & (in_bout < 0.75))),
        "frac_0_75_to_1_00": float(np.mean((in_bout >= 0.75) & (in_bout < 1.00))),
        "frac_ge_1_00": float(np.mean(in_bout >= 1.00)),
        "median_speed_in_bout": float(np.nanmedian(in_bout)),
        "p90_speed_in_bout": float(np.nanpercentile(in_bout, 90)),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"OUTDIR: {OUTDIR}")

    old_cache, old_errors = collect_video_cache(OLD, "old_15fps")
    new_cache, new_errors = collect_video_cache(NEW, "yoshidak")
    cache = old_cache + new_cache

    pd.DataFrame(old_errors + new_errors).to_csv(OUTDIR / "errors.csv", index=False)

    per_video_rows = []
    for threshold in THRESHOLDS:
        for entry in cache:
            module = OLD if entry["dataset"] == "old_15fps" else NEW
            row, _ = summarize_entry_at_threshold(module, entry, threshold)
            per_video_rows.append(row)
    per_video_threshold = pd.DataFrame(per_video_rows)
    per_video_threshold.to_csv(OUTDIR / "per_video_bout_metrics_by_threshold.csv", index=False)

    subject_age_threshold = summarize_subject_age(per_video_threshold)
    subject_age_threshold.to_csv(OUTDIR / "subject_age_bout_metrics_by_threshold.csv", index=False)

    grouped_summary, comparison = build_threshold_comparison(subject_age_threshold)
    grouped_summary.to_csv(OUTDIR / "threshold_sensitivity_grouped_summary.csv", index=False)
    comparison.to_csv(OUTDIR / "threshold_sensitivity_comparison.csv", index=False)

    per_video_default = per_video_threshold[np.isclose(per_video_threshold["threshold_bl_s"], DEFAULT_THRESHOLD)].copy()
    example_index = choose_examples(per_video_default)
    example_index.to_csv(OUTDIR / "matched_trace_examples.csv", index=False)

    cache_map = {entry["video_path"]: entry for entry in cache}
    for row in example_index.itertuples(index=False):
        old_entry = cache_map[row.old_video_path]
        new_entry = cache_map[row.new_video_path]
        fname = f"trace_pair_{row.sex}_{row.subject}_age_{row.age_months:.1f}_thr_{DEFAULT_THRESHOLD:.2f}.png"
        plot_trace_pair(old_entry, new_entry, DEFAULT_THRESHOLD, OUTDIR / fname)

    per_bout_default_rows = []
    for entry in cache:
        module = OLD if entry["dataset"] == "old_15fps" else NEW
        _, bout_rows = summarize_entry_at_threshold(module, entry, DEFAULT_THRESHOLD)
        per_bout_default_rows.extend(bout_rows)
    per_bout_default = pd.DataFrame(per_bout_default_rows)
    per_bout_default.to_csv(OUTDIR / "per_bout_default_threshold.csv", index=False)
    plot_per_bout_relationships(per_bout_default, OUTDIR / "per_bout_relationships_default_threshold.png")

    within_bout_speed_bands = pd.DataFrame(
        [summarize_within_bout_speed_bands(entry, DEFAULT_THRESHOLD) for entry in cache]
    )
    within_bout_speed_bands.to_csv(OUTDIR / "within_bout_speed_band_per_video.csv", index=False)
    within_bout_speed_band_summary = (
        within_bout_speed_bands.groupby(["dataset", "sex", "age_months"], as_index=False)
        .agg(
            {
                "subject": "nunique",
                "video_path": "count",
                "frames_in_bout": "median",
                "frac_0_50_to_0_75": "median",
                "frac_0_75_to_1_00": "median",
                "frac_ge_1_00": "median",
                "median_speed_in_bout": "median",
                "p90_speed_in_bout": "median",
            }
        )
        .rename(columns={"subject": "n_subjects", "video_path": "n_videos"})
        .sort_values(["dataset", "sex", "age_months"])
        .reset_index(drop=True)
    )
    within_bout_speed_band_summary.to_csv(OUTDIR / "within_bout_speed_band_summary.csv", index=False)

    print(f"Saved: {OUTDIR / 'per_video_bout_metrics_by_threshold.csv'}")
    print(f"Saved: {OUTDIR / 'subject_age_bout_metrics_by_threshold.csv'}")
    print(f"Saved: {OUTDIR / 'threshold_sensitivity_comparison.csv'}")
    print(f"Saved: {OUTDIR / 'matched_trace_examples.csv'}")
    print(f"Saved: {OUTDIR / 'per_bout_default_threshold.csv'}")
    print(f"Saved: {OUTDIR / 'within_bout_speed_band_summary.csv'}")


if __name__ == "__main__":
    main()
