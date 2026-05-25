from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

try:
    from lifelines import CoxPHFitter
except Exception:
    CoxPHFitter = None

try:
    from kilifish_paths import KILLIFISH_V2_ENCODED_ROOT, OUT_V6, OUT_V6_HR, PROJECT_ROOT
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from kilifish_paths import KILLIFISH_V2_ENCODED_ROOT, OUT_V6, OUT_V6_HR, PROJECT_ROOT


ROOT = PROJECT_ROOT
VIDEO_ROOT = KILLIFISH_V2_ENCODED_ROOT
OUTDIR = OUT_V6_HR
TIMELINE_OUTDIR = OUTDIR / "hr_timelines"
BASELINE_CSV = OUT_V6 / "baseline_all_months_with_Mv_features.csv"
PER_VIDEO_CSV = OUT_V6 / "per_video_metrics_robust.csv"
SURVIVAL_CSV = OUT_V6 / "survival_table_autoderived.csv"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_heart_rate as hrmod  # noqa: E402


HR_FEATURES = [
    "hr_bpm_median",
    "hr_bpm_mean",
    "hr_bpm_iqr",
    "hr_conf_mean",
    "hr_valid_window_fraction",
    "hr_mean_track_conf",
]


def parse_age_months(path: str) -> float:
    b = os.path.basename(path)
    m = re.search(r"(\d+(?:\.\d+)?)\s*mon", b)
    if m:
        return float(m.group(1))
    if "2mom" in b:
        return 2.0
    raise ValueError(f"Could not parse age from filename: {path}")


def parse_sex(path: str) -> str:
    low = path.lower()
    if f"{os.sep}male{os.sep}" in low:
        return "male"
    if f"{os.sep}female{os.sep}" in low:
        return "female"
    raise ValueError(f"Could not infer sex from path: {path}")


def parse_subject_id(path: str) -> str:
    low = path.lower()
    m = re.search(rf"{re.escape(os.sep)}(female\d+|male\d+){re.escape(os.sep)}", low)
    if m:
        return m.group(1)
    b = os.path.basename(low)
    m = re.search(r"(female|male)-(\d+)-", b)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    raise ValueError(f"Could not parse subject id from path: {path}")


def relative_video_path(path: Path) -> str:
    return os.path.join("..", os.path.relpath(path, ROOT))


def zscore_series(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    return (s - mu) / sd


def bh_qvalues(pvals: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=pvals.index, dtype=float)
    mask = pvals.notna()
    if not mask.any():
        return out
    ranked = pvals[mask].sort_values()
    m = len(ranked)
    q = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = float(ranked.iloc[i]) * m / rank
        prev = min(prev, val)
        q[i] = prev
    out.loc[ranked.index] = np.clip(q, 0.0, 1.0)
    return out


def safe_nanmean(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if not len(finite):
        return np.nan
    return float(np.mean(finite))


def safe_nanmedian(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if not len(finite):
        return np.nan
    return float(np.median(finite))


def ensure_unique_keys(df: pd.DataFrame, keys: list[str], label: str) -> None:
    dup = df.groupby(keys).size().reset_index(name="n").query("n > 1")
    if dup.empty:
        return
    preview = dup.head(5).to_dict(orient="records")
    raise ValueError(f"{label} has duplicate merge keys on {keys}: {preview}")


def build_hr_args() -> SimpleNamespace:
    return SimpleNamespace(
        track_file="",
        track_pcutoff=0.6,
        max_gap_frames=5,
        channel="1",
        roi_scale=0.4,
        evm_alpha=0.0,
        evm_resize=24,
        decimate=3,
        evm_low_bpm=0.0,
        evm_high_bpm=0.0,
        bpm_min=10.0,
        bpm_max=80.0,
        breath_notch_bpm=0.0,
        win_s=40.0,
        hop_s=2.0,
        min_valid_fraction=0.75,
        skip_overlay=True,
        conf_min=0.0,
    )


def summarize_timeline(timeline_df: pd.DataFrame, summary_row: dict) -> dict:
    bpm = pd.to_numeric(timeline_df["bpm"], errors="coerce")
    conf = pd.to_numeric(timeline_df["confidence"], errors="coerce")
    track_conf = pd.to_numeric(timeline_df["mean_track_conf"], errors="coerce")
    finite_bpm = bpm[np.isfinite(bpm)]
    if len(finite_bpm):
        q75, q25 = np.nanpercentile(finite_bpm.to_numpy(dtype=float), [75, 25])
        bpm_iqr = float(q75 - q25)
        bpm_mean = float(np.nanmean(finite_bpm))
    else:
        bpm_iqr = np.nan
        bpm_mean = np.nan
    return {
        "hr_bpm_median": summary_row.get("median_bpm", np.nan),
        "hr_bpm_mean": bpm_mean,
        "hr_bpm_iqr": bpm_iqr,
        "hr_conf_mean": summary_row.get("mean_conf", np.nan),
        "hr_valid_window_fraction": summary_row.get("valid_window_fraction", np.nan),
        "hr_mean_track_conf": safe_nanmean(track_conf),
        "hr_n_windows": int(np.isfinite(bpm).sum()),
    }


def run_or_load_hr_for_video(vp: Path, args: SimpleNamespace, outdir: Path) -> dict:
    timeline_csv = outdir / f"{vp.stem}_{hrmod.TRACKED_ROI_KEY}_moving_bpm.csv"
    qc_csv = outdir / f"{vp.stem}_{hrmod.TRACKED_ROI_KEY}_qc.csv"
    track_file = hrmod.resolve_track_file(vp, args.track_file)

    if timeline_csv.exists() and qc_csv.exists():
        timeline_df = pd.read_csv(timeline_csv)
        summary_row = {
            "median_bpm": safe_nanmedian(timeline_df["bpm"]),
            "mean_conf": safe_nanmean(timeline_df["confidence"]),
            "valid_window_fraction": safe_nanmean(timeline_df["valid_fraction"]),
            "track_file": str(track_file),
            "timeline_csv": str(timeline_csv),
            "qc_csv": str(qc_csv),
        }
    else:
        result = hrmod.run_tracked_roi_pipeline(vp, args, outdir)[0]
        timeline_df = pd.read_csv(result["timeline_csv"])
        summary_row = result

    base = summarize_timeline(timeline_df, summary_row)
    base.update(
        {
            "video_path": relative_video_path(vp),
            "video_path_abs": str(vp),
            "sex": parse_sex(str(vp)),
            "subject": parse_subject_id(str(vp)),
            "age_months": parse_age_months(str(vp)),
            "timeline_csv": str(timeline_csv),
            "qc_csv": str(qc_csv),
            "track_file": str(track_file),
        }
    )
    return base


def run_or_load_hr_for_video_worker(video_path: str, outdir_path: str) -> dict:
    return run_or_load_hr_for_video(Path(video_path), build_hr_args(), Path(outdir_path))


def fit_cox_one_metric(df: pd.DataFrame, metric: str, month: float) -> dict:
    work = df[["age_at_death_months", "event", "sex_male", "weight_z", metric]].dropna().copy()
    n = len(work)
    events = int(work["event"].sum()) if n else 0
    if n < 8 or events < 5:
        return {
            "month": month,
            "metric": metric,
            "n": n,
            "events": events,
            "hr": np.nan,
            "hr_ci_low": np.nan,
            "hr_ci_high": np.nan,
            "p": np.nan,
            "note": "too_few_rows_or_events",
        }
    if work[metric].nunique(dropna=True) < 2:
        return {
            "month": month,
            "metric": metric,
            "n": n,
            "events": events,
            "hr": np.nan,
            "hr_ci_low": np.nan,
            "hr_ci_high": np.nan,
            "p": np.nan,
            "note": "predictor_has_too_little_variation",
        }
    if CoxPHFitter is None:
        return {
            "month": month,
            "metric": metric,
            "n": n,
            "events": events,
            "hr": np.nan,
            "hr_ci_low": np.nan,
            "hr_ci_high": np.nan,
            "p": np.nan,
            "note": "lifelines_not_available",
        }

    work[f"{metric}_z"] = zscore_series(work[metric].astype(float))
    try:
        cph = CoxPHFitter()
        cph.fit(
            work[["age_at_death_months", "event", f"{metric}_z", "sex_male", "weight_z"]],
            duration_col="age_at_death_months",
            event_col="event",
        )
        row = cph.summary.loc[f"{metric}_z"]
        return {
            "month": month,
            "metric": metric,
            "n": n,
            "events": events,
            "hr": float(row["exp(coef)"]),
            "hr_ci_low": float(row["exp(coef) lower 95%"]),
            "hr_ci_high": float(row["exp(coef) upper 95%"]),
            "p": float(row["p"]),
            "note": "",
        }
    except Exception as exc:
        return {
            "month": month,
            "metric": metric,
            "n": n,
            "events": events,
            "hr": np.nan,
            "hr_ci_low": np.nan,
            "hr_ci_high": np.nan,
            "p": np.nan,
            "note": f"cox_failed:{type(exc).__name__}",
        }


def run_monthly_cox(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for month in sorted(df["baseline_month"].dropna().unique().tolist()):
        sub = df.loc[df["baseline_month"] == month].copy()
        for metric in metrics:
            if metric not in sub.columns:
                continue
            rows.append(fit_cox_one_metric(sub, metric, month))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_global"] = bh_qvalues(out["p"])
    out["q_month"] = out.groupby("month", group_keys=False)["p"].apply(bh_qvalues)
    out["q_metric"] = out.groupby("metric", group_keys=False)["p"].apply(bh_qvalues)
    return out


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run tracked HR over Killifish-v2-encoded and attach HR to v6 Cox analyses.")
    ap.add_argument("--video-root", default=str(VIDEO_ROOT))
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--baseline-csv", default=str(BASELINE_CSV))
    ap.add_argument("--per-video-csv", default=str(PER_VIDEO_CSV))
    ap.add_argument("--survival-csv", default=str(SURVIVAL_CSV))
    ap.add_argument("--jobs", type=int, default=4, help="Parallel workers for per-video HR extraction.")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    video_root = Path(args.video_root)
    outdir = Path(args.outdir)
    timeline_outdir = outdir / "hr_timelines"
    outdir.mkdir(parents=True, exist_ok=True)
    timeline_outdir.mkdir(parents=True, exist_ok=True)

    videos = [vp for vp in hrmod.find_videos(video_root) if vp.suffix.lower() == ".mp4"]

    rows = []
    errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        future_to_video = {
            ex.submit(run_or_load_hr_for_video_worker, str(vp), str(timeline_outdir)): vp
            for vp in videos
        }
        for future in concurrent.futures.as_completed(future_to_video):
            vp = future_to_video[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append({"video_path": str(vp), "error": str(exc)})

    per_video_hr = pd.DataFrame(rows).sort_values(["sex", "subject", "age_months"]).reset_index(drop=True)
    err_df = pd.DataFrame(errors)

    per_video_existing = pd.read_csv(args.per_video_csv)
    baseline_existing = pd.read_csv(args.baseline_csv)
    survival = pd.read_csv(args.survival_csv)

    merge_keys = ["sex", "subject", "age_months"]
    ensure_unique_keys(per_video_hr, merge_keys, "per_video_hr")
    per_video_merged = per_video_existing.merge(
        per_video_hr[[*merge_keys, "video_path", *HR_FEATURES, "hr_n_windows", "timeline_csv", "qc_csv", "track_file"]],
        on=merge_keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "_hr"),
    )

    baseline_with_hr = baseline_existing.merge(
        per_video_hr[[*merge_keys, *HR_FEATURES, "hr_n_windows", "timeline_csv", "qc_csv", "track_file"]],
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )
    baseline_with_hr = baseline_with_hr.merge(
        survival[["sex", "subject", "age_at_death_months", "event"]],
        on=["sex", "subject"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_survival_dup"),
    )
    baseline_with_hr = baseline_with_hr.loc[:, ~baseline_with_hr.columns.duplicated()].copy()

    hr_cox = run_monthly_cox(baseline_with_hr, HR_FEATURES)
    hr_shortlist = hr_cox.loc[hr_cox["q_global"].le(0.1, fill_value=False)].copy()

    per_video_hr.to_csv(outdir / "per_video_hr_metrics.csv", index=False)
    err_df.to_csv(outdir / "errors.csv", index=False)
    per_video_merged.to_csv(outdir / "per_video_metrics_with_hr.csv", index=False)
    baseline_with_hr.to_csv(outdir / "baseline_all_months_with_hr_features.csv", index=False)
    hr_cox.to_csv(outdir / "cox_hr_plus_sex_plus_weight_by_month.csv", index=False)
    hr_cox.sort_values(["q_global", "p", "month", "metric"]).to_csv(
        outdir / "cox_hr_metric_month_ranked.csv", index=False
    )
    hr_shortlist.to_csv(outdir / "cox_hr_exploratory_shortlist.csv", index=False)

    overview = pd.DataFrame(
        {
            "n_videos_found": [len(videos)],
            "n_hr_ok": [len(per_video_hr)],
            "n_hr_errors": [len(err_df)],
            "n_subjects": [per_video_hr[["sex", "subject"]].drop_duplicates().shape[0] if len(per_video_hr) else 0],
            "months": [",".join(map(str, sorted(per_video_hr["age_months"].dropna().unique().tolist()))) if len(per_video_hr) else ""],
            "hr_channel": ["green"],
            "roi_scale": [build_hr_args().roi_scale],
            "win_s": [build_hr_args().win_s],
            "bpm_min": [build_hr_args().bpm_min],
            "bpm_max": [build_hr_args().bpm_max],
            "jobs": [args.jobs],
        }
    )
    overview.to_csv(outdir / "dataset_overview.csv", index=False)

    print(f"Saved HR per-video metrics: {outdir / 'per_video_hr_metrics.csv'}")
    print(f"Saved merged baseline table: {outdir / 'baseline_all_months_with_hr_features.csv'}")
    print(f"Saved HR Cox table: {outdir / 'cox_hr_plus_sex_plus_weight_by_month.csv'}")
    if len(err_df):
        print(f"Errors on {len(err_df)} videos; see {outdir / 'errors.csv'}")


if __name__ == "__main__":
    main()
