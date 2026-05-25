from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

try:
    from kilifish_paths import OUT_V6_YOSHIDAK, PROJECT_ROOT
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from kilifish_paths import OUT_V6_YOSHIDAK, PROJECT_ROOT


ROOT = PROJECT_ROOT
OUTDIR = OUT_V6_YOSHIDAK
INPUT = OUTDIR / "per_video_metrics_robust.csv"

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
    m for m in (NEW_METRICS + ARENA_METRICS + HABITUATION_METRICS) if m not in CORE_METRICS
]

METRIC_LABELS = {
    "total_path_bl": "Total Path [BL]",
    "avg_speed_bl_s": "Average Speed [BL/s]",
    "bout_freq_per_min": "Bout Frequency [bout/min]",
    "bout_avg_speed_bl_s": "Bout Average Speed [BL/s]",
    "bout_peak_speed_bl_s": "Bout Peak Speed [BL/s]",
    "bout_avg_path_bl": "Bout Average Path [BL]",
    "bout_avg_duration_s": "Bout Average Duration [s]",
}


def zscore_series(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - mu) / sd


def bh_qvalues(pvals: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=pvals.index, dtype=float)
    mask = pvals.notna()
    if not mask.any():
        return out
    ranked = pvals[mask].sort_values()
    n = len(ranked)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        prev = min(prev, float(ranked.iloc[i]) * n / rank)
        q[i] = prev
    out.loc[ranked.index] = np.clip(q, 0.0, 1.0)
    return out


def summarize_subject_age_ampm(per_video: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "video_path": "count",
        "camera": lambda s: ",".join(sorted(pd.unique(s))),
    }
    for col in [
        "fps",
        "duration_s",
        "frac_missing_head",
        "body_length_px",
        "frame_width_px",
        "frame_height_px",
    ] + METRICS:
        if col in per_video.columns:
            agg[col] = "median"
    return (
        per_video.groupby(["sex", "subject", "age_months", "am_pm"], as_index=False)
        .agg(agg)
        .rename(columns={"video_path": "n_videos", "camera": "cameras"})
        .sort_values(["sex", "subject", "age_months", "am_pm"])
        .reset_index(drop=True)
    )


def summarize_groups(subject_age_ampm: pd.DataFrame) -> pd.DataFrame:
    summary_cols = [
        "fps",
        "duration_s",
        "frac_missing_head",
        "body_length_px",
        "frame_width_px",
        "frame_height_px",
    ] + [m for m in METRICS if m in subject_age_ampm.columns]
    agg = {"subject": "nunique", "n_videos": "sum"}
    for col in summary_cols:
        agg[col] = "median"
    return (
        subject_age_ampm.groupby(["sex", "age_months", "am_pm"], as_index=False)
        .agg(agg)
        .rename(columns={"subject": "n_subjects"})
        .sort_values(["sex", "age_months", "am_pm"])
        .reset_index(drop=True)
    )


def fit_one_model(d: pd.DataFrame):
    model = smf.mixedlm("y ~ pm + age_c + sex + pm:sex", d, groups=d["subject"])
    errors = []
    for method in [None, "lbfgs", "powell", "cg", "bfgs", "nm"]:
        try:
            kwargs = {"reml": False}
            if method is not None:
                kwargs["method"] = method
            if method in {"powell", "nm"}:
                kwargs["maxiter"] = 2000
            res = model.fit(**kwargs)
            return res, "default" if method is None else method
        except Exception as exc:
            label = "default" if method is None else method
            errors.append(f"{label}: {exc}")
    raise RuntimeError(" | ".join(errors))


def fit_am_pm_models(subject_age_ampm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        if metric not in subject_age_ampm.columns:
            continue
        d = subject_age_ampm[["subject", "sex", "age_months", "am_pm", metric]].dropna().copy()
        if d["am_pm"].nunique() < 2:
            rows.append({"metric": metric, "note": "missing_am_or_pm"})
            continue
        if d["subject"].nunique() < 6 or d["sex"].nunique() < 2:
            rows.append({"metric": metric, "note": "too_few_subjects"})
            continue
        d["pm"] = (d["am_pm"] == "pm").astype(int)
        d["y"] = zscore_series(d[metric].astype(float))
        d["age_c"] = d["age_months"] - d["age_months"].mean()
        counts = (
            d.groupby(["am_pm", "age_months"])["subject"]
            .nunique()
            .rename("n")
            .to_dict()
        )
        try:
            res, fit_method = fit_one_model(d)
            params = res.params.to_dict()
            pvals = res.pvalues.to_dict()
            rows.append(
                {
                    "metric": metric,
                    "n_rows": int(len(d)),
                    "n_subjects": int(d["subject"].nunique()),
                    "n_am_2mo": int(counts.get(("am", 2.0), 0)),
                    "n_pm_2mo": int(counts.get(("pm", 2.0), 0)),
                    "n_am_2p5mo": int(counts.get(("am", 2.5), 0)),
                    "n_pm_2p5mo": int(counts.get(("pm", 2.5), 0)),
                    "ages": ",".join(map(str, sorted(d["age_months"].unique().tolist()))),
                    "coef_pm": params.get("pm", np.nan),
                    "p_pm": pvals.get("pm", np.nan),
                    "coef_age_c": params.get("age_c", np.nan),
                    "p_age_c": pvals.get("age_c", np.nan),
                    "coef_sex_male": params.get("sex[T.male]", np.nan),
                    "p_sex_male": pvals.get("sex[T.male]", np.nan),
                    "coef_pm_by_sex_male": params.get("pm:sex[T.male]", np.nan),
                    "p_pm_by_sex_male": pvals.get("pm:sex[T.male]", np.nan),
                    "converged": bool(getattr(res, "converged", False)),
                    "fit_method": fit_method,
                }
            )
        except Exception as exc:
            rows.append({"metric": metric, "note": f"mixedlm_failed: {exc}"})

    out = pd.DataFrame(rows)
    if "p_pm" in out.columns:
        out["q_pm"] = bh_qvalues(out["p_pm"])
    if "p_pm_by_sex_male" in out.columns:
        out["q_pm_by_sex_male"] = bh_qvalues(out["p_pm_by_sex_male"])
    return out


def build_core_table(results: pd.DataFrame) -> pd.DataFrame:
    core = results.loc[results["metric"].isin(CORE_METRICS)].copy()
    core["metric_label"] = core["metric"].map(METRIC_LABELS).fillna(core["metric"])
    core["metric_order"] = core["metric"].map({m: i for i, m in enumerate(CORE_METRICS)})
    cols = [
        "metric_order",
        "metric",
        "metric_label",
        "n_rows",
        "n_subjects",
        "coef_pm",
        "p_pm",
        "q_pm",
        "coef_age_c",
        "p_age_c",
        "coef_sex_male",
        "p_sex_male",
        "coef_pm_by_sex_male",
        "p_pm_by_sex_male",
        "q_pm_by_sex_male",
    ]
    return core[[c for c in cols if c in core.columns]].sort_values("metric_order")


def fmt_num(x: float) -> str:
    if not np.isfinite(x):
        return ""
    if abs(x) >= 100 or (abs(x) < 0.001 and x != 0):
        return f"{x:.2e}"
    return f"{x:.3f}"


def fmt_p(x: float) -> str:
    if not np.isfinite(x):
        return ""
    if x < 0.001:
        return "<0.001"
    return f"{x:.3f}"


def write_summary(results: pd.DataFrame, core: pd.DataFrame) -> None:
    ranked = results.sort_values(["q_pm", "p_pm", "metric"]).head(12)
    lines = [
        "# Yoshidak AM vs PM mixed model",
        "",
        "Model: `z(metric) ~ pm + age_c + sex + pm:sex + (1|fish)`.",
        "",
        "`pm=1` is PM and `pm=0` is AM, so `coef_pm` is the PM-AM difference in females after adjusting for age.",
        "`coef_pm_by_sex_male` is the male correction to that PM-AM difference.",
        "",
        "## Core pose metrics",
        "",
        "| Metric | PM-AM beta | p | q | Male correction | p | q |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in core.iterrows():
        lines.append(
            "| {metric} | {b} | {p} | {q} | {bi} | {pi} | {qi} |".format(
                metric=row["metric_label"],
                b=fmt_num(row.get("coef_pm", np.nan)),
                p=fmt_p(row.get("p_pm", np.nan)),
                q=fmt_p(row.get("q_pm", np.nan)),
                bi=fmt_num(row.get("coef_pm_by_sex_male", np.nan)),
                pi=fmt_p(row.get("p_pm_by_sex_male", np.nan)),
                qi=fmt_p(row.get("q_pm_by_sex_male", np.nan)),
            )
        )
    lines.extend(
        [
            "",
            "## Strongest PM-AM effects",
            "",
            "| Metric | PM-AM beta | p | q |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in ranked.iterrows():
        lines.append(
            "| {metric} | {b} | {p} | {q} |".format(
                metric=row["metric"],
                b=fmt_num(row.get("coef_pm", np.nan)),
                p=fmt_p(row.get("p_pm", np.nan)),
                q=fmt_p(row.get("q_pm", np.nan)),
            )
        )
    (OUTDIR / "am_pm_effects_summary.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate AM/PM effects from YOSHIDAK per-video metrics.")
    parser.add_argument("--outdir", default=str(OUTDIR), help="Directory for AM/PM output files.")
    parser.add_argument(
        "--input",
        default=None,
        help="Input per_video_metrics_robust.csv. Defaults to <outdir>/per_video_metrics_robust.csv.",
    )
    return parser.parse_args()


def main() -> None:
    global INPUT, OUTDIR

    args = parse_args()
    OUTDIR = Path(args.outdir)
    INPUT = Path(args.input) if args.input else OUTDIR / "per_video_metrics_robust.csv"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    per_video = pd.read_csv(INPUT)
    subject_age_ampm = summarize_subject_age_ampm(per_video)
    group_summary = summarize_groups(subject_age_ampm)
    results = fit_am_pm_models(subject_age_ampm)
    core = build_core_table(results)

    subject_age_ampm.to_csv(OUTDIR / "subject_age_am_pm_summary.csv", index=False)
    group_summary.to_csv(OUTDIR / "am_pm_group_summary.csv", index=False)
    results.to_csv(OUTDIR / "am_pm_effects_mixedlm.csv", index=False)
    if "p_pm" in results.columns:
        results.sort_values(["q_pm", "p_pm", "metric"]).to_csv(
            OUTDIR / "am_pm_effects_ranked.csv",
            index=False,
        )
    core.to_csv(OUTDIR / "am_pm_effects_core_table.csv", index=False)
    write_summary(results, core)

    print(f"Saved: {OUTDIR / 'subject_age_am_pm_summary.csv'}")
    print(f"Saved: {OUTDIR / 'am_pm_group_summary.csv'}")
    print(f"Saved: {OUTDIR / 'am_pm_effects_mixedlm.csv'}")
    print(f"Saved: {OUTDIR / 'am_pm_effects_ranked.csv'}")
    print(f"Saved: {OUTDIR / 'am_pm_effects_core_table.csv'}")
    print(f"Saved: {OUTDIR / 'am_pm_effects_summary.md'}")


if __name__ == "__main__":
    main()
