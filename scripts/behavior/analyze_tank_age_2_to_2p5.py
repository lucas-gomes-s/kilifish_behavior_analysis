from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

try:
    from kilifish_paths import OUT_V6_YOSHIDAK, PROJECT_ROOT
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from kilifish_paths import OUT_V6_YOSHIDAK, PROJECT_ROOT


ROOT = PROJECT_ROOT
OUTDIR = OUT_V6_YOSHIDAK
INPUT = OUTDIR / "pooled_overlap_subject_age.csv"

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


def lincomb(params: pd.Series, cov: pd.DataFrame, terms: Iterable[str]) -> tuple[float, float, float]:
    terms = list(terms)
    est = float(sum(params.get(t, 0.0) for t in terms))
    try:
        var = 0.0
        for a in terms:
            for b in terms:
                var += float(cov.loc[a, b])
        if not np.isfinite(var) or var < 0:
            return est, np.nan, np.nan
        se = float(np.sqrt(var))
        if se == 0:
            return est, np.nan, np.nan
        p = float(2.0 * stats.norm.sf(abs(est / se)))
        return est, se, p
    except Exception:
        return est, np.nan, np.nan


def fit_models(pooled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keep = pooled.loc[pooled["age_months"].isin([2.0, 2.5])].copy()
    keep["age_step"] = (keep["age_months"].astype(float) - 2.0) / 0.5

    for metric in METRICS:
        if metric not in keep.columns:
            continue
        d = keep[["group_id", "dataset", "tank_small", "sex", "age_months", "age_step", metric]].dropna().copy()
        if d["tank_small"].nunique() < 2 or d["age_months"].nunique() < 2:
            rows.append({"metric": metric, "note": "missing_tank_or_age_level"})
            continue
        if d["group_id"].nunique() < 6:
            rows.append({"metric": metric, "note": "too_few_subjects"})
            continue

        d["y"] = zscore_series(d[metric].astype(float))
        counts = (
            d.groupby(["tank_small", "age_months"])["group_id"]
            .nunique()
            .rename("n")
            .to_dict()
        )

        model = smf.mixedlm(
            "y ~ age_step * tank_small + sex",
            d,
            groups=d["group_id"],
        )
        res = None
        fit_method = None
        errors = []
        for method in [None, "lbfgs", "powell", "cg", "bfgs", "nm"]:
            try:
                kwargs = {"reml": False}
                if method is not None:
                    kwargs["method"] = method
                if method in {"powell", "nm"}:
                    kwargs["maxiter"] = 2000
                res = model.fit(**kwargs)
                fit_method = "default" if method is None else method
                break
            except Exception as exc:
                label = "default" if method is None else method
                errors.append(f"{label}: {exc}")
        if res is None:
            rows.append({"metric": metric, "note": "mixedlm_failed: " + " | ".join(errors)})
            continue

        params = res.params
        pvals = res.pvalues
        cov = res.cov_params()
        small_est, small_se, small_p = lincomb(params, cov, ["age_step", "age_step:tank_small"])

        rows.append(
            {
                "metric": metric,
                "n_rows": int(len(d)),
                "n_subjects": int(d["group_id"].nunique()),
                "n_big": int(d.loc[d["tank_small"].eq(0), "group_id"].nunique()),
                "n_small": int(d.loc[d["tank_small"].eq(1), "group_id"].nunique()),
                "n_big_2mo": int(counts.get((0, 2.0), 0)),
                "n_big_2p5mo": int(counts.get((0, 2.5), 0)),
                "n_small_2mo": int(counts.get((1, 2.0), 0)),
                "n_small_2p5mo": int(counts.get((1, 2.5), 0)),
                "coef_age_step_big": float(params.get("age_step", np.nan)),
                "p_age_step_big": float(pvals.get("age_step", np.nan)),
                "coef_tank_small_at_2mo": float(params.get("tank_small", np.nan)),
                "p_tank_small_at_2mo": float(pvals.get("tank_small", np.nan)),
                "coef_sex_male": float(params.get("sex[T.male]", np.nan)),
                "p_sex_male": float(pvals.get("sex[T.male]", np.nan)),
                "coef_age_step_by_tank_small": float(params.get("age_step:tank_small", np.nan)),
                "p_age_step_by_tank_small": float(pvals.get("age_step:tank_small", np.nan)),
                "coef_age_step_small": small_est,
                "se_age_step_small": small_se,
                "p_age_step_small": small_p,
                "converged": bool(getattr(res, "converged", False)),
                "fit_method": fit_method,
            }
        )

    out = pd.DataFrame(rows)
    for col in [
        "p_age_step_big",
        "p_tank_small_at_2mo",
        "p_age_step_by_tank_small",
        "p_age_step_small",
    ]:
        if col in out.columns:
            out["q" + col[1:]] = bh_qvalues(out[col])
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
        "coef_age_step_big",
        "p_age_step_big",
        "q_age_step_big",
        "coef_age_step_by_tank_small",
        "p_age_step_by_tank_small",
        "q_age_step_by_tank_small",
        "coef_age_step_small",
        "p_age_step_small",
        "q_age_step_small",
        "coef_tank_small_at_2mo",
        "p_tank_small_at_2mo",
        "q_tank_small_at_2mo",
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


def draw_core_tables(core: pd.DataFrame) -> None:
    left = core[["metric_label", "coef_age_step_big", "p_age_step_big"]].copy()
    right = core[["metric_label", "coef_age_step_by_tank_small", "p_age_step_by_tank_small"]].copy()

    def table_data(df: pd.DataFrame, value_col: str, p_col: str) -> list[list[str]]:
        return [[r["metric_label"], fmt_num(r[value_col]), fmt_p(r[p_col])] for _, r in df.iterrows()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    titles = [
        "Large Tank 2.0 -> 2.5 mo Change (beta1)",
        "Small Tank Change Correction (beta3)",
    ]
    tables = [
        table_data(left, "coef_age_step_big", "p_age_step_big"),
        table_data(right, "coef_age_step_by_tank_small", "p_age_step_by_tank_small"),
    ]
    for ax, title, rows in zip(axes, titles, tables):
        ax.axis("off")
        ax.set_title(title, fontsize=13, color="#003366", pad=12)
        tbl = ax.table(
            cellText=rows,
            colLabels=["Metric", "Value", "p-value"],
            colWidths=[0.55, 0.20, 0.20],
            loc="center",
            cellLoc="right",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.5)
        tbl.scale(1.0, 1.35)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_facecolor("#0b6b53")
                cell.set_text_props(color="white", weight="bold")
            elif col == 0:
                cell.set_text_props(ha="left")
            if row > 0:
                p_text = rows[row - 1][2]
                if p_text == "<0.001" or (p_text and float(p_text) < 0.05):
                    cell.set_facecolor("#d8f0d2")
                elif p_text and float(p_text) > 0.3:
                    cell.set_facecolor("#fde0e0")
        ax.text(
            0.0,
            -0.08,
            "Model: z(metric) ~ age_step * tank_small + sex + (1|fish); age_step = 0 at 2.0 mo, 1 at 2.5 mo.",
            transform=ax.transAxes,
            fontsize=8.5,
            ha="left",
            va="top",
        )
    fig.tight_layout()
    fig.savefig(OUTDIR / "tank_age_slope_2_2p5_core_tables.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(results: pd.DataFrame, core: pd.DataFrame) -> None:
    ranked = results.sort_values(["q_age_step_by_tank_small", "p_age_step_by_tank_small", "metric"]).head(12)
    lines = [
        "# Tank-size age-slope analysis: 2.0 vs 2.5 months",
        "",
        "Model: `z(metric) ~ age_step * tank_small + sex + (1|fish)`.",
        "",
        "`age_step = 0` at 2.0 months and `age_step = 1` at 2.5 months, so the age coefficients are standardized changes over the 2.0 -> 2.5 month interval.",
        "",
        "## Core pose metrics",
        "",
        "| Metric | Large-tank change beta1 | p | Small-tank correction beta3 | p | Small-tank total change | p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in core.iterrows():
        lines.append(
            "| {metric} | {b1} | {p1} | {b3} | {p3} | {bs} | {ps} |".format(
                metric=row["metric_label"],
                b1=fmt_num(row["coef_age_step_big"]),
                p1=fmt_p(row["p_age_step_big"]),
                b3=fmt_num(row["coef_age_step_by_tank_small"]),
                p3=fmt_p(row["p_age_step_by_tank_small"]),
                bs=fmt_num(row["coef_age_step_small"]),
                ps=fmt_p(row["p_age_step_small"]),
            )
        )
    lines.extend(
        [
            "",
            "## Strongest tank-differential age changes",
            "",
            "| Metric | beta3 small correction | p | q |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in ranked.iterrows():
        lines.append(
            "| {metric} | {b3} | {p3} | {q3} |".format(
                metric=row["metric"],
                b3=fmt_num(row.get("coef_age_step_by_tank_small", np.nan)),
                p3=fmt_p(row.get("p_age_step_by_tank_small", np.nan)),
                q3=fmt_p(row.get("q_age_step_by_tank_small", np.nan)),
            )
        )
    (OUTDIR / "tank_age_slope_2_2p5_summary.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze 2.0 to 2.5 month tank-age slope effects.")
    parser.add_argument("--outdir", default=str(OUTDIR), help="Directory for tank-age output files.")
    parser.add_argument(
        "--input",
        default=None,
        help="Input pooled_overlap_subject_age.csv. Defaults to <outdir>/pooled_overlap_subject_age.csv.",
    )
    return parser.parse_args()


def main() -> None:
    global INPUT, OUTDIR

    args = parse_args()
    OUTDIR = Path(args.outdir)
    INPUT = Path(args.input) if args.input else OUTDIR / "pooled_overlap_subject_age.csv"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    pooled = pd.read_csv(INPUT)
    results = fit_models(pooled)
    core = build_core_table(results)

    results.to_csv(OUTDIR / "tank_age_slope_2_2p5_mixedlm.csv", index=False)
    results.sort_values(["q_age_step_by_tank_small", "p_age_step_by_tank_small", "metric"]).to_csv(
        OUTDIR / "tank_age_slope_2_2p5_ranked.csv",
        index=False,
    )
    core.to_csv(OUTDIR / "tank_age_slope_2_2p5_core_table.csv", index=False)
    draw_core_tables(core)
    write_summary(results, core)

    print(f"Saved: {OUTDIR / 'tank_age_slope_2_2p5_mixedlm.csv'}")
    print(f"Saved: {OUTDIR / 'tank_age_slope_2_2p5_ranked.csv'}")
    print(f"Saved: {OUTDIR / 'tank_age_slope_2_2p5_core_table.csv'}")
    print(f"Saved: {OUTDIR / 'tank_age_slope_2_2p5_core_tables.png'}")
    print(f"Saved: {OUTDIR / 'tank_age_slope_2_2p5_summary.md'}")


if __name__ == "__main__":
    main()
