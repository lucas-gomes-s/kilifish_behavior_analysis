from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

try:
    from kilifish_paths import (
        OUT_BOUT_PATH_INVARIANCE,
        OUT_V6,
        OUT_V6_OLD_15FPS,
        OUT_V6_YOSHIDAK,
        PROJECT_ROOT,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from kilifish_paths import (
        OUT_BOUT_PATH_INVARIANCE,
        OUT_V6,
        OUT_V6_OLD_15FPS,
        OUT_V6_YOSHIDAK,
        PROJECT_ROOT,
    )


ROOT = PROJECT_ROOT
OUTDIR = OUT_BOUT_PATH_INVARIANCE

DEFAULT_DATASETS = {
    "old15": OUT_V6_OLD_15FPS / "subject_age_summary.csv",
    "v6_original": OUT_V6 / "per_video_metrics_robust.csv",
    "yoshidak": OUT_V6_YOSHIDAK / "subject_age_summary.csv",
}

CORE_METRICS = [
    "total_path_bl",
    "avg_speed_bl_s",
    "bout_freq_per_min",
    "bout_avg_speed_bl_s",
    "bout_peak_speed_bl_s",
    "bout_avg_path_bl",
    "bout_avg_duration_s",
]

STORY_METRICS = [
    "bout_avg_path_bl",
    "bout_avg_speed_bl_s",
    "bout_avg_duration_s",
]

METRIC_LABELS = {
    "total_path_bl": "Total path [BL]",
    "avg_speed_bl_s": "Average speed [BL/s]",
    "bout_freq_per_min": "Bout frequency [bout/min]",
    "bout_avg_speed_bl_s": "Bout speed [BL/s]",
    "bout_peak_speed_bl_s": "Bout peak speed [BL/s]",
    "bout_avg_path_bl": "Bout path [BL]",
    "bout_avg_duration_s": "Bout duration [s]",
}

SENTINEL_FISH = [
    ("female", "female1"),
    ("female", "female2"),
    ("male", "male1"),
    ("male", "male2"),
    ("male", "male9"),
]


@dataclass(frozen=True)
class FitResult:
    dataset: str
    metric: str
    transform: str
    n_rows: int
    n_subjects: int
    age_min: float
    age_max: float
    beta: float
    se: float
    p: float
    model_type: str
    fit_method: str
    converged: bool
    mean_metric: float
    sd_metric: float
    note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build bout path invariance analyses from subject-age killifish metrics."
    )
    parser.add_argument("--outdir", default=str(OUTDIR))
    parser.add_argument(
        "--primary-dataset",
        default="old15",
        help="Dataset used for sentinel, per-fish, and leave-one-fish-out figures.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=CSV",
        help="Additional or replacement dataset. CSV can be subject-age or per-video metrics.",
    )
    parser.add_argument(
        "--only-custom-datasets",
        action="store_true",
        help="Use only datasets passed with --dataset instead of the defaults.",
    )
    parser.add_argument(
        "--equiv-z",
        type=float,
        default=0.10,
        help="TOST equivalence bound in SD units per month.",
    )
    parser.add_argument(
        "--equiv-pct-lifespan",
        type=float,
        default=15.0,
        help="TOST equivalence bound as absolute percent change across the observed age span.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-ages", type=int, default=3)
    return parser.parse_args()


def subject_key(df: pd.DataFrame) -> pd.Series:
    return df["sex"].astype(str) + ":" + df["subject"].astype(str)


def fmt_num(x: float, digits: int = 3) -> str:
    if not np.isfinite(x):
        return ""
    if abs(x) >= 1000 or (abs(x) < 0.001 and x != 0):
        return f"{x:.2e}"
    return f"{x:.{digits}f}"


def fmt_p(x: float) -> str:
    if not np.isfinite(x):
        return ""
    if x < 0.001:
        return "<0.001"
    return f"{x:.3f}"


def zscore(values: pd.Series) -> tuple[pd.Series, float, float]:
    mean = float(values.mean())
    sd = float(values.std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        return values * np.nan, mean, sd
    return (values - mean) / sd, mean, sd


def infer_subject_age_table(df: pd.DataFrame) -> pd.DataFrame:
    if "n_videos" in df.columns and df[["sex", "subject", "age_months"]].duplicated().sum() == 0:
        return df.copy()

    agg: dict[str, str] = {}
    count_col = "video_path" if "video_path" in df.columns else df.columns[0]
    for col in df.columns:
        if col in {"sex", "subject", "age_months"}:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            agg[col] = "median"
    out = (
        df.groupby(["sex", "subject", "age_months"], as_index=False)
        .agg({count_col: "count", **agg})
        .rename(columns={count_col: "n_videos"})
        .sort_values(["sex", "subject", "age_months"])
        .reset_index(drop=True)
    )
    return out


def load_dataset(name: str, path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"sex", "subject", "age_months"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df = infer_subject_age_table(raw)
    df["dataset"] = name
    df["subject_id"] = subject_key(df)
    df["source_csv"] = str(path)
    return df


def load_datasets(args: argparse.Namespace) -> pd.DataFrame:
    entries: dict[str, Path] = {} if args.only_custom_datasets else dict(DEFAULT_DATASETS)
    for item in args.dataset:
        if "=" not in item:
            raise ValueError(f"--dataset must look like NAME=CSV, got {item!r}")
        name, path = item.split("=", 1)
        entries[name] = Path(path).expanduser()

    frames = []
    for name, path in entries.items():
        if not path.exists():
            print(f"Skipping missing dataset {name}: {path}")
            continue
        frames.append(load_dataset(name, path))
    if not frames:
        raise RuntimeError("No datasets were found.")
    return pd.concat(frames, ignore_index=True, sort=False)


def fit_model_with_fallback(d: pd.DataFrame, formula: str):
    groups = d["subject_id"]
    errors = []
    model = smf.mixedlm(formula, d, groups=groups)
    for method in [None, "lbfgs", "powell", "cg", "bfgs", "nm"]:
        try:
            kwargs = {"reml": False}
            if method is not None:
                kwargs["method"] = method
            if method in {"powell", "nm"}:
                kwargs["maxiter"] = 2000
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = model.fit(**kwargs)
            return res, "mixedlm", "default" if method is None else method
        except Exception as exc:
            label = "default" if method is None else method
            errors.append(f"{label}: {exc}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = smf.ols(formula, d).fit(cov_type="HC3")
    res._fallback_errors = " | ".join(errors)  # type: ignore[attr-defined]
    return res, "ols_hc3", "ols_hc3"


def fit_metric_slope(dataset: str, df: pd.DataFrame, metric: str, transform: str) -> FitResult:
    if metric not in df.columns:
        return FitResult(dataset, metric, transform, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan, "", "", False, np.nan, np.nan, "missing_metric")

    d = df[["sex", "subject", "subject_id", "age_months", metric]].dropna().copy()
    d = d.loc[np.isfinite(d["age_months"].astype(float))]
    if transform == "log":
        d = d.loc[d[metric] > 0].copy()
    if d["subject_id"].nunique() < 3 or d["age_months"].nunique() < 2:
        return FitResult(dataset, metric, transform, len(d), d["subject_id"].nunique(), np.nan, np.nan, np.nan, np.nan, np.nan, "", "", False, np.nan, np.nan, "too_few_rows")

    values = d[metric].astype(float)
    mean_metric = float(values.mean())
    sd_metric = float(values.std(ddof=0))
    if transform == "z":
        d["y"], mean_metric, sd_metric = zscore(values)
        if not np.isfinite(sd_metric) or sd_metric == 0:
            return FitResult(dataset, metric, transform, len(d), d["subject_id"].nunique(), np.nan, np.nan, np.nan, np.nan, np.nan, "", "", False, mean_metric, sd_metric, "zero_sd")
    elif transform == "log":
        d["y"] = np.log(values)
    else:
        d["y"] = values

    d["age_c"] = d["age_months"].astype(float) - float(d["age_months"].mean())
    try:
        res, model_type, fit_method = fit_model_with_fallback(d, "y ~ age_c + sex")
        beta = float(res.params.get("age_c", np.nan))
        se = float(res.bse.get("age_c", np.nan))
        p = float(res.pvalues.get("age_c", np.nan))
        note = getattr(res, "_fallback_errors", "")
        return FitResult(
            dataset=dataset,
            metric=metric,
            transform=transform,
            n_rows=int(len(d)),
            n_subjects=int(d["subject_id"].nunique()),
            age_min=float(d["age_months"].min()),
            age_max=float(d["age_months"].max()),
            beta=beta,
            se=se,
            p=p,
            model_type=model_type,
            fit_method=fit_method,
            converged=bool(getattr(res, "converged", True)),
            mean_metric=mean_metric,
            sd_metric=sd_metric,
            note=note,
        )
    except Exception as exc:
        return FitResult(dataset, metric, transform, len(d), d["subject_id"].nunique(), float(d["age_months"].min()), float(d["age_months"].max()), np.nan, np.nan, np.nan, "", "", False, mean_metric, sd_metric, str(exc))


def tost(beta: float, se: float, lower: float, upper: float, alpha: float) -> dict[str, float | bool]:
    if not np.isfinite(beta) or not np.isfinite(se) or se <= 0:
        return {
            "ci90_low": np.nan,
            "ci90_high": np.nan,
            "p_lower": np.nan,
            "p_upper": np.nan,
            "p_tost": np.nan,
            "equivalent": False,
        }
    zcrit = stats.norm.ppf(1.0 - alpha)
    ci90_low = beta - zcrit * se
    ci90_high = beta + zcrit * se
    p_lower = float(stats.norm.sf((beta - lower) / se))
    p_upper = float(stats.norm.cdf((beta - upper) / se))
    p_tost = max(p_lower, p_upper)
    return {
        "ci90_low": float(ci90_low),
        "ci90_high": float(ci90_high),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": p_tost,
        "equivalent": bool(p_tost < alpha),
    }


def build_equivalence_tests(all_data: pd.DataFrame, equiv_z: float, equiv_pct_lifespan: float, alpha: float) -> pd.DataFrame:
    rows = []
    for dataset, df in all_data.groupby("dataset", sort=False):
        for metric in CORE_METRICS:
            fit = fit_metric_slope(dataset, df, metric, "z")
            age_span = fit.age_max - fit.age_min if np.isfinite(fit.age_max) and np.isfinite(fit.age_min) else np.nan
            z_eq = tost(fit.beta, fit.se, -equiv_z, equiv_z, alpha)

            raw_beta = fit.beta * fit.sd_metric if np.isfinite(fit.beta) and np.isfinite(fit.sd_metric) else np.nan
            raw_se = fit.se * fit.sd_metric if np.isfinite(fit.se) and np.isfinite(fit.sd_metric) else np.nan
            pct_beta = raw_beta / fit.mean_metric * 100.0 if np.isfinite(raw_beta) and fit.mean_metric != 0 else np.nan
            pct_se = raw_se / fit.mean_metric * 100.0 if np.isfinite(raw_se) and fit.mean_metric != 0 else np.nan
            pct_bound = equiv_pct_lifespan / age_span if np.isfinite(age_span) and age_span > 0 else np.nan
            pct_eq = tost(pct_beta, pct_se, -pct_bound, pct_bound, alpha)

            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "n_rows": fit.n_rows,
                    "n_subjects": fit.n_subjects,
                    "age_min": fit.age_min,
                    "age_max": fit.age_max,
                    "age_span": age_span,
                    "beta_z_per_month": fit.beta,
                    "se_z_per_month": fit.se,
                    "p_age": fit.p,
                    "ci90_low_z": z_eq["ci90_low"],
                    "ci90_high_z": z_eq["ci90_high"],
                    "equiv_bound_z_per_month": equiv_z,
                    "p_tost_z": z_eq["p_tost"],
                    "equivalent_z": z_eq["equivalent"],
                    "mean_metric": fit.mean_metric,
                    "sd_metric": fit.sd_metric,
                    "beta_raw_per_month": raw_beta,
                    "beta_pct_mean_per_month": pct_beta,
                    "se_pct_mean_per_month": pct_se,
                    "equiv_bound_pct_per_month": pct_bound,
                    "p_tost_pct": pct_eq["p_tost"],
                    "equivalent_pct": pct_eq["equivalent"],
                    "model_type": fit.model_type,
                    "fit_method": fit.fit_method,
                    "converged": fit.converged,
                    "note": fit.note,
                }
            )
    return pd.DataFrame(rows)


def fit_per_fish_slopes(all_data: pd.DataFrame, min_ages: int) -> pd.DataFrame:
    rows = []
    for (dataset, sex, subject), g in all_data.groupby(["dataset", "sex", "subject"], sort=False):
        subject_id = f"{sex}:{subject}"
        for metric in CORE_METRICS:
            if metric not in g.columns:
                continue
            d = g[["age_months", metric]].dropna().sort_values("age_months")
            if len(d) < min_ages or d["age_months"].nunique() < min_ages:
                continue
            x = d["age_months"].astype(float).to_numpy()
            y = d[metric].astype(float).to_numpy()
            if not np.all(np.isfinite(y)):
                continue
            lr = stats.linregress(x, y)
            mean_y = float(np.mean(y))
            age_span = float(np.max(x) - np.min(x))
            rows.append(
                {
                    "dataset": dataset,
                    "sex": sex,
                    "subject": subject,
                    "subject_id": subject_id,
                    "metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "n_ages": int(len(d)),
                    "age_min": float(np.min(x)),
                    "age_max": float(np.max(x)),
                    "age_span": age_span,
                    "mean_metric": mean_y,
                    "slope_raw_per_month": float(lr.slope),
                    "se_raw_per_month": float(lr.stderr) if lr.stderr is not None else np.nan,
                    "p_slope": float(lr.pvalue),
                    "r2": float(lr.rvalue**2),
                    "slope_pct_mean_per_month": float(lr.slope / mean_y * 100.0) if mean_y != 0 else np.nan,
                    "total_pct_over_observed_span": float(lr.slope * age_span / mean_y * 100.0) if mean_y != 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_per_fish_slopes(slopes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, metric), g in slopes.groupby(["dataset", "metric"], sort=False):
        vals = g["slope_pct_mean_per_month"].replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            continue
        try:
            wilcoxon_p = float(stats.wilcoxon(vals).pvalue) if len(vals) >= 3 and not np.allclose(vals, 0) else np.nan
        except Exception:
            wilcoxon_p = np.nan
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "metric_label": METRIC_LABELS.get(metric, metric),
                "n_fish": int(len(vals)),
                "median_pct_per_month": float(vals.median()),
                "iqr_low_pct_per_month": float(vals.quantile(0.25)),
                "iqr_high_pct_per_month": float(vals.quantile(0.75)),
                "mean_abs_pct_per_month": float(vals.abs().mean()),
                "median_abs_pct_per_month": float(vals.abs().median()),
                "wilcoxon_p_vs_zero": wilcoxon_p,
            }
        )
    return pd.DataFrame(rows)


def build_decomposition(all_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for dataset, df in all_data.groupby("dataset", sort=False):
        for metric in STORY_METRICS:
            fit = fit_metric_slope(dataset, df, metric, "log")
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "beta_log_per_month": fit.beta,
                    "se_log_per_month": fit.se,
                    "p_age": fit.p,
                    "n_rows": fit.n_rows,
                    "n_subjects": fit.n_subjects,
                    "age_min": fit.age_min,
                    "age_max": fit.age_max,
                    "model_type": fit.model_type,
                    "fit_method": fit.fit_method,
                    "note": fit.note,
                }
            )
    detail = pd.DataFrame(rows)
    summary_rows = []
    for dataset, g in detail.groupby("dataset", sort=False):
        lookup = g.set_index("metric")
        if not set(STORY_METRICS).issubset(lookup.index):
            continue
        path = float(lookup.loc["bout_avg_path_bl", "beta_log_per_month"])
        speed = float(lookup.loc["bout_avg_speed_bl_s", "beta_log_per_month"])
        duration = float(lookup.loc["bout_avg_duration_s", "beta_log_per_month"])
        speed_plus_duration = speed + duration
        denom = abs(speed) + abs(duration)
        cancellation_fraction = 1.0 - abs(speed_plus_duration) / denom if denom > 0 else np.nan
        summary_rows.append(
            {
                "dataset": dataset,
                "path_log_slope_per_month": path,
                "speed_log_slope_per_month": speed,
                "duration_log_slope_per_month": duration,
                "speed_plus_duration_log_slope_per_month": speed_plus_duration,
                "path_minus_speed_plus_duration": path - speed_plus_duration,
                "speed_duration_opposite_signs": bool(speed * duration < 0),
                "cancellation_fraction": cancellation_fraction,
            }
        )
    return detail, pd.DataFrame(summary_rows)


def build_leave_one_fish_out(primary: pd.DataFrame, dataset: str, equiv_z: float, alpha: float) -> pd.DataFrame:
    rows = []
    subjects = sorted(primary["subject_id"].dropna().unique().tolist())
    targets: list[tuple[str, pd.DataFrame]] = [("NONE_FULL_MODEL", primary)]
    targets.extend((subject, primary.loc[primary["subject_id"] != subject].copy()) for subject in subjects)
    for omitted, df in targets:
        fit = fit_metric_slope(dataset, df, "bout_avg_path_bl", "z")
        eq = tost(fit.beta, fit.se, -equiv_z, equiv_z, alpha)
        rows.append(
            {
                "dataset": dataset,
                "omitted_subject_id": omitted,
                "n_rows": fit.n_rows,
                "n_subjects": fit.n_subjects,
                "beta_z_per_month": fit.beta,
                "se_z_per_month": fit.se,
                "p_age": fit.p,
                "ci90_low_z": eq["ci90_low"],
                "ci90_high_z": eq["ci90_high"],
                "equiv_bound_z_per_month": equiv_z,
                "p_tost_z": eq["p_tost"],
                "equivalent_z": eq["equivalent"],
                "model_type": fit.model_type,
                "fit_method": fit.fit_method,
                "note": fit.note,
            }
        )
    return pd.DataFrame(rows)


def draw_sentinel_raw(primary: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(len(SENTINEL_FISH), len(STORY_METRICS), figsize=(11, 11), squeeze=False)
    for row_idx, (sex, subject) in enumerate(SENTINEL_FISH):
        g = primary.loc[(primary["sex"] == sex) & (primary["subject"] == subject)].sort_values("age_months")
        for col_idx, metric in enumerate(STORY_METRICS):
            ax = axes[row_idx][col_idx]
            if metric in g.columns and not g.empty:
                ax.plot(g["age_months"], g[metric], marker="o", lw=1.8)
            ax.axhline(0, color="#cccccc", lw=0.8)
            if row_idx == 0:
                ax.set_title(METRIC_LABELS[metric], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{sex} {subject.replace(sex, '')}".strip(), fontsize=9)
            if row_idx == len(SENTINEL_FISH) - 1:
                ax.set_xlabel("Age [months]")
            ax.grid(True, color="#e6e6e6", linewidth=0.6)
    fig.suptitle("Sentinel fish raw trajectories", y=0.995, fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "sentinel_raw_trajectories.png", dpi=220)
    plt.close(fig)


def draw_sentinel_relative(primary: pd.DataFrame, outdir: Path) -> None:
    colors = {
        "bout_avg_path_bl": "#0b6b53",
        "bout_avg_speed_bl_s": "#3b5ba9",
        "bout_avg_duration_s": "#b6542d",
    }
    fig, axes = plt.subplots(len(SENTINEL_FISH), 1, figsize=(8.5, 10), sharex=True)
    for ax, (sex, subject) in zip(axes, SENTINEL_FISH):
        g = primary.loc[(primary["sex"] == sex) & (primary["subject"] == subject)].sort_values("age_months")
        for metric in STORY_METRICS:
            if metric not in g.columns:
                continue
            d = g[["age_months", metric]].dropna()
            if d.empty:
                continue
            base = float(d.iloc[0][metric])
            if base == 0 or not np.isfinite(base):
                continue
            ax.plot(
                d["age_months"],
                d[metric] / base,
                marker="o",
                lw=1.8,
                color=colors[metric],
                label=METRIC_LABELS[metric],
            )
        ax.axhline(1.0, color="#999999", lw=0.9, linestyle="--")
        ax.set_ylabel(f"{sex} {subject.replace(sex, '')}".strip(), fontsize=9)
        ax.grid(True, color="#e6e6e6", linewidth=0.6)
    axes[0].legend(loc="upper right", ncols=3, fontsize=8)
    axes[-1].set_xlabel("Age [months]")
    fig.suptitle("Sentinel fish trajectories relative to first age", y=0.995, fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "sentinel_relative_trajectories.png", dpi=220)
    plt.close(fig)


def draw_slope_distribution(slopes: pd.DataFrame, primary_dataset: str, outdir: Path) -> None:
    d = slopes.loc[(slopes["dataset"] == primary_dataset) & (slopes["metric"].isin(STORY_METRICS))].copy()
    if d.empty:
        return
    order = STORY_METRICS
    data = [d.loc[d["metric"] == metric, "slope_pct_mean_per_month"].dropna().to_numpy() for metric in order]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bp = ax.boxplot(data, labels=[METRIC_LABELS[m] for m in order], patch_artist=True, showfliers=False)
    fills = ["#bfe5d7", "#cdd7f0", "#f2cdbd"]
    for patch, color in zip(bp["boxes"], fills):
        patch.set_facecolor(color)
        patch.set_edgecolor("#555555")
    rng = np.random.default_rng(7)
    for idx, vals in enumerate(data, start=1):
        jitter = rng.normal(0.0, 0.035, size=len(vals))
        ax.scatter(np.full(len(vals), idx) + jitter, vals, color="#222222", s=22, alpha=0.75, zorder=3)
    ax.axhline(0, color="#222222", lw=1.0)
    ax.set_ylabel("Per-fish slope [% of fish mean per month]")
    ax.set_title(f"Per-fish age slopes: {primary_dataset}")
    ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(outdir / "per_fish_slope_distribution.png", dpi=220)
    plt.close(fig)


def draw_equivalence_forest(equiv: pd.DataFrame, primary_dataset: str, outdir: Path) -> None:
    d = equiv.loc[(equiv["dataset"] == primary_dataset) & (equiv["metric"].isin(CORE_METRICS))].copy()
    if d.empty:
        return
    d["order"] = d["metric"].map({m: i for i, m in enumerate(CORE_METRICS)})
    d = d.sort_values("order")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 5.2))
    x = d["beta_z_per_month"].to_numpy()
    xerr_low = x - d["ci90_low_z"].to_numpy()
    xerr_high = d["ci90_high_z"].to_numpy() - x
    ax.errorbar(x, y, xerr=[xerr_low, xerr_high], fmt="o", color="#0b6b53", ecolor="#555555", capsize=3)
    bound = float(d["equiv_bound_z_per_month"].dropna().iloc[0])
    ax.axvline(-bound, color="#b6542d", linestyle="--", lw=1)
    ax.axvline(bound, color="#b6542d", linestyle="--", lw=1)
    ax.axvline(0, color="#222222", lw=0.8)
    ax.set_yticks(y, d["metric_label"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("Mixed-model age slope [SD/month], 90% CI")
    ax.set_title(f"Equivalence test against +/-{bound:g} SD/month: {primary_dataset}")
    ax.grid(True, axis="x", color="#e6e6e6", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(outdir / "equivalence_forest_core_metrics.png", dpi=220)
    plt.close(fig)


def draw_decomposition(decomp_summary: pd.DataFrame, outdir: Path) -> None:
    if decomp_summary.empty:
        return
    fig, axes = plt.subplots(len(decomp_summary), 1, figsize=(8, max(3, 2.2 * len(decomp_summary))), squeeze=False)
    for ax, (_, row) in zip(axes[:, 0], decomp_summary.iterrows()):
        labels = ["speed", "duration", "speed+duration", "path"]
        values = [
            row["speed_log_slope_per_month"],
            row["duration_log_slope_per_month"],
            row["speed_plus_duration_log_slope_per_month"],
            row["path_log_slope_per_month"],
        ]
        colors = ["#3b5ba9", "#b6542d", "#777777", "#0b6b53"]
        ax.bar(labels, values, color=colors)
        ax.axhline(0, color="#222222", lw=0.9)
        ax.set_ylabel("log slope/month")
        ax.set_title(str(row["dataset"]))
        ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.6)
    fig.suptitle("Speed-duration decomposition of bout path", y=0.995, fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "speed_duration_log_decomposition.png", dpi=220)
    plt.close(fig)


def draw_leave_one_out(loo: pd.DataFrame, outdir: Path) -> None:
    d = loo.copy()
    if d.empty:
        return
    d = d.sort_values("beta_z_per_month")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(d))))
    x = d["beta_z_per_month"].to_numpy()
    xerr_low = x - d["ci90_low_z"].to_numpy()
    xerr_high = d["ci90_high_z"].to_numpy() - x
    colors = np.where(d["omitted_subject_id"].eq("NONE_FULL_MODEL"), "#0b6b53", "#555555")
    ax.errorbar(x, y, xerr=[xerr_low, xerr_high], fmt="none", ecolor="#777777", capsize=2)
    ax.scatter(x, y, color=colors, s=28, zorder=3)
    bound = float(d["equiv_bound_z_per_month"].dropna().iloc[0])
    ax.axvline(-bound, color="#b6542d", linestyle="--", lw=1)
    ax.axvline(bound, color="#b6542d", linestyle="--", lw=1)
    ax.axvline(0, color="#222222", lw=0.8)
    ax.set_yticks(y, d["omitted_subject_id"].tolist())
    ax.set_xlabel("Bout path age slope [SD/month], 90% CI")
    ax.set_title("Leave-one-fish-out bout path equivalence")
    ax.grid(True, axis="x", color="#e6e6e6", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(outdir / "leave_one_fish_out_bout_path.png", dpi=220)
    plt.close(fig)


def write_summary(
    outdir: Path,
    primary_dataset: str,
    equiv: pd.DataFrame,
    slope_summary: pd.DataFrame,
    decomp_summary: pd.DataFrame,
    loo: pd.DataFrame,
) -> None:
    lines = [
        "# Bout path invariance analysis",
        "",
        f"Primary dataset: `{primary_dataset}`.",
        "",
        "The invariant claim is strongest when the bout path age slope is both statistically small by TOST and visually stable within fish, while bout speed and bout duration move in opposite directions.",
        "",
        "## Core equivalence results",
        "",
        "| Dataset | Metric | beta z/month | p(age) | 90% CI | TOST p | Equivalent +/-0.10 SD/mo | beta %mean/mo | Equivalent 15% span |",
        "|---|---|---:|---:|---:|---:|---|---:|---|",
    ]
    core = equiv.loc[equiv["metric"].isin(STORY_METRICS)].copy()
    core["dataset_order"] = core["dataset"].eq(primary_dataset).map({True: 0, False: 1})
    core["metric_order"] = core["metric"].map({m: i for i, m in enumerate(STORY_METRICS)})
    for _, row in core.sort_values(["dataset_order", "dataset", "metric_order"]).iterrows():
        lines.append(
            "| {dataset} | {metric} | {beta} | {p_age} | [{lo}, {hi}] | {p_tost} | {equiv_z} | {pct} | {equiv_pct} |".format(
                dataset=row["dataset"],
                metric=row["metric_label"],
                beta=fmt_num(row["beta_z_per_month"]),
                p_age=fmt_p(row["p_age"]),
                lo=fmt_num(row["ci90_low_z"]),
                hi=fmt_num(row["ci90_high_z"]),
                p_tost=fmt_p(row["p_tost_z"]),
                equiv_z="yes" if bool(row["equivalent_z"]) else "no",
                pct=fmt_num(row["beta_pct_mean_per_month"]),
                equiv_pct="yes" if bool(row["equivalent_pct"]) else "no",
            )
        )

    lines.extend(["", "## Per-fish slope distribution", "", "| Dataset | Metric | n fish | median %/mo | IQR %/mo | median abs %/mo |", "|---|---|---:|---:|---:|---:|"])
    sf = slope_summary.loc[slope_summary["metric"].isin(STORY_METRICS)].copy()
    sf["dataset_order"] = sf["dataset"].eq(primary_dataset).map({True: 0, False: 1})
    sf["metric_order"] = sf["metric"].map({m: i for i, m in enumerate(STORY_METRICS)})
    for _, row in sf.sort_values(["dataset_order", "dataset", "metric_order"]).iterrows():
        lines.append(
            "| {dataset} | {metric} | {n} | {med} | [{q1}, {q3}] | {abs_med} |".format(
                dataset=row["dataset"],
                metric=row["metric_label"],
                n=int(row["n_fish"]),
                med=fmt_num(row["median_pct_per_month"]),
                q1=fmt_num(row["iqr_low_pct_per_month"]),
                q3=fmt_num(row["iqr_high_pct_per_month"]),
                abs_med=fmt_num(row["median_abs_pct_per_month"]),
            )
        )

    lines.extend(["", "## Speed-duration decomposition", "", "| Dataset | speed log slope | duration log slope | speed+duration | path log slope | cancellation fraction |", "|---|---:|---:|---:|---:|---:|"])
    for _, row in decomp_summary.iterrows():
        lines.append(
            "| {dataset} | {speed} | {duration} | {summed} | {path} | {cancel} |".format(
                dataset=row["dataset"],
                speed=fmt_num(row["speed_log_slope_per_month"]),
                duration=fmt_num(row["duration_log_slope_per_month"]),
                summed=fmt_num(row["speed_plus_duration_log_slope_per_month"]),
                path=fmt_num(row["path_log_slope_per_month"]),
                cancel=fmt_num(row["cancellation_fraction"]),
            )
        )

    if not loo.empty:
        loo_nonfull = loo.loc[~loo["omitted_subject_id"].eq("NONE_FULL_MODEL")]
        full = loo.loc[loo["omitted_subject_id"].eq("NONE_FULL_MODEL")].head(1)
        lines.extend(["", "## Leave-one-fish-out", ""])
        if not full.empty:
            row = full.iloc[0]
            lines.append(
                "Full model bout path slope: {beta} SD/month, 90% CI [{lo}, {hi}], TOST p={p_tost}, equivalent={equiv}.".format(
                    beta=fmt_num(row["beta_z_per_month"]),
                    lo=fmt_num(row["ci90_low_z"]),
                    hi=fmt_num(row["ci90_high_z"]),
                    p_tost=fmt_p(row["p_tost_z"]),
                    equiv="yes" if bool(row["equivalent_z"]) else "no",
                )
            )
        if not loo_nonfull.empty:
            lines.append(
                "Leave-one-out slope range: {lo} to {hi} SD/month; {n_eq}/{n} leave-one-out fits pass the +/-0.10 SD/month TOST.".format(
                    lo=fmt_num(loo_nonfull["beta_z_per_month"].min()),
                    hi=fmt_num(loo_nonfull["beta_z_per_month"].max()),
                    n_eq=int(loo_nonfull["equivalent_z"].sum()),
                    n=len(loo_nonfull),
                )
            )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `equivalence_tests.csv`: mixed-model age slopes and TOST equivalence tests.",
            "- `per_fish_slopes.csv`: simple within-fish slopes for each metric.",
            "- `per_fish_slope_summary.csv`: distribution summary for per-fish slopes.",
            "- `log_decomposition_slopes.csv` and `log_decomposition_summary.csv`: log-space speed-duration cancellation.",
            "- `leave_one_fish_out_bout_path.csv`: influence check for the bout path slope.",
            "- PNG figures: sentinel trajectories, slope distribution, equivalence forest, log decomposition, and leave-one-out.",
        ]
    )
    (outdir / "bout_path_invariance_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_data = load_datasets(args)
    all_data.to_csv(outdir / "combined_subject_age_metrics.csv", index=False)

    dataset_overview = (
        all_data.groupby("dataset", as_index=False)
        .agg(
            n_rows=("subject_id", "size"),
            n_subjects=("subject_id", "nunique"),
            age_min=("age_months", "min"),
            age_max=("age_months", "max"),
            source_csv=("source_csv", "first"),
        )
        .sort_values("dataset")
    )
    dataset_overview.to_csv(outdir / "dataset_overview.csv", index=False)

    if args.primary_dataset not in set(all_data["dataset"]):
        raise RuntimeError(f"Primary dataset {args.primary_dataset!r} was not loaded.")
    primary = all_data.loc[all_data["dataset"] == args.primary_dataset].copy()

    equiv = build_equivalence_tests(all_data, args.equiv_z, args.equiv_pct_lifespan, args.alpha)
    equiv.to_csv(outdir / "equivalence_tests.csv", index=False)

    per_fish = fit_per_fish_slopes(all_data, args.min_ages)
    per_fish.to_csv(outdir / "per_fish_slopes.csv", index=False)
    slope_summary = summarize_per_fish_slopes(per_fish)
    slope_summary.to_csv(outdir / "per_fish_slope_summary.csv", index=False)

    decomp_detail, decomp_summary = build_decomposition(all_data)
    decomp_detail.to_csv(outdir / "log_decomposition_slopes.csv", index=False)
    decomp_summary.to_csv(outdir / "log_decomposition_summary.csv", index=False)

    loo = build_leave_one_fish_out(primary, args.primary_dataset, args.equiv_z, args.alpha)
    loo.to_csv(outdir / "leave_one_fish_out_bout_path.csv", index=False)

    draw_sentinel_raw(primary, outdir)
    draw_sentinel_relative(primary, outdir)
    draw_slope_distribution(per_fish, args.primary_dataset, outdir)
    draw_equivalence_forest(equiv, args.primary_dataset, outdir)
    draw_decomposition(decomp_summary, outdir)
    draw_leave_one_out(loo, outdir)

    write_summary(outdir, args.primary_dataset, equiv, slope_summary, decomp_summary, loo)

    print(f"Saved combined data: {outdir / 'combined_subject_age_metrics.csv'}")
    print(f"Saved equivalence tests: {outdir / 'equivalence_tests.csv'}")
    print(f"Saved per-fish slopes: {outdir / 'per_fish_slopes.csv'}")
    print(f"Saved summary: {outdir / 'bout_path_invariance_summary.md'}")


if __name__ == "__main__":
    main()
