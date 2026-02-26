from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd

from gribcheck.config import PipelineConfig
from gribcheck.date_utils import season_from_date
from gribcheck.fire import assign_fire_proximity_bins, build_daily_fire_index, load_filtered_fire_records
from gribcheck.metrics import compute_regression_metrics

LOGGER = logging.getLogger(__name__)


EPA_REGION_BY_STATE_NAME: dict[str, str] = {
    # Region 1
    "Connecticut": "EPA-1",
    "Maine": "EPA-1",
    "Massachusetts": "EPA-1",
    "New Hampshire": "EPA-1",
    "Rhode Island": "EPA-1",
    "Vermont": "EPA-1",
    # Region 2
    "New Jersey": "EPA-2",
    "New York": "EPA-2",
    "Puerto Rico": "EPA-2",
    "Virgin Islands": "EPA-2",
    # Region 3
    "Delaware": "EPA-3",
    "District Of Columbia": "EPA-3",
    "Maryland": "EPA-3",
    "Pennsylvania": "EPA-3",
    "Virginia": "EPA-3",
    "West Virginia": "EPA-3",
    # Region 4
    "Alabama": "EPA-4",
    "Florida": "EPA-4",
    "Georgia": "EPA-4",
    "Kentucky": "EPA-4",
    "Mississippi": "EPA-4",
    "North Carolina": "EPA-4",
    "South Carolina": "EPA-4",
    "Tennessee": "EPA-4",
    # Region 5
    "Illinois": "EPA-5",
    "Indiana": "EPA-5",
    "Michigan": "EPA-5",
    "Minnesota": "EPA-5",
    "Ohio": "EPA-5",
    "Wisconsin": "EPA-5",
    # Region 6
    "Arkansas": "EPA-6",
    "Louisiana": "EPA-6",
    "New Mexico": "EPA-6",
    "Oklahoma": "EPA-6",
    "Texas": "EPA-6",
    # Region 7
    "Iowa": "EPA-7",
    "Kansas": "EPA-7",
    "Missouri": "EPA-7",
    "Nebraska": "EPA-7",
    # Region 8
    "Colorado": "EPA-8",
    "Montana": "EPA-8",
    "North Dakota": "EPA-8",
    "South Dakota": "EPA-8",
    "Utah": "EPA-8",
    "Wyoming": "EPA-8",
    # Region 9
    "Arizona": "EPA-9",
    "California": "EPA-9",
    "Hawaii": "EPA-9",
    "Nevada": "EPA-9",
    # Region 10
    "Alaska": "EPA-10",
    "Idaho": "EPA-10",
    "Oregon": "EPA-10",
    "Washington": "EPA-10",
}


@dataclass(frozen=True)
class AccuracyStats:
    evaluated_rows: int
    summary_rows: int


def _default_workers(workers: int | None) -> int:
    if workers is not None:
        return max(1, int(workers))
    cpu_count = os.cpu_count() or 2
    return max(1, min(8, cpu_count))


def _summarize_group(df: pd.DataFrame, group_name: str, group_value: str, predictor_col: str) -> dict[str, object]:
    subset = df[["pm25_value", predictor_col]].dropna()
    y_true = subset["pm25_value"].to_numpy(dtype=np.float64)
    y_pred = subset[predictor_col].to_numpy(dtype=np.float64)

    metrics = compute_regression_metrics(y_true=y_true, y_pred=y_pred)
    return {
        "group_name": group_name,
        "group_value": group_value,
        "predictor": predictor_col,
        "n": int(len(subset)),
        **metrics,
    }


def _build_summary(df: pd.DataFrame, predictors: list[str], workers: int) -> pd.DataFrame:
    tasks: list[tuple[pd.DataFrame, str, str, str]] = []

    for predictor in predictors:
        tasks.append((df, "overall", "overall", predictor))

        for year, group in df.groupby("year", dropna=True):
            tasks.append((group, "year", str(int(year)), predictor))

        for season, group in df.groupby("season", dropna=True):
            tasks.append((group, "season", str(season), predictor))

        for state, group in df.groupby("state_name", dropna=True):
            tasks.append((group, "state", str(state), predictor))

        for region, group in df.groupby("epa_region", dropna=True):
            tasks.append((group, "epa_region", str(region), predictor))

        for fire_bin, group in df.groupby("fire_proximity_bin", dropna=True):
            tasks.append((group, "fire_proximity_bin", str(fire_bin), predictor))

    if workers <= 1:
        rows = [_summarize_group(group, gname, gval, pred) for group, gname, gval, pred in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_summarize_group, group, group_name, group_value, predictor)
                for group, group_name, group_value, predictor in tasks
            ]
            rows = [f.result() for f in futures]

    out = pd.DataFrame(rows)
    out = out.sort_values(["predictor", "group_name", "group_value"]).reset_index(drop=True)
    return out


def _quantiles(values: np.ndarray, probs: list[float]) -> list[float]:
    if values.size == 0:
        return [float("nan")] * len(probs)
    return [float(v) for v in np.quantile(values, probs)]


def _distribution_row(y_true: np.ndarray, y_pred: np.ndarray, predictor: str) -> dict[str, object]:
    residual = y_pred - y_true
    true_q = _quantiles(y_true, [0.5, 0.9, 0.95, 0.99])
    pred_q = _quantiles(y_pred, [0.5, 0.9, 0.95, 0.99])
    res_q = _quantiles(residual, [0.5, 0.9, 0.95, 0.99])

    return {
        "predictor": predictor,
        "n": int(y_true.size),
        "pm25_mean": float(np.mean(y_true)) if y_true.size else float("nan"),
        "pm25_std": float(np.std(y_true)) if y_true.size else float("nan"),
        "pm25_p50": true_q[0],
        "pm25_p90": true_q[1],
        "pm25_p95": true_q[2],
        "pm25_p99": true_q[3],
        "pred_mean": float(np.mean(y_pred)) if y_pred.size else float("nan"),
        "pred_std": float(np.std(y_pred)) if y_pred.size else float("nan"),
        "pred_p50": pred_q[0],
        "pred_p90": pred_q[1],
        "pred_p95": pred_q[2],
        "pred_p99": pred_q[3],
        "residual_mean": float(np.mean(residual)) if residual.size else float("nan"),
        "residual_std": float(np.std(residual)) if residual.size else float("nan"),
        "residual_p50": res_q[0],
        "residual_p90": res_q[1],
        "residual_p95": res_q[2],
        "residual_p99": res_q[3],
        "pm25_over_35_frac": float(np.mean(y_true > 35.0)) if y_true.size else float("nan"),
        "pred_over_35_frac": float(np.mean(y_pred > 35.0)) if y_pred.size else float("nan"),
        "pm25_over_55_frac": float(np.mean(y_true > 55.0)) if y_true.size else float("nan"),
        "pred_over_55_frac": float(np.mean(y_pred > 55.0)) if y_pred.size else float("nan"),
    }


def _overall_metrics_with_logs(df: pd.DataFrame, predictors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []

    for predictor in predictors:
        subset = df[["pm25_value", predictor]].dropna()
        y_true = subset["pm25_value"].to_numpy(dtype=np.float64)
        y_pred = subset[predictor].to_numpy(dtype=np.float64)

        raw_metrics = compute_regression_metrics(y_true=y_true, y_pred=y_pred)
        metrics_rows.append(
            {
                "predictor": predictor,
                "transform": "raw",
                "n": int(y_true.size),
                **raw_metrics,
            }
        )
        distribution_rows.append(_distribution_row(y_true=y_true, y_pred=y_pred, predictor=predictor))

        valid_log = (y_true > -0.999999) & (y_pred > -0.999999)
        y_true_log = np.log1p(y_true[valid_log])
        y_pred_log = np.log1p(y_pred[valid_log])
        log_metrics = compute_regression_metrics(y_true=y_true_log, y_pred=y_pred_log)
        metrics_rows.append(
            {
                "predictor": predictor,
                "transform": "log1p",
                "n": int(y_true_log.size),
                **log_metrics,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["predictor", "transform"]).reset_index(drop=True)
    dist_df = pd.DataFrame(distribution_rows).sort_values(["predictor"]).reset_index(drop=True)
    return metrics_df, dist_df


def _console_overall_block(overall_metrics: pd.DataFrame, distribution: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("=== Accuracy Full-Set Metrics ===")
    for _, row in overall_metrics.iterrows():
        lines.append(
            f"{row['predictor']} [{row['transform']}]: n={int(row['n'])}, RMSE={row['rmse']:.4f}, "
            f"Spearman={row['spearman_r']:.4f}, slope={row['slope']:.4f}, "
            f"MAE={row['mae']:.4f}, bias={row['bias']:.4f}, R2={row['r2']:.4f}"
        )

    lines.append("=== Distribution Diagnostics (raw) ===")
    for _, row in distribution.iterrows():
        lines.append(
            f"{row['predictor']}: pm25 p50/p95/p99={row['pm25_p50']:.2f}/{row['pm25_p95']:.2f}/{row['pm25_p99']:.2f}, "
            f"pred p50/p95/p99={row['pred_p50']:.2f}/{row['pred_p95']:.2f}/{row['pred_p99']:.2f}, "
            f"residual mean/std={row['residual_mean']:.2f}/{row['residual_std']:.2f}, "
            f"pm25>35={row['pm25_over_35_frac']:.3f}, pred>35={row['pred_over_35_frac']:.3f}"
        )
    return "\n".join(lines)


def _write_report(summary: pd.DataFrame, overall_metrics: pd.DataFrame, distribution: pd.DataFrame, output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# HRRR vs PM2.5 Accuracy Report")
    lines.append("")

    overall = summary[summary["group_name"] == "overall"].copy()
    lines.append("## Overall")
    lines.append("")
    for _, row in overall.iterrows():
        lines.append(
            f"- `{row['predictor']}`: n={int(row['n'])}, MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}, "
            f"Bias={row['bias']:.4f}, Pearson r={row['pearson_r']:.4f}, Spearman r={row['spearman_r']:.4f}, "
            f"Slope={row['slope']:.4f}, R²={row['r2']:.4f}"
        )

    lines.append("")
    lines.append("## Full-Set Raw And Log1p")
    lines.append("")
    for _, row in overall_metrics.iterrows():
        lines.append(
            f"- `{row['predictor']}` [{row['transform']}]: n={int(row['n'])}, RMSE={row['rmse']:.4f}, "
            f"Spearman r={row['spearman_r']:.4f}, Slope={row['slope']:.4f}, "
            f"MAE={row['mae']:.4f}, Bias={row['bias']:.4f}, R²={row['r2']:.4f}"
        )

    lines.append("")
    lines.append("## Distribution Diagnostics")
    lines.append("")
    for _, row in distribution.iterrows():
        lines.append(
            f"- `{row['predictor']}`: pm25 p50/p95/p99={row['pm25_p50']:.2f}/{row['pm25_p95']:.2f}/{row['pm25_p99']:.2f}, "
            f"pred p50/p95/p99={row['pred_p50']:.2f}/{row['pred_p95']:.2f}/{row['pred_p99']:.2f}, "
            f"residual mean/std={row['residual_mean']:.2f}/{row['residual_std']:.2f}, "
            f"pm25>35={row['pm25_over_35_frac']:.3f}, pred>35={row['pred_over_35_frac']:.3f}, "
            f"pm25>55={row['pm25_over_55_frac']:.3f}, pred>55={row['pred_over_55_frac']:.3f}"
        )

    lines.append("")
    lines.append("## Stratifications")
    lines.append("")
    lines.append("Metrics saved in `accuracy_summary.parquet` for: year, season, state, EPA region, and fire proximity bin.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _maybe_make_plots(df: pd.DataFrame, predictors: list[str], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        LOGGER.warning("matplotlib is not available; skipping accuracy plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for predictor in predictors:
        subset = df[["pm25_value", predictor]].dropna()
        if subset.empty:
            continue

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.scatter(subset["pm25_value"], subset[predictor], s=2, alpha=0.2)
        ax.set_xlabel("PM2.5 (sensor)")
        ax.set_ylabel(predictor)
        ax.set_title(f"Sensor PM2.5 vs {predictor}")
        fig.tight_layout()
        fig.savefig(out_dir / f"scatter_{predictor}.png", dpi=160)
        plt.close(fig)


def run(
    config: PipelineConfig,
    workers: int | None = None,
    from_checkpoint: bool = False,
    checkpoint_dir: Path | None = None,
) -> AccuracyStats:
    input_path = config.paths.station_daily_output
    if from_checkpoint:
        from gribcheck.pipelines import station_daily

        input_path = station_daily.materialize_checkpoint_snapshot(
            config,
            checkpoint_dir=checkpoint_dir,
        )

    df = pd.read_parquet(input_path)
    if df.empty:
        raise ValueError("Joined station dataset is empty; run build-station-hrrr-daily first")
    worker_count = _default_workers(workers)

    usable_massden = int(df["massden_daily_mean"].notna().sum()) if "massden_daily_mean" in df.columns else 0
    usable_colmd = int(df["colmd_daily_mean"].notna().sum()) if "colmd_daily_mean" in df.columns else 0
    LOGGER.info(
        "Accuracy input rows=%d from %s (usable massden=%d, usable colmd=%d)",
        len(df),
        input_path,
        usable_massden,
        usable_colmd,
    )

    df["date_local"] = pd.to_datetime(df["date_local"]).dt.date
    df["year"] = pd.to_datetime(df["date_local"]).dt.year
    df["season"] = df["date_local"].map(season_from_date)
    df["epa_region"] = df["state_name"].astype(str).map(EPA_REGION_BY_STATE_NAME).fillna("EPA-unknown")

    LOGGER.info("Loading wildfire records for proximity bins with %d worker(s)", worker_count)
    fire_records = load_filtered_fire_records(config)
    daily_index = build_daily_fire_index(fire_records)
    df["fire_proximity_bin"] = assign_fire_proximity_bins(df, daily_fires=daily_index, workers=worker_count)

    predictors = ["massden_daily_mean", "colmd_daily_mean"]
    summary = _build_summary(df, predictors=predictors, workers=worker_count)
    overall_metrics, distribution = _overall_metrics_with_logs(df, predictors=predictors)

    config.paths.accuracy_summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(config.paths.accuracy_summary_output, index=False)
    overall_path = config.paths.processed_dir / "accuracy_overall_metrics.parquet"
    distribution_path = config.paths.processed_dir / "accuracy_distribution.parquet"
    overall_metrics.to_parquet(overall_path, index=False)
    distribution.to_parquet(distribution_path, index=False)

    _write_report(summary, overall_metrics, distribution, config.paths.accuracy_report_output)
    _maybe_make_plots(df, predictors=predictors, out_dir=config.paths.figures_dir)

    console_block = _console_overall_block(overall_metrics, distribution)
    print(dedent(console_block))
    LOGGER.info("Wrote overall metrics to %s and distribution diagnostics to %s", overall_path, distribution_path)

    LOGGER.info("Accuracy evaluation complete: %d summary rows", len(summary))
    return AccuracyStats(evaluated_rows=len(df), summary_rows=len(summary))
