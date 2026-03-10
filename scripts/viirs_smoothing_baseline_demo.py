#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from gribcheck.config import PipelineConfig, load_config
from gribcheck.fire import load_filtered_fire_records
from gribcheck.geo_utils import hrrr_transformer_to_xy
from gribcheck.hrrr import HRRRAnalysisReader
from gribcheck.models import FireRecord, VariableSpec
from gribcheck.viirs import load_or_build_viirs_hourly_points


@dataclass
class SmallGrowthCase:
    fire: FireRecord
    day: date
    today_count: int
    next_count: int
    growth_ratio: float
    score: float


@dataclass(frozen=True)
class GridSpec:
    xmin: float
    ymin: float
    nx: int
    ny: int
    pixel_size_m: float

    @property
    def xmax(self) -> float:
        return self.xmin + self.nx * self.pixel_size_m

    @property
    def ymax(self) -> float:
        return self.ymin + self.ny * self.pixel_size_m


def _utc_datetime(day_utc: date, hour_utc: int) -> datetime:
    return datetime.combine(day_utc, time(hour_utc, 0), tzinfo=timezone.utc)


def _build_bounds_xy(rec: FireRecord, buffer_km: float) -> tuple[float, float, float, float]:
    transformer = hrrr_transformer_to_xy()
    x0, y0 = transformer.transform(rec.min_lon, rec.min_lat)
    x1, y1 = transformer.transform(rec.max_lon, rec.max_lat)
    xmin = min(x0, x1)
    xmax = max(x0, x1)
    ymin = min(y0, y1)
    ymax = max(y0, y1)
    buffer_m = buffer_km * 1000.0
    return (xmin - buffer_m, ymin - buffer_m, xmax + buffer_m, ymax + buffer_m)


def _subset_to_bounds(field, bounds_xy):
    xmin, ymin, xmax, ymax = bounds_xy
    x_vals = field["x"].to_numpy()
    y_vals = field["y"].to_numpy()
    x_slice = slice(xmin, xmax) if x_vals[0] <= x_vals[-1] else slice(xmax, xmin)
    y_slice = slice(ymin, ymax) if y_vals[0] <= y_vals[-1] else slice(ymax, ymin)
    subset = field.sel(x=x_slice, y=y_slice)
    if subset.size == 0:
        return None
    return subset


def _build_fixed_grid(
    bounds_xy: tuple[float, float, float, float],
    pixel_size_m: float,
    max_grid_dim: int,
) -> GridSpec | None:
    xmin, ymin, xmax, ymax = bounds_xy
    width = max(float(xmax - xmin), float(pixel_size_m))
    height = max(float(ymax - ymin), float(pixel_size_m))
    nx = int(np.ceil(width / float(pixel_size_m)))
    ny = int(np.ceil(height / float(pixel_size_m)))
    if nx < 2 or ny < 2:
        return None
    if nx > int(max_grid_dim) or ny > int(max_grid_dim):
        return None
    return GridSpec(
        xmin=float(xmin),
        ymin=float(ymin),
        nx=int(nx),
        ny=int(ny),
        pixel_size_m=float(pixel_size_m),
    )


def _resample_to_grid(field, grid: GridSpec) -> np.ndarray | None:
    if field is None or field.size == 0:
        return None
    x_vals = field["x"].to_numpy()
    y_vals = field["y"].to_numpy()
    if len(x_vals) < 2 or len(y_vals) < 2:
        return None

    x_new = grid.xmin + (np.arange(grid.nx, dtype=np.float64) + 0.5) * grid.pixel_size_m
    y_new = grid.ymin + (np.arange(grid.ny, dtype=np.float64) + 0.5) * grid.pixel_size_m
    interp = field.interp(x=x_new, y=y_new, method="linear")
    arr = interp.to_numpy().astype(np.float32, copy=False)
    if arr.shape != (grid.ny, grid.nx):
        return None
    return arr


def _extract_hrrr_patch(
    reader: HRRRAnalysisReader,
    run_time_utc: datetime,
    spec: VariableSpec,
    bounds_xy: tuple[float, float, float, float],
    grid: GridSpec,
) -> np.ndarray | None:
    da = reader.load_field(run_time_utc, spec)
    if da is None:
        return None
    subset = _subset_to_bounds(da, bounds_xy)
    return _resample_to_grid(subset, grid=grid)


def _shift_with_zero_fill(arr: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    shifted = np.roll(arr, shift=(shift_y, shift_x), axis=(0, 1))
    if shift_y > 0:
        shifted[:shift_y, :] = 0.0
    elif shift_y < 0:
        shifted[shift_y:, :] = 0.0
    if shift_x > 0:
        shifted[:, :shift_x] = 0.0
    elif shift_x < 0:
        shifted[:, shift_x:] = 0.0
    return shifted


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    maxv = float(np.nanmax(arr))
    if not np.isfinite(maxv) or maxv <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / maxv, 0.0, 1.0)


def _rasterize_day_with_sample_hours(
    viirs_xyf_df: pd.DataFrame,
    sample_day: date,
    sample_hours_utc: tuple[int, ...],
    grid: GridSpec,
) -> np.ndarray:
    agg = np.zeros((grid.ny, grid.nx), dtype=np.float32)
    day_mask = viirs_xyf_df["day_utc"] == sample_day
    if not bool(np.any(day_mask)):
        return agg

    hour_mask = viirs_xyf_df["hour_utc"].isin(list(sample_hours_utc))
    sub = viirs_xyf_df.loc[day_mask & hour_mask, ["x", "y", "frp"]]
    if sub.empty:
        return agg

    x = sub["x"].to_numpy(dtype=np.float64)
    y = sub["y"].to_numpy(dtype=np.float64)
    frp = sub["frp"].to_numpy(dtype=np.float32)

    in_bounds = (x >= grid.xmin) & (x < grid.xmax) & (y >= grid.ymin) & (y < grid.ymax)
    if not np.any(in_bounds):
        return agg
    x = x[in_bounds]
    y = y[in_bounds]
    frp = frp[in_bounds]

    ix = np.floor((x - grid.xmin) / grid.pixel_size_m).astype(np.int32)
    iy = np.floor((y - grid.ymin) / grid.pixel_size_m).astype(np.int32)
    ix = np.clip(ix, 0, grid.nx - 1)
    iy = np.clip(iy, 0, grid.ny - 1)
    np.add.at(agg, (iy, ix), frp)
    return agg


def _find_small_growth_case(
    records: list[FireRecord],
    viirs_xy_day_df: pd.DataFrame,
    buffer_km: float,
    sample_hours_utc: tuple[int, ...],
    pixel_size_m: float,
    max_grid_dim: int,
    target_growth: float,
    min_daily_count: int,
    max_growth_ratio: float,
    include_non_conus: bool,
) -> SmallGrowthCase:
    x_all = viirs_xy_day_df["x"].to_numpy(dtype=np.float64)
    y_all = viirs_xy_day_df["y"].to_numpy(dtype=np.float64)
    day_ord_all = viirs_xy_day_df["day_utc"].map(date.toordinal).to_numpy(dtype=np.int32)
    hour_all = viirs_xy_day_df["hour_utc"].to_numpy(dtype=np.int16)
    allowed_hours = set(int(h) for h in sample_hours_utc)

    best: SmallGrowthCase | None = None

    # Prefer medium-duration fires to avoid pathological timelines.
    ordered_records = sorted(records, key=lambda r: ((r.end_date - r.start_date).days, -r.size_acres))

    for rec in ordered_records:
        if not include_non_conus and (rec.state or "").upper() in {"US-AK", "US-HI"}:
            continue
        bounds = _build_bounds_xy(rec, buffer_km=buffer_km)
        grid = _build_fixed_grid(bounds, pixel_size_m=pixel_size_m, max_grid_dim=max_grid_dim)
        if grid is None:
            continue
        xmin, ymin, xmax, ymax = bounds
        in_bbox = (x_all >= xmin) & (x_all <= xmax) & (y_all >= ymin) & (y_all <= ymax)
        if int(np.count_nonzero(in_bbox)) < (min_daily_count * 2):
            continue

        in_hours = np.isin(hour_all, list(allowed_hours))
        in_bbox = in_bbox & in_hours
        day_ord = day_ord_all[in_bbox]
        if day_ord.size == 0:
            continue

        start_ord = rec.start_date.toordinal()
        end_ord = rec.end_date.toordinal()
        in_window = (day_ord >= start_ord) & (day_ord <= end_ord + 1)
        day_ord = day_ord[in_window]
        if day_ord.size == 0:
            continue

        counts = np.bincount(day_ord - start_ord, minlength=(end_ord - start_ord + 2))
        for idx in range(0, len(counts) - 1):
            c0 = int(counts[idx])
            c1 = int(counts[idx + 1])
            if c0 < min_daily_count or c1 < min_daily_count:
                continue

            growth = float(c1 - c0) / float(c0)
            if growth <= 0.0 or growth > max_growth_ratio:
                continue

            day = date.fromordinal(start_ord + idx)
            score = abs(growth - target_growth) + 0.0004 * float(c0 + c1)
            candidate = SmallGrowthCase(
                fire=rec,
                day=day,
                today_count=c0,
                next_count=c1,
                growth_ratio=growth,
                score=score,
            )
            if best is None or candidate.score < best.score:
                best = candidate

    if best is None:
        raise RuntimeError(
            "No small-growth VIIRS day found with current filters; "
            "try lowering --min-daily-count or increasing --max-growth-ratio."
        )
    return best


def _build_rothermel_like_baseline(
    current_binary: np.ndarray,
    mean_u_ms: float,
    mean_v_ms: float,
    lead_hours: float,
    spread_gain: float,
    pixel_size_m: float,
) -> np.ndarray:
    seed = gaussian_filter(current_binary.astype(np.float32), sigma=1.0)

    wind_speed = float(np.hypot(mean_u_ms, mean_v_ms))
    # Convert wind advection into pixel shift and clamp to avoid runaway offsets.
    dx_pix = int(np.clip(np.rint((mean_u_ms * lead_hours * 3600.0) / float(pixel_size_m)), -48, 48))
    dy_pix = int(np.clip(np.rint((mean_v_ms * lead_hours * 3600.0) / float(pixel_size_m)), -48, 48))
    advected = _shift_with_zero_fill(seed, shift_y=dy_pix, shift_x=dx_pix)

    # Rothermel-like spread proxy: wind-dependent isotropic spread + advection.
    spread_sigma = float(np.clip(1.2 + spread_gain * wind_speed, 1.0, 8.0))
    spread = gaussian_filter(seed, sigma=spread_sigma)

    baseline = 0.6 * spread + 0.4 * advected
    baseline = gaussian_filter(baseline, sigma=1.0)
    return _normalize_01(baseline)


def _compute_mean_wind_for_day(
    reader: HRRRAnalysisReader,
    bounds_xy: tuple[float, float, float, float],
    grid: GridSpec,
    sample_day: date,
    sample_hours_utc: tuple[int, ...],
) -> tuple[float, float]:
    u_spec = VariableSpec(variable="UGRD", level="10m_above_ground")
    v_spec = VariableSpec(variable="VGRD", level="10m_above_ground")

    u_vals: list[float] = []
    v_vals: list[float] = []
    for hour in sample_hours_utc:
        ts = _utc_datetime(sample_day, hour)
        u_patch = _extract_hrrr_patch(reader, ts, u_spec, bounds_xy=bounds_xy, grid=grid)
        v_patch = _extract_hrrr_patch(reader, ts, v_spec, bounds_xy=bounds_xy, grid=grid)
        if u_patch is None or v_patch is None:
            continue
        u_vals.append(float(np.nanmean(u_patch)))
        v_vals.append(float(np.nanmean(v_patch)))

    if not u_vals or not v_vals:
        return 0.0, 0.0
    return float(np.mean(u_vals)), float(np.mean(v_vals))


def _iou_score(pred_binary: np.ndarray, truth_binary: np.ndarray) -> float:
    pred = pred_binary.astype(bool)
    truth = truth_binary.astype(bool)
    inter = int(np.count_nonzero(pred & truth))
    union = int(np.count_nonzero(pred | truth))
    if union == 0:
        return 1.0
    return float(inter / union)


def build_demo(
    config: PipelineConfig,
    output_dir: Path,
    sample_hours_utc: tuple[int, ...],
    pixel_size_m: float,
    max_grid_dim: int,
    sigma_truth: float,
    residual_clip: float,
    spread_gain: float,
    lead_hours: float,
    min_daily_count: int,
    max_growth_ratio: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_filtered_fire_records(config)
    viirs_df = load_or_build_viirs_hourly_points(config)

    transformer = hrrr_transformer_to_xy()
    x, y = transformer.transform(viirs_df["longitude"].to_numpy(), viirs_df["latitude"].to_numpy())
    viirs_xy_day_df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "day_utc": pd.to_datetime(viirs_df["hour_utc"], utc=True).dt.date,
            "hour_utc": pd.to_datetime(viirs_df["hour_utc"], utc=True).dt.hour.astype(np.int16),
            "frp": pd.to_numeric(viirs_df["frp"], errors="coerce").fillna(0.0).astype(np.float32),
        }
    )

    case = _find_small_growth_case(
        records=records,
        viirs_xy_day_df=viirs_xy_day_df,
        buffer_km=config.wildfire.buffer_km,
        sample_hours_utc=sample_hours_utc,
        pixel_size_m=pixel_size_m,
        max_grid_dim=max_grid_dim,
        target_growth=0.10,
        min_daily_count=min_daily_count,
        max_growth_ratio=max_growth_ratio,
        include_non_conus=False,
    )

    bounds_xy = _build_bounds_xy(case.fire, buffer_km=config.wildfire.buffer_km)
    grid = _build_fixed_grid(bounds_xy, pixel_size_m=pixel_size_m, max_grid_dim=max_grid_dim)
    if grid is None:
        raise RuntimeError("Selected case cannot fit in requested fixed grid settings.")
    next_day = case.day + timedelta(days=1)

    frp_today = _rasterize_day_with_sample_hours(
        viirs_xy_day_df,
        sample_day=case.day,
        sample_hours_utc=sample_hours_utc,
        grid=grid,
    )
    frp_next = _rasterize_day_with_sample_hours(
        viirs_xy_day_df,
        sample_day=next_day,
        sample_hours_utc=sample_hours_utc,
        grid=grid,
    )

    current_binary = (frp_today > 0).astype(np.float32)
    truth_binary = (frp_next > 0).astype(np.float32)
    truth_smooth = _normalize_01(gaussian_filter(truth_binary, sigma=sigma_truth))

    reader = HRRRAnalysisReader(config.hrrr, max_cache_entries=64)
    mean_u_ms, mean_v_ms = _compute_mean_wind_for_day(
        reader=reader,
        bounds_xy=bounds_xy,
        grid=grid,
        sample_day=case.day,
        sample_hours_utc=sample_hours_utc,
    )

    baseline = _build_rothermel_like_baseline(
        current_binary=current_binary,
        mean_u_ms=mean_u_ms,
        mean_v_ms=mean_v_ms,
        lead_hours=lead_hours,
        spread_gain=spread_gain,
        pixel_size_m=pixel_size_m,
    )

    residual_needed = truth_smooth - baseline
    constrained_delta = np.clip(residual_needed, -residual_clip, residual_clip)
    constrained_pred = np.clip(baseline + constrained_delta, 0.0, 1.0)

    baseline_iou = _iou_score((baseline >= 0.30).astype(np.float32), truth_binary)
    constrained_iou = _iou_score((constrained_pred >= 0.30).astype(np.float32), truth_binary)
    baseline_mse_smooth = float(np.mean((baseline - truth_smooth) ** 2))
    constrained_mse_smooth = float(np.mean((constrained_pred - truth_smooth) ** 2))

    cmap_name = "magma"
    fig1, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    im0 = axes[0].imshow(current_binary, cmap=cmap_name, vmin=0.0, vmax=1.0, origin="lower")
    axes[0].set_title("Current Binary (t)")
    axes[1].imshow(truth_binary, cmap=cmap_name, vmin=0.0, vmax=1.0, origin="lower")
    axes[1].set_title("Next-Day Binary Truth")
    axes[2].imshow(truth_smooth, cmap=cmap_name, vmin=0.0, vmax=1.0, origin="lower")
    axes[2].set_title(f"Smoothed Truth (sigma={sigma_truth:.1f})")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig1.colorbar(im0, ax=axes, fraction=0.046, pad=0.02)
    fig1.suptitle(f"Fire {case.fire.unique_fire_id} on {case.day.isoformat()} -> {next_day.isoformat()}")

    fig2, axes2 = plt.subplots(1, 4, figsize=(19, 4.5), constrained_layout=True)
    deviation_abs = np.clip(np.abs(residual_needed), 0.0, 1.0)
    imb = axes2[0].imshow(baseline, cmap=cmap_name, vmin=0.0, vmax=1.0, origin="lower")
    axes2[0].set_title("Rothermel-Like Baseline")
    axes2[1].imshow(truth_smooth, cmap=cmap_name, vmin=0.0, vmax=1.0, origin="lower")
    axes2[1].set_title("Smoothed Truth")
    axes2[2].imshow(deviation_abs, cmap=cmap_name, vmin=0.0, vmax=1.0, origin="lower")
    axes2[2].set_title("Deviation Magnitude\n|truth - baseline|")
    axes2[3].imshow(constrained_pred, cmap=cmap_name, vmin=0.0, vmax=1.0, origin="lower")
    axes2[3].set_title(f"Constrained Prediction\n(delta clip={residual_clip:.2f})")
    for ax in axes2:
        ax.set_xticks([])
        ax.set_yticks([])
    fig2.colorbar(imb, ax=axes2, fraction=0.046, pad=0.02)

    fig1_path = output_dir / "viirs_truth_binary_vs_smoothed.png"
    fig2_path = output_dir / "viirs_rothermel_like_constrained_demo.png"
    fig1.savefig(fig1_path, dpi=170)
    fig2.savefig(fig2_path, dpi=170)
    plt.close(fig1)
    plt.close(fig2)

    summary = {
        "fire_id": case.fire.unique_fire_id,
        "incident_name": case.fire.incident_name,
        "state": case.fire.state,
        "sample_day": case.day.isoformat(),
        "target_day": next_day.isoformat(),
        "today_viirs_count": case.today_count,
        "next_viirs_count": case.next_count,
        "growth_ratio": case.growth_ratio,
        "sample_hours_utc": list(sample_hours_utc),
        "pixel_size_m": float(pixel_size_m),
        "grid_shape": [int(grid.ny), int(grid.nx)],
        "mean_u_ms": mean_u_ms,
        "mean_v_ms": mean_v_ms,
        "mean_wind_speed_ms": float(np.hypot(mean_u_ms, mean_v_ms)),
        "baseline_iou_vs_binary_truth": baseline_iou,
        "constrained_iou_vs_binary_truth": constrained_iou,
        "baseline_mse_vs_smoothed_truth": baseline_mse_smooth,
        "constrained_mse_vs_smoothed_truth": constrained_mse_smooth,
        "sigma_truth": sigma_truth,
        "residual_clip": residual_clip,
        "spread_gain": spread_gain,
        "lead_hours": lead_hours,
        "figure_binary_vs_smoothed": str(fig1_path),
        "figure_baseline_constrained": str(fig2_path),
    }

    summary_path = output_dir / "viirs_smoothing_demo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def parse_hours_csv(value: str) -> tuple[int, ...]:
    vals = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        hour = int(token)
        if hour < 0 or hour > 23:
            raise ValueError(f"Invalid hour: {hour}")
        vals.append(hour)
    if not vals:
        raise ValueError("No sample hours were provided")
    return tuple(sorted(set(vals)))


def main() -> None:
    parser = argparse.ArgumentParser(description="VIIRS truth smoothing + constrained baseline demo")
    parser.add_argument("--config", type=str, default="/Users/lorn/Code/gribcheck/config/pipeline_config.yaml")
    parser.add_argument("--output-dir", type=str, default="/Users/lorn/Code/gribcheck/reports/figures/viirs_smoothing_demo")
    parser.add_argument("--sample-hours-utc", type=str, default="0,6,12,18")
    parser.add_argument("--pixel-size-m", type=float, default=375.0)
    parser.add_argument("--max-grid-dim", type=int, default=768)
    parser.add_argument("--sigma-truth", type=float, default=1.8)
    parser.add_argument("--residual-clip", type=float, default=0.35)
    parser.add_argument("--spread-gain", type=float, default=0.35)
    parser.add_argument("--lead-hours", type=float, default=24.0)
    parser.add_argument("--min-daily-count", type=int, default=25)
    parser.add_argument("--max-growth-ratio", type=float, default=0.35)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    sample_hours = parse_hours_csv(args.sample_hours_utc)

    summary = build_demo(
        config=config,
        output_dir=output_dir,
        sample_hours_utc=sample_hours,
        pixel_size_m=float(args.pixel_size_m),
        max_grid_dim=int(args.max_grid_dim),
        sigma_truth=float(args.sigma_truth),
        residual_clip=float(args.residual_clip),
        spread_gain=float(args.spread_gain),
        lead_hours=float(args.lead_hours),
        min_daily_count=int(args.min_daily_count),
        max_growth_ratio=float(args.max_growth_ratio),
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
