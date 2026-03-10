#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

from gribcheck.config import PipelineConfig, load_config
from gribcheck.fire import load_filtered_fire_records
from gribcheck.geo_utils import hrrr_transformer_to_xy
from gribcheck.models import FireRecord
from gribcheck.viirs import load_or_build_viirs_hourly_points


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


@dataclass(frozen=True)
class FireExample:
    fire: FireRecord
    anchor_day: date
    window_start_day: date
    window_end_day: date
    duration_days: int
    peak_daily_frp: float
    anchor_daily_frp: float
    anchor_daily_count: int
    anchor_rel_peak: float
    grid: GridSpec


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
    return GridSpec(xmin=float(xmin), ymin=float(ymin), nx=int(nx), ny=int(ny), pixel_size_m=float(pixel_size_m))


def _rasterize_xyf(points_df: pd.DataFrame, grid: GridSpec) -> np.ndarray:
    out = np.zeros((grid.ny, grid.nx), dtype=np.float32)
    if points_df.empty:
        return out

    x = points_df["x"].to_numpy(dtype=np.float64)
    y = points_df["y"].to_numpy(dtype=np.float64)
    frp = points_df["frp"].to_numpy(dtype=np.float32)

    in_bounds = (x >= grid.xmin) & (x < grid.xmax) & (y >= grid.ymin) & (y < grid.ymax)
    if not np.any(in_bounds):
        return out
    x = x[in_bounds]
    y = y[in_bounds]
    frp = frp[in_bounds]

    ix = np.floor((x - grid.xmin) / grid.pixel_size_m).astype(np.int32)
    iy = np.floor((y - grid.ymin) / grid.pixel_size_m).astype(np.int32)
    ix = np.clip(ix, 0, grid.nx - 1)
    iy = np.clip(iy, 0, grid.ny - 1)
    np.add.at(out, (iy, ix), frp)
    return out


def _rasterize_presence(points_df: pd.DataFrame, grid: GridSpec) -> np.ndarray:
    out = np.zeros((grid.ny, grid.nx), dtype=bool)
    if points_df.empty:
        return out

    x = points_df["x"].to_numpy(dtype=np.float64)
    y = points_df["y"].to_numpy(dtype=np.float64)
    in_bounds = (x >= grid.xmin) & (x < grid.xmax) & (y >= grid.ymin) & (y < grid.ymax)
    if not np.any(in_bounds):
        return out
    x = x[in_bounds]
    y = y[in_bounds]

    ix = np.floor((x - grid.xmin) / grid.pixel_size_m).astype(np.int32)
    iy = np.floor((y - grid.ymin) / grid.pixel_size_m).astype(np.int32)
    ix = np.clip(ix, 0, grid.nx - 1)
    iy = np.clip(iy, 0, grid.ny - 1)
    out[iy, ix] = True
    return out


def _prepare_viirs_xy_day_df(config: PipelineConfig) -> pd.DataFrame:
    viirs_df = load_or_build_viirs_hourly_points(config)
    transformer = hrrr_transformer_to_xy()
    x, y = transformer.transform(viirs_df["longitude"].to_numpy(), viirs_df["latitude"].to_numpy())
    out = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "frp": pd.to_numeric(viirs_df["frp"], errors="coerce").fillna(0.0).astype(np.float32),
            "day_utc": pd.to_datetime(viirs_df["hour_utc"], utc=True).dt.date,
        }
    )
    return out


def _select_examples(
    records: list[FireRecord],
    viirs_xy_day_df: pd.DataFrame,
    n_examples: int,
    min_fire_size_acres: float,
    min_duration_days: int,
    min_daily_count: int,
    target_rel_peak: float,
    late_start_frac: float,
    late_end_frac: float,
    max_rel_peak: float,
    window_days: int,
    pixel_size_m: float,
    max_grid_dim: int,
    buffer_km: float,
) -> list[FireExample]:
    x_all = viirs_xy_day_df["x"].to_numpy(dtype=np.float64)
    y_all = viirs_xy_day_df["y"].to_numpy(dtype=np.float64)
    day_ord_all = viirs_xy_day_df["day_utc"].map(date.toordinal).to_numpy(dtype=np.int32)
    frp_all = viirs_xy_day_df["frp"].to_numpy(dtype=np.float32)

    candidates: list[tuple[float, FireExample]] = []

    filtered_fires = [
        r
        for r in records
        if r.size_acres >= min_fire_size_acres and (r.state or "").upper() not in {"US-AK", "US-HI"}
    ]
    filtered_fires = sorted(filtered_fires, key=lambda r: (-r.size_acres, r.start_date, r.unique_fire_id))

    for rec in filtered_fires:
        duration_days = int((rec.end_date - rec.start_date).days) + 1
        if duration_days < min_duration_days:
            continue

        bounds = _build_bounds_xy(rec, buffer_km=buffer_km)
        grid = _build_fixed_grid(bounds, pixel_size_m=pixel_size_m, max_grid_dim=max_grid_dim)
        if grid is None:
            continue

        xmin, ymin, xmax, ymax = bounds
        in_bbox = (x_all >= xmin) & (x_all <= xmax) & (y_all >= ymin) & (y_all <= ymax)
        if int(np.count_nonzero(in_bbox)) < (min_daily_count * 4):
            continue

        start_ord = rec.start_date.toordinal()
        end_ord = rec.end_date.toordinal()

        day_ord = day_ord_all[in_bbox]
        frp = frp_all[in_bbox]
        in_fire_days = (day_ord >= start_ord) & (day_ord <= end_ord)
        day_ord = day_ord[in_fire_days]
        frp = frp[in_fire_days]
        if day_ord.size == 0:
            continue

        rel_day = day_ord - start_ord
        daily_frp = np.bincount(rel_day, weights=frp, minlength=duration_days).astype(np.float64)
        daily_count = np.bincount(rel_day, minlength=duration_days).astype(np.int32)

        peak = float(np.max(daily_frp))
        if peak <= 0:
            continue

        late_start_idx = int(np.floor((duration_days - 1) * late_start_frac))
        late_end_idx = int(np.floor((duration_days - 1) * late_end_frac))
        late_start_idx = max(0, min(late_start_idx, duration_days - 1))
        late_end_idx = max(late_start_idx, min(late_end_idx, duration_days - 1))

        for idx in range(late_start_idx, late_end_idx + 1):
            frp_day = float(daily_frp[idx])
            count_day = int(daily_count[idx])
            if count_day < min_daily_count:
                continue
            rel = frp_day / peak
            if rel <= 0.0 or rel > max_rel_peak:
                continue

            anchor_day = date.fromordinal(start_ord + idx)
            window_end = anchor_day
            window_start = anchor_day - timedelta(days=max(0, window_days - 1))
            if window_start < rec.start_date:
                window_start = rec.start_date

            # Prefer larger fires, late in lifecycle, and rel FRP close to target.
            frac_to_end = float(end_ord - (start_ord + idx)) / max(1.0, float(duration_days - 1))
            score = (
                abs(rel - target_rel_peak)
                + 0.75 * frac_to_end
                + 0.02 * max(0, min_daily_count - count_day)
                - 0.000004 * float(rec.size_acres)
            )

            ex = FireExample(
                fire=rec,
                anchor_day=anchor_day,
                window_start_day=window_start,
                window_end_day=window_end,
                duration_days=duration_days,
                peak_daily_frp=peak,
                anchor_daily_frp=frp_day,
                anchor_daily_count=count_day,
                anchor_rel_peak=rel,
                grid=grid,
            )
            candidates.append((score, ex))

    if not candidates:
        return []

    # Deduplicate by fire id, keep best score per fire.
    best_by_fire: dict[str, tuple[float, FireExample]] = {}
    for score, ex in candidates:
        key = ex.fire.unique_fire_id
        prev = best_by_fire.get(key)
        if prev is None or score < prev[0]:
            best_by_fire[key] = (score, ex)

    chosen = sorted(best_by_fire.values(), key=lambda x: x[0])
    return [ex for _, ex in chosen[:n_examples]]


def build_visuals(
    config: PipelineConfig,
    output_dir: Path,
    n_examples: int,
    pixel_size_m: float,
    max_grid_dim: int,
    buffer_km: float,
    window_days: int,
    min_fire_size_acres: float,
    min_duration_days: int,
    min_daily_count: int,
    target_rel_peak: float,
    max_rel_peak: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_filtered_fire_records(config)
    viirs = _prepare_viirs_xy_day_df(config)

    examples = _select_examples(
        records=records,
        viirs_xy_day_df=viirs,
        n_examples=n_examples,
        min_fire_size_acres=min_fire_size_acres,
        min_duration_days=min_duration_days,
        min_daily_count=min_daily_count,
        target_rel_peak=target_rel_peak,
        late_start_frac=0.70,
        late_end_frac=0.98,
        max_rel_peak=max_rel_peak,
        window_days=window_days,
        pixel_size_m=pixel_size_m,
        max_grid_dim=max_grid_dim,
        buffer_km=buffer_km,
    )

    if not examples:
        raise RuntimeError("No suitable late-stage examples found. Relax thresholds.")

    # Build per-example rasters first so we can use shared color scales.
    rasters: list[tuple[np.ndarray, np.ndarray]] = []
    meta_rows: list[dict[str, object]] = []

    x_all = viirs["x"].to_numpy(dtype=np.float64)
    y_all = viirs["y"].to_numpy(dtype=np.float64)
    day_ord_all = viirs["day_utc"].map(date.toordinal).to_numpy(dtype=np.int32)

    for ex in examples:
        bounds = _build_bounds_xy(ex.fire, buffer_km=buffer_km)
        xmin, ymin, xmax, ymax = bounds
        in_bbox = (x_all >= xmin) & (x_all <= xmax) & (y_all >= ymin) & (y_all <= ymax)

        start_ord = ex.fire.start_date.toordinal()
        end_ord = ex.fire.end_date.toordinal()
        cumulative_mask = in_bbox & (day_ord_all >= start_ord) & (day_ord_all <= end_ord)

        w_start = ex.window_start_day.toordinal()
        w_end = ex.window_end_day.toordinal()
        recent_mask = in_bbox & (day_ord_all >= w_start) & (day_ord_all <= w_end)

        cum_raster = _rasterize_xyf(viirs.loc[cumulative_mask, ["x", "y", "frp"]], ex.grid)
        recent_points = viirs.loc[recent_mask, ["x", "y", "frp", "day_utc"]]
        day_mask = np.zeros((ex.grid.ny, ex.grid.nx), dtype=np.int16)
        day_pixel_counts: list[int] = []
        for day_idx in range(window_days):
            day_val = ex.window_start_day + timedelta(days=day_idx)
            points_day = recent_points.loc[recent_points["day_utc"] == day_val, ["x", "y"]]
            presence = _rasterize_presence(points_day, ex.grid)
            day_pixel_counts.append(int(np.count_nonzero(presence)))
            # Keep latest day index where a pixel is detected in the window.
            day_mask[presence] = int(day_idx + 1)

        log_cum = np.log1p(cum_raster)
        rasters.append((log_cum, day_mask))

        meta_rows.append(
            {
                "fire_id": ex.fire.unique_fire_id,
                "incident_name": ex.fire.incident_name,
                "state": ex.fire.state,
                "size_acres": float(ex.fire.size_acres),
                "start_date": ex.fire.start_date.isoformat(),
                "end_date": ex.fire.end_date.isoformat(),
                "duration_days": int(ex.duration_days),
                "anchor_day": ex.anchor_day.isoformat(),
                "anchor_daily_frp": float(ex.anchor_daily_frp),
                "peak_daily_frp": float(ex.peak_daily_frp),
                "anchor_rel_peak": float(ex.anchor_rel_peak),
                "anchor_daily_count": int(ex.anchor_daily_count),
                "window_start_day": ex.window_start_day.isoformat(),
                "window_end_day": ex.window_end_day.isoformat(),
                "grid_shape": [int(ex.grid.ny), int(ex.grid.nx)],
                "pixel_size_m": float(ex.grid.pixel_size_m),
                "cumulative_total_frp": float(np.sum(np.expm1(log_cum))),
                "recent_window_total_frp": float(np.sum(recent_points["frp"].to_numpy(dtype=np.float64))),
                "recent_window_day_pixel_counts": day_pixel_counts,
            }
        )

    global_max = 0.0
    for log_cum, _day_mask in rasters:
        global_max = max(global_max, float(np.nanmax(log_cum)))
    if not np.isfinite(global_max) or global_max <= 0:
        global_max = 1.0

    day_colors = plt.get_cmap("turbo")(np.linspace(0.12, 0.95, window_days))
    day_cmap = ListedColormap(day_colors)
    day_cmap.set_bad(color="#f5f5f5")
    day_norm = BoundaryNorm(np.arange(0.5, window_days + 1.5, 1), day_cmap.N)

    n = len(examples)
    cmap = "magma"
    figure_paths: list[str] = []
    for i, (ex, (log_cum, day_mask)) in enumerate(zip(examples, rasters), start=1):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
        ax0, ax1 = axes

        im_log = ax0.imshow(log_cum, cmap=cmap, vmin=0.0, vmax=global_max, origin="lower")
        day_mask_ma = np.ma.masked_where(day_mask == 0, day_mask)
        im_day = ax1.imshow(day_mask_ma, cmap=day_cmap, norm=day_norm, origin="lower")

        ax0.set_title(
            f"{ex.fire.incident_name} ({ex.fire.state})\n"
            f"log1p(Cumulative FRP: start→end)"
        )
        ax1.set_title(
            f"{ex.fire.incident_name} ({ex.fire.state})\n"
            f"Late {window_days}-day detection-day mask"
        )

        subtitle = (
            f"Anchor={ex.anchor_day.isoformat()} | rel={ex.anchor_rel_peak:.3f} of peak | "
            f"size={ex.fire.size_acres:,.0f} ac | grid={ex.grid.ny}x{ex.grid.nx} @ {pixel_size_m:.0f}m"
        )
        ax0.set_xlabel(subtitle)
        ax1.set_xlabel(
            f"Window={ex.window_start_day.isoformat()}→{ex.window_end_day.isoformat()} | "
            f"day index: 1=start ... {window_days}=end"
        )

        for ax in (ax0, ax1):
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(
            f"Late-Stage VIIRS Example {i}/{n} ({pixel_size_m:.0f}m grid)",
            fontsize=13,
        )
        fig.colorbar(im_log, ax=[ax0], fraction=0.045, pad=0.02, label="log1p(FRP)")
        fig.colorbar(
            im_day,
            ax=[ax1],
            fraction=0.045,
            pad=0.02,
            ticks=np.arange(1, window_days + 1),
            label=f"Detection Day Index (1={window_days}-day window start, {window_days}=window end)",
        )

        safe_fire_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in ex.fire.unique_fire_id)
        out_png = output_dir / f"viirs_endstage_example_{i:02d}_{safe_fire_id}.png"
        fig.savefig(out_png, dpi=190)
        plt.close(fig)
        figure_paths.append(str(out_png))

    summary = {
        "pixel_size_m": float(pixel_size_m),
        "window_days": int(window_days),
        "n_examples": int(n),
        "shared_colorbar_vmin": 0.0,
        "shared_colorbar_vmax": float(global_max),
        "day_mask_index_min": 1,
        "day_mask_index_max": int(window_days),
        "figure_paths": figure_paths,
        "examples": meta_rows,
    }
    out_json = output_dir / "viirs_endstage_cumulative_vs_recent_5day_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(out_json)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize cumulative FRP vs late-stage 5-day detection-day mask on 375m grid")
    parser.add_argument("--config", type=str, default="/Users/lorn/Code/gribcheck/config/pipeline_config.yaml")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/lorn/Code/gribcheck/reports/figures/viirs_endstage_examples",
    )
    parser.add_argument("--n-examples", type=int, default=3)
    parser.add_argument("--pixel-size-m", type=float, default=375.0)
    parser.add_argument("--max-grid-dim", type=int, default=768)
    parser.add_argument("--buffer-km", type=float, default=30.0)
    parser.add_argument("--window-days", type=int, default=5)
    parser.add_argument("--min-fire-size-acres", type=float, default=50000.0)
    parser.add_argument("--min-duration-days", type=int, default=18)
    parser.add_argument("--min-daily-count", type=int, default=20)
    parser.add_argument("--target-rel-peak", type=float, default=0.10)
    parser.add_argument("--max-rel-peak", type=float, default=0.25)
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary = build_visuals(
        config=cfg,
        output_dir=Path(args.output_dir),
        n_examples=int(args.n_examples),
        pixel_size_m=float(args.pixel_size_m),
        max_grid_dim=int(args.max_grid_dim),
        buffer_km=float(args.buffer_km),
        window_days=int(args.window_days),
        min_fire_size_acres=float(args.min_fire_size_acres),
        min_duration_days=int(args.min_duration_days),
        min_daily_count=int(args.min_daily_count),
        target_rel_peak=float(args.target_rel_peak),
        max_rel_peak=float(args.max_rel_peak),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
