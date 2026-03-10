#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from pyproj import Transformer

from gribcheck.config import load_config
from gribcheck.geo_utils import HRRR_CRS, WGS84
from gribcheck.hrrr import HRRRAnalysisReader
from gribcheck.models import VariableSpec


def _parse_utc_timestamp(value: str) -> datetime:
    value = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _hourly_range(start: datetime, end: datetime) -> list[datetime]:
    out: list[datetime] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += timedelta(hours=1)
    return out


def _smoke_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "smoke_intense",
        [
            "#0b0f1a",
            "#1f2a44",
            "#234f6f",
            "#2f8f9d",
            "#7ccba2",
            "#f5d76e",
            "#f39c34",
            "#df5a2b",
            "#a61b29",
            "#5c0b1a",
        ],
        N=512,
    )


def _positive_values(arr: np.ndarray) -> np.ndarray:
    return arr[np.isfinite(arr) & (arr > 0.0)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render HRRR smoke plume GIF animation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline_config.yaml"),
        help="Pipeline config path.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2023-06-06T00:00:00Z",
        help="Animation start timestamp (UTC ISO).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2023-06-08T23:00:00Z",
        help="Animation end timestamp (UTC ISO).",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=2,
        help="Grid stride for plotting speed (1 = full resolution).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Animation frames per second.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output GIF DPI.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/figures/hrrr_smoke_plume_20230606_20230608_hourly.gif"),
        help="Output GIF path.",
    )
    args = parser.parse_args()

    start_utc = _parse_utc_timestamp(args.start)
    end_utc = _parse_utc_timestamp(args.end)
    if end_utc < start_utc:
        raise ValueError("end must be >= start")
    stride = max(1, int(args.downsample))

    config = load_config(str(args.config))
    reader = HRRRAnalysisReader(config.hrrr)
    smoke_spec = VariableSpec(variable="MASSDEN", level="8m_above_ground")

    timestamps = _hourly_range(start_utc, end_utc)
    frames: list[tuple[datetime, np.ndarray]] = []
    lon = None
    lat = None

    transformer = Transformer.from_crs(HRRR_CRS, WGS84, always_xy=True)
    for i, ts in enumerate(timestamps, start=1):
        da = reader.load_field(ts, smoke_spec)
        if da is None:
            print(f"[{i}/{len(timestamps)}] missing {ts.isoformat()}")
            continue

        arr = da.to_numpy().astype(np.float32, copy=False)
        x = da["x"].to_numpy()[::stride]
        y = da["y"].to_numpy()[::stride]
        arr = arr[::stride, ::stride]

        if lon is None or lat is None:
            xx, yy = np.meshgrid(x, y)
            lon_raw, lat_raw = transformer.transform(xx, yy)
            lon = lon_raw.astype(np.float32, copy=False)
            lat = lat_raw.astype(np.float32, copy=False)

        conus_mask = (lon >= -130.0) & (lon <= -60.0) & (lat >= 20.0) & (lat <= 56.0)
        arr = np.where(conus_mask, arr, np.nan)
        frames.append((ts, arr))
        print(f"[{i}/{len(timestamps)}] loaded {ts.isoformat()}")

    if not frames:
        raise RuntimeError("No frames were loaded from HRRR.")

    # Estimate robust global log scale with sampled positive pixels.
    samples: list[np.ndarray] = []
    for _, arr in frames:
        pos = _positive_values(arr)
        if pos.size == 0:
            continue
        if pos.size > 25000:
            idx = np.random.default_rng(42).choice(pos.size, size=25000, replace=False)
            pos = pos[idx]
        samples.append(pos)

    if not samples:
        raise RuntimeError("Loaded frames contain no positive smoke values.")
    pooled = np.concatenate(samples)
    vmin = float(np.quantile(pooled, 0.20))
    vmax = float(np.quantile(pooled, 0.9995))
    if not np.isfinite(vmin) or vmin <= 0.0:
        vmin = float(np.nanmin(pooled[pooled > 0]))
    if vmax <= vmin:
        vmax = float(np.nanmax(pooled))

    fig = plt.figure(figsize=(14, 8), dpi=args.dpi)
    ax = fig.add_subplot(111)
    ax.set_facecolor("#080b14")
    cmap = _smoke_cmap()
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    first_ts, first_arr = frames[0]
    mesh = ax.pcolormesh(
        lon,
        lat,
        first_arr,
        shading="nearest",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    cb = fig.colorbar(mesh, ax=ax, pad=0.014, fraction=0.03)
    cb.set_label("MASSDEN (kg m$^{-3}$)", fontsize=11)
    cb.ax.tick_params(labelsize=10)

    ax.set_xlim(-130, -60)
    ax.set_ylim(20, 56)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.grid(color="white", alpha=0.12, linewidth=0.6)
    title = ax.set_title(
        f"HRRR Smoke Plume (MASSDEN 8 m AGL)\n{first_ts.strftime('%Y-%m-%d %H:%M UTC')}",
        fontsize=17,
        pad=12,
    )
    stats_text = ax.text(
        0.012,
        0.026,
        "",
        transform=ax.transAxes,
        fontsize=10,
        color="white",
        bbox=dict(facecolor="black", edgecolor="none", alpha=0.38, boxstyle="round,pad=0.35"),
    )

    def _update(frame_idx: int):
        ts, arr = frames[frame_idx]
        mesh.set_array(arr)
        title.set_text(f"HRRR Smoke Plume (MASSDEN 8 m AGL)\n{ts.strftime('%Y-%m-%d %H:%M UTC')}")
        pos = _positive_values(arr)
        if pos.size > 0:
            mean_val = float(np.mean(pos))
            p99 = float(np.quantile(pos, 0.99))
            max_val = float(np.max(pos))
            stats_text.set_text(f"mean: {mean_val:.2e}\n99th pct: {p99:.2e}\nmax: {max_val:.2e}")
        else:
            stats_text.set_text("no positive pixels")
        return mesh, title, stats_text

    animation = FuncAnimation(
        fig=fig,
        func=_update,
        frames=len(frames),
        interval=int(1000 / max(1, int(args.fps))),
        blit=False,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=max(1, int(args.fps)))
    animation.save(str(args.output), writer=writer)
    plt.close(fig)

    print(f"frames={len(frames)}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
