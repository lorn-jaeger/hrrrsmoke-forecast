#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import zipfile
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from matplotlib.animation import FuncAnimation, PillowWriter
from netCDF4 import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.wfigs_viirs_missing_causes_sample import (  # noqa: E402
    GoesIndex,
    ensure_goes_file,
    parse_goes_start_from_key,
)
from scripts.wfigs_viirs_stats import WfigsFire, load_wfigs_fires  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate GOES cloud mask vs VIIRS fire detections for one WFIGS fire")
    parser.add_argument(
        "--wfigs-geojson",
        type=Path,
        default=Path("/Users/lorn/Downloads/WFIGS_Interagency_Perimeters_6781836551080060975.geojson"),
    )
    parser.add_argument(
        "--viirs-zip",
        type=Path,
        default=Path("/Users/lorn/Downloads/DL_FIRE_J1V-C2_718831.zip"),
    )
    parser.add_argument("--fire-id", type=str, default=None, help="Exact WFIGS fire ID (attr_UniqueFireIdentifier)")
    parser.add_argument("--fire-name", type=str, default=None, help="Case-insensitive name contains match")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--days-from-start", type=int, default=5, help="Used if end-date is omitted")
    parser.add_argument("--grid-nx", type=int, default=220)
    parser.add_argument("--grid-ny", type=int, default=160)
    parser.add_argument("--bbox-pad-deg", type=float, default=0.35)
    parser.add_argument(
        "--frame-step-minutes",
        type=int,
        default=5,
        help="Target cadence in minutes; use <=0 to include every GOES granule frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional hard cap; <=0 means no cap.",
    )
    parser.add_argument(
        "--goes-base-url",
        type=str,
        default="https://noaa-goes18.s3.amazonaws.com",
    )
    parser.add_argument("--goes-prefix", type=str, default="ABI-L2-ACMC")
    parser.add_argument(
        "--goes-cache-dir",
        type=Path,
        default=Path("/Users/lorn/Code/gribcheck/data/external/goes18_acmc_cache"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Users/lorn/Code/gribcheck/reports/figures/goes_vs_viirs_animation.gif"),
    )
    return parser.parse_args()


def choose_fire(fires: list[WfigsFire], fire_id: str | None, fire_name: str | None) -> WfigsFire:
    if fire_id:
        for f in fires:
            if f.fire_id == fire_id:
                return f
        raise ValueError(f"fire-id not found: {fire_id}")
    if fire_name:
        target = fire_name.strip().lower()
        matches = [f for f in fires if target in f.name.lower()]
        if not matches:
            raise ValueError(f"fire-name not found: {fire_name}")
        return max(matches, key=lambda x: x.size_acres)
    # default: largest fire in filtered list
    return max(fires, key=lambda x: x.size_acres)


def parse_date_opt(s: str | None) -> date | None:
    if s is None:
        return None
    return date.fromisoformat(s)


def load_viirs_for_bbox_time(
    viirs_zip: Path,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    usecols = ["latitude", "longitude", "acq_date", "acq_time"]
    frames = []
    with zipfile.ZipFile(viirs_zip) as zf:
        member = [n for n in zf.namelist() if n.lower().endswith(".csv") and "archive" in n.lower()][0]
        with zf.open(member) as fp:
            for chunk in pd.read_csv(fp, usecols=usecols, chunksize=350_000, low_memory=False):
                lat = pd.to_numeric(chunk["latitude"], errors="coerce")
                lon = pd.to_numeric(chunk["longitude"], errors="coerce")
                keep = lat.between(min_lat, max_lat) & lon.between(min_lon, max_lon)
                if not keep.any():
                    continue
                sub = chunk.loc[keep].copy()
                time_str = sub["acq_time"].astype(str).str.strip().str.zfill(4)
                dt_text = (
                    sub["acq_date"].astype(str).str.strip()
                    + " "
                    + time_str.str.slice(0, 2)
                    + ":"
                    + time_str.str.slice(2, 4)
                )
                dt = pd.to_datetime(dt_text, format="%Y-%m-%d %H:%M", errors="coerce", utc=True)
                sub["acq_datetime"] = dt
                sub = sub.dropna(subset=["acq_datetime", "latitude", "longitude"])
                sub = sub[(sub["acq_datetime"] >= start_dt) & (sub["acq_datetime"] <= end_dt)]
                if sub.empty:
                    continue
                frames.append(sub[["acq_datetime", "latitude", "longitude"]])
    if not frames:
        return pd.DataFrame(columns=["acq_datetime", "latitude", "longitude"])
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("acq_datetime").reset_index(drop=True)
    return out


def list_goes_keys_for_range(
    idx: GoesIndex,
    start_dt: datetime,
    end_dt: datetime,
) -> list[tuple[datetime, str]]:
    keys = {}
    cursor = start_dt.replace(minute=0, second=0, microsecond=0)
    end_hr = end_dt.replace(minute=0, second=0, microsecond=0)
    while cursor <= end_hr:
        y = cursor.year
        doy = cursor.timetuple().tm_yday
        h = cursor.hour
        for ref in idx.list_hour(y, doy, h):
            if start_dt <= ref.start_time_utc <= end_dt:
                keys[ref.key] = ref.start_time_utc
        cursor += timedelta(hours=1)
    items = sorted([(t, k) for k, t in keys.items()], key=lambda x: x[0])
    return items


def build_frame_schedule(goes_times: list[datetime], step_minutes: int, max_frames: int) -> list[datetime]:
    if not goes_times:
        return []
    if step_minutes <= 0:
        schedule = list(goes_times)
    else:
        schedule = [goes_times[0]]
        min_step = timedelta(minutes=step_minutes)
        for t in goes_times[1:]:
            if (t - schedule[-1]) >= min_step:
                schedule.append(t)
    if max_frames > 0 and len(schedule) > max_frames:
        idx = np.linspace(0, len(schedule) - 1, max_frames, dtype=int)
        schedule = [schedule[i] for i in idx]
    return schedule


def nearest_goes_key(schedule_dt: datetime, goes_items: list[tuple[datetime, str]]) -> str | None:
    if not goes_items:
        return None
    times = np.array([x[0].timestamp() for x in goes_items], dtype=np.float64)
    target = schedule_dt.timestamp()
    i = int(np.argmin(np.abs(times - target)))
    if abs(times[i] - target) > 20 * 60:
        return None
    return goes_items[i][1]


def iter_geometry_exteriors(geom):
    if geom.geom_type == "Polygon":
        yield geom.exterior
    elif geom.geom_type == "MultiPolygon":
        for p in geom.geoms:
            yield p.exterior


def main() -> None:
    args = parse_args()
    args.goes_cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading WFIGS cohort...")
    fires = load_wfigs_fires(
        path=args.wfigs_geojson,
        start_year=2016,
        end_year=2026,
        min_acres=1000.0,
    )
    fire = choose_fire(fires, args.fire_id, args.fire_name)
    print(f"Selected fire: {fire.fire_id} | {fire.name} | {fire.start_date} -> {fire.end_date}")

    start_d = parse_date_opt(args.start_date) or fire.start_date
    end_d = parse_date_opt(args.end_date) or min(fire.end_date, start_d + timedelta(days=max(1, args.days_from_start)))
    start_dt = datetime.combine(start_d, time(0, 0), tzinfo=timezone.utc)
    end_dt = datetime.combine(end_d, time(23, 59), tzinfo=timezone.utc)

    minx, miny, maxx, maxy = fire.geometry.bounds
    minx -= args.bbox_pad_deg
    maxx += args.bbox_pad_deg
    miny -= args.bbox_pad_deg
    maxy += args.bbox_pad_deg

    print("Loading VIIRS points in bbox/time...")
    viirs = load_viirs_for_bbox_time(
        viirs_zip=args.viirs_zip,
        min_lon=minx,
        min_lat=miny,
        max_lon=maxx,
        max_lat=maxy,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    print(f"VIIRS rows in window: {len(viirs):,}")

    print("Listing GOES files...")
    goes_idx = GoesIndex(base_url=args.goes_base_url, product_prefix=args.goes_prefix)
    goes_items = list_goes_keys_for_range(goes_idx, start_dt=start_dt, end_dt=end_dt)
    if not goes_items:
        raise RuntimeError("No GOES ACMC files found for requested range.")
    goes_times = [t for t, _k in goes_items]
    schedule = build_frame_schedule(goes_times, step_minutes=args.frame_step_minutes, max_frames=args.max_frames)
    print(f"Frames: {len(schedule):,}")

    # Prepare visualization grid in lon/lat.
    lon_grid = np.linspace(minx, maxx, args.grid_nx, dtype=np.float64)
    lat_grid = np.linspace(miny, maxy, args.grid_ny, dtype=np.float64)
    lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)

    # Get projection and grid mapping from first needed file.
    first_key = goes_items[0][1]
    first_path = ensure_goes_file(args.goes_base_url, first_key, args.goes_cache_dir)
    with Dataset(first_path) as ds:
        xg = ds.variables["x"][:].astype(np.float64)
        yg = ds.variables["y"][:].astype(np.float64)
        proj = ds.variables["goes_imager_projection"]
        h = float(proj.perspective_point_height)
        lon_0 = float(proj.longitude_of_projection_origin)
        semi_major = float(proj.semi_major_axis)
        semi_minor = float(proj.semi_minor_axis)

    crs_geos = pyproj.CRS.from_proj4(
        f"+proj=geos +h={h} +lon_0={lon_0} +sweep=x +a={semi_major} +b={semi_minor} +no_defs"
    )
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs_geos, always_xy=True)
    x_m, y_m = transformer.transform(lon2d.ravel(), lat2d.ravel())
    xr = (np.asarray(x_m, dtype=np.float64) / h).reshape(lat2d.shape)
    yr = (np.asarray(y_m, dtype=np.float64) / h).reshape(lat2d.shape)

    valid = (
        np.isfinite(xr)
        & np.isfinite(yr)
        & (xr >= float(xg.min()))
        & (xr <= float(xg.max()))
        & (yr >= float(yg.min()))
        & (yr <= float(yg.max()))
    )
    xi = np.searchsorted(xg, xr, side="left")
    xi = np.clip(xi, 0, len(xg) - 1)
    xi_prev = np.clip(xi - 1, 0, len(xg) - 1)
    xi = np.where(np.abs(xg[xi_prev] - xr) < np.abs(xg[xi] - xr), xi_prev, xi).astype(np.int32)

    yi = np.searchsorted(yg, yr, side="left")
    yi = np.clip(yi, 0, len(yg) - 1)
    yi_prev = np.clip(yi - 1, 0, len(yg) - 1)
    yi = np.where(np.abs(yg[yi_prev] - yr) < np.abs(yg[yi] - yr), yi_prev, yi).astype(np.int32)

    print("Preparing frame masks...")
    goes_current = []
    viirs_current = []
    viirs_times = viirs["acq_datetime"].to_numpy(dtype="datetime64[ns]") if not viirs.empty else np.array([], dtype="datetime64[ns]")
    viirs_lons = viirs["longitude"].to_numpy(dtype=np.float64) if not viirs.empty else np.array([], dtype=np.float64)
    viirs_lats = viirs["latitude"].to_numpy(dtype=np.float64) if not viirs.empty else np.array([], dtype=np.float64)

    for idx_f, dt in enumerate(schedule, start=1):
        g_key = nearest_goes_key(dt, goes_items)
        if g_key is None:
            g_mask = np.zeros_like(valid, dtype=np.uint8)
        else:
            g_path = ensure_goes_file(args.goes_base_url, g_key, args.goes_cache_dir)
            with Dataset(g_path) as ds:
                bcm = ds.variables["BCM"][:]
                dqf = ds.variables["DQF"][:]
                vals = bcm[yi, xi]
                q = dqf[yi, xi]
                g_mask = ((vals == 1) & (q != 255) & valid).astype(np.uint8)
        goes_current.append(g_mask)

        if viirs_times.size == 0:
            v_mask = np.zeros((args.grid_ny, args.grid_nx), dtype=np.uint8)
        else:
            half = timedelta(minutes=args.frame_step_minutes / 2.0)
            lo = np.datetime64((dt - half).isoformat())
            hi = np.datetime64((dt + half).isoformat())
            sel = (viirs_times >= lo) & (viirs_times <= hi)
            v_mask = np.zeros((args.grid_ny, args.grid_nx), dtype=np.uint8)
            if np.any(sel):
                lx = viirs_lons[sel]
                ly = viirs_lats[sel]
                iix = np.clip(np.searchsorted(lon_grid, lx, side="left"), 0, len(lon_grid) - 1)
                iiy = np.clip(np.searchsorted(lat_grid, ly, side="left"), 0, len(lat_grid) - 1)
                np.add.at(v_mask, (iiy, iix), 1)
                v_mask = (v_mask > 0).astype(np.uint8)
        viirs_current.append(v_mask)

        if idx_f % 40 == 0:
            print(f"Prepared frames: {idx_f}/{len(schedule)}")

    goes_cum = []
    viirs_cum = []
    g_acc = np.zeros_like(goes_current[0], dtype=np.uint8)
    v_acc = np.zeros_like(viirs_current[0], dtype=np.uint8)
    for g, v in zip(goes_current, viirs_current, strict=True):
        g_acc = np.maximum(g_acc, g)
        v_acc = np.maximum(v_acc, v)
        goes_cum.append(g_acc.copy())
        viirs_cum.append(v_acc.copy())

    print("Rendering animation...")
    extent = [minx, maxx, miny, maxy]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax_vc, ax_gc, ax_vs, ax_gs = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    im_vc = ax_vc.imshow(viirs_cum[0], origin="lower", extent=extent, cmap="magma", vmin=0, vmax=1, alpha=0.95)
    im_gc = ax_gc.imshow(goes_cum[0], origin="lower", extent=extent, cmap="Blues", vmin=0, vmax=1, alpha=0.95)
    im_vs = ax_vs.imshow(viirs_current[0], origin="lower", extent=extent, cmap="hot", vmin=0, vmax=1, alpha=0.95)
    im_gs = ax_gs.imshow(goes_current[0], origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=1, alpha=0.95)

    for ax in [ax_vc, ax_gc, ax_vs, ax_gs]:
        for ext in iter_geometry_exteriors(fire.geometry):
            x, y = ext.xy
            ax.plot(x, y, color="lime", linewidth=1.0)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")

    ax_vc.set_title("VIIRS Cumulative Mask")
    ax_gc.set_title("GOES Cumulative Cloud Mask")
    ax_vs.set_title("VIIRS Current Snapshot")
    ax_gs.set_title("GOES Current Snapshot")
    suptitle = fig.suptitle("")

    def update(frame_i: int):
        im_vc.set_data(viirs_cum[frame_i])
        im_gc.set_data(goes_cum[frame_i])
        im_vs.set_data(viirs_current[frame_i])
        im_gs.set_data(goes_current[frame_i])
        suptitle.set_text(
            f"{fire.fire_id} | {fire.name} | frame {frame_i+1}/{len(schedule)} | {schedule[frame_i].strftime('%Y-%m-%d %H:%M UTC')}"
        )
        return im_vc, im_gc, im_vs, im_gs, suptitle

    anim = FuncAnimation(fig, update, frames=len(schedule), interval=120, blit=False)
    out = args.output
    if out.suffix.lower() == ".gif":
        anim.save(out, writer=PillowWriter(fps=8))
    else:
        anim.save(out, fps=8)
    plt.close(fig)
    print(f"Animation written: {out}")


if __name__ == "__main__":
    main()
