#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import urllib.parse
import urllib.request
import zipfile
from datetime import date, datetime, time, timedelta, timezone
from io import BytesIO
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

from scripts.wfigs_viirs_missing_causes_sample import GoesIndex, ensure_goes_file  # noqa: E402
from scripts.wfigs_viirs_stats import WfigsFire, load_wfigs_fires  # noqa: E402

# GOES FDCF fire-like classes from Mask flag values/meanings.
GOES_FIRE_VALUES = np.array([10, 11, 12, 13, 14, 15, 30, 31, 32, 33, 34, 35], dtype=np.int16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate GOES-18 fire mask (15-min) vs VIIRS detections (12-hour held bins) for one WFIGS fire"
    )
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

    # Fire selection options (first matching strategy wins in this order).
    parser.add_argument("--fire-id", type=str, default=None, help="Exact WFIGS fire ID")
    parser.add_argument("--fire-name", type=str, default=None, help="Case-insensitive incident-name contains match")
    parser.add_argument("--center-lat", type=float, default=None, help="Pick nearest fire centroid to this latitude")
    parser.add_argument("--center-lon", type=float, default=None, help="Pick nearest fire centroid to this longitude")
    parser.add_argument("--date-hint", type=str, default=None, help="Optional YYYY-MM-DD date used to disambiguate lat/lon selection")

    # Time range options.
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--days-from-start", type=int, default=3, help="Used if end-date is omitted and --use-fire-window is false")
    parser.add_argument("--use-fire-window", action="store_true", help="Use full WFIGS start->end window")

    # Grid and cadence.
    parser.add_argument("--grid-nx", type=int, default=260)
    parser.add_argument("--grid-ny", type=int, default=200)
    parser.add_argument("--bbox-pad-deg", type=float, default=0.35)
    parser.add_argument("--frame-step-minutes", type=int, default=15)
    parser.add_argument("--viirs-hold-hours", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=0, help="Optional hard cap; <=0 means no cap")
    parser.add_argument("--fps", type=int, default=10)

    # GOES access.
    parser.add_argument("--goes-base-url", type=str, default="https://noaa-goes18.s3.amazonaws.com")
    parser.add_argument("--goes-prefix", type=str, default="ABI-L2-FDCF")
    parser.add_argument(
        "--goes-cache-dir",
        type=Path,
        default=Path("/Users/lorn/Code/gribcheck/data/external/goes18_fdcf_cache"),
    )

    # Basemap.
    parser.add_argument("--map-width", type=int, default=1280)
    parser.add_argument("--map-height", type=int, default=840)
    parser.add_argument("--map-alpha", type=float, default=0.85)
    parser.set_defaults(map_background=True)
    parser.add_argument("--map-background", dest="map_background", action="store_true")
    parser.add_argument("--no-map-background", dest="map_background", action="store_false")

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Users/lorn/Code/gribcheck/reports/figures/goes_vs_viirs_progression.gif"),
    )
    return parser.parse_args()


def parse_date_opt(s: str | None) -> date | None:
    if s is None:
        return None
    return date.fromisoformat(s)


def choose_fire(
    fires: list[WfigsFire],
    fire_id: str | None,
    fire_name: str | None,
    center_lat: float | None,
    center_lon: float | None,
    date_hint: date | None,
) -> WfigsFire:
    if fire_id:
        for f in fires:
            if f.fire_id == fire_id:
                return f
        raise ValueError(f"fire-id not found: {fire_id}")

    if fire_name:
        q = fire_name.strip().lower()
        matches = [f for f in fires if q in f.name.lower()]
        if not matches:
            raise ValueError(f"fire-name not found: {fire_name}")
        return max(matches, key=lambda r: r.size_acres)

    if center_lat is not None and center_lon is not None:
        def score(f: WfigsFire) -> float:
            c = f.geometry.centroid
            d = math.hypot(float(c.y) - center_lat, float(c.x) - center_lon)
            if date_hint is None:
                return d
            if f.start_date <= date_hint <= f.end_date:
                return d
            dd = min(abs((date_hint - f.start_date).days), abs((date_hint - f.end_date).days))
            return d + (dd / 100.0)

        return min(fires, key=score)

    return max(fires, key=lambda r: r.size_acres)


def load_viirs_for_bbox_time(
    viirs_zip: Path,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    usecols = ["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp"]
    rows = []
    with zipfile.ZipFile(viirs_zip) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv") and "archive" in n.lower()]
        if not members:
            raise FileNotFoundError(f"No archive CSV found in {viirs_zip}")
        with zf.open(members[0]) as fp:
            for chunk in pd.read_csv(fp, usecols=usecols, chunksize=300_000, low_memory=False):
                lat = pd.to_numeric(chunk["latitude"], errors="coerce")
                lon = pd.to_numeric(chunk["longitude"], errors="coerce")
                keep = lat.between(min_lat, max_lat) & lon.between(min_lon, max_lon)
                if not keep.any():
                    continue

                sub = chunk.loc[keep].copy()
                tm = sub["acq_time"].astype(str).str.strip().str.zfill(4)
                dt_text = sub["acq_date"].astype(str).str.strip() + " " + tm.str.slice(0, 2) + ":" + tm.str.slice(2, 4)
                sub["acq_datetime"] = pd.to_datetime(dt_text, format="%Y-%m-%d %H:%M", errors="coerce", utc=True)
                sub["latitude"] = pd.to_numeric(sub["latitude"], errors="coerce")
                sub["longitude"] = pd.to_numeric(sub["longitude"], errors="coerce")
                sub["frp"] = pd.to_numeric(sub["frp"], errors="coerce")

                sub = sub.dropna(subset=["acq_datetime", "latitude", "longitude"])
                sub = sub[(sub["acq_datetime"] >= start_dt) & (sub["acq_datetime"] <= end_dt)]
                if sub.empty:
                    continue
                rows.append(sub[["acq_datetime", "latitude", "longitude", "confidence", "frp"]])

    if not rows:
        return pd.DataFrame(columns=["acq_datetime", "latitude", "longitude", "confidence", "frp"])
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values("acq_datetime").reset_index(drop=True)
    return out


def list_goes_keys_for_range(idx: GoesIndex, start_dt: datetime, end_dt: datetime) -> list[tuple[datetime, str]]:
    by_key: dict[str, datetime] = {}
    cursor = start_dt.replace(minute=0, second=0, microsecond=0)
    end_hr = end_dt.replace(minute=0, second=0, microsecond=0)
    while cursor <= end_hr:
        year = cursor.year
        doy = cursor.timetuple().tm_yday
        hour = cursor.hour
        for ref in idx.list_hour(year, doy, hour):
            if start_dt <= ref.start_time_utc <= end_dt:
                by_key[ref.key] = ref.start_time_utc
        cursor += timedelta(hours=1)
    return sorted([(v, k) for k, v in by_key.items()], key=lambda x: x[0])


def build_regular_schedule(start_dt: datetime, end_dt: datetime, step_minutes: int, max_frames: int) -> list[datetime]:
    if step_minutes <= 0:
        raise ValueError("frame-step-minutes must be > 0")
    step = timedelta(minutes=step_minutes)
    out = []
    t = start_dt
    while t <= end_dt:
        out.append(t)
        t += step
    if max_frames > 0 and len(out) > max_frames:
        idx = np.linspace(0, len(out) - 1, max_frames, dtype=int)
        out = [out[i] for i in idx]
    return out


def nearest_goes_keys_for_schedule(
    schedule: list[datetime],
    goes_items: list[tuple[datetime, str]],
    max_abs_minutes: float,
) -> list[str | None]:
    if not goes_items:
        return [None] * len(schedule)

    ts = np.array([dt.timestamp() for dt, _k in goes_items], dtype=np.float64)
    keys = [k for _dt, k in goes_items]
    out: list[str | None] = []

    for dt in schedule:
        target = dt.timestamp()
        pos = int(np.searchsorted(ts, target))
        cand = []
        if 0 <= pos < len(ts):
            cand.append(pos)
        if pos - 1 >= 0:
            cand.append(pos - 1)
        if pos + 1 < len(ts):
            cand.append(pos + 1)

        if not cand:
            out.append(None)
            continue

        best_i = min(cand, key=lambda i: abs(ts[i] - target))
        diff_min = abs(ts[best_i] - target) / 60.0
        if diff_min <= max_abs_minutes:
            out.append(keys[best_i])
        else:
            out.append(None)
    return out


def iter_geometry_exteriors(geom):
    if geom.geom_type == "Polygon":
        yield geom.exterior
    elif geom.geom_type == "MultiPolygon":
        for p in geom.geoms:
            yield p.exterior


def lonlat_to_tile_xy(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    n = float(2**zoom)
    lat_clamped = max(-85.05112878, min(85.05112878, lat))
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat_clamped)
    y = (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n
    return x, y


def choose_osm_zoom(min_lon: float, min_lat: float, max_lon: float, max_lat: float, width: int, height: int) -> int:
    for z in range(14, 3, -1):
        x0, y0 = lonlat_to_tile_xy(min_lon, max_lat, z)
        x1, y1 = lonlat_to_tile_xy(max_lon, min_lat, z)
        px_w = abs(x1 - x0) * 256.0
        px_h = abs(y1 - y0) * 256.0
        if px_w <= width * 2.0 and px_h <= height * 2.0:
            return z
    return 4


def fetch_tile_png(url: str) -> np.ndarray:
    req = urllib.request.Request(url, headers={"User-Agent": "gribcheck/1.0 (animation)"})
    with urllib.request.urlopen(req, timeout=20.0) as resp:
        data = resp.read()
    return plt.imread(BytesIO(data), format="png")


def fetch_osm_tile_background(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    width: int,
    height: int,
) -> np.ndarray | None:
    z = choose_osm_zoom(min_lon, min_lat, max_lon, max_lat, width=width, height=height)
    x0f, y0f = lonlat_to_tile_xy(min_lon, max_lat, z)
    x1f, y1f = lonlat_to_tile_xy(max_lon, min_lat, z)
    x0 = int(math.floor(min(x0f, x1f)))
    x1 = int(math.floor(max(x0f, x1f)))
    y0 = int(math.floor(min(y0f, y1f)))
    y1 = int(math.floor(max(y0f, y1f)))

    tile_count = (x1 - x0 + 1) * (y1 - y0 + 1)
    if tile_count <= 0 or tile_count > 144:
        return None

    mosaic = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256, 4), dtype=np.float32)
    n = 2**z
    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            tx_wrapped = tx % n
            ty_clamped = min(max(ty, 0), n - 1)
            url = f"https://tile.openstreetmap.org/{z}/{tx_wrapped}/{ty_clamped}.png"
            tile = fetch_tile_png(url)
            iy = (ty - y0) * 256
            ix = (tx - x0) * 256
            if tile.shape[2] == 3:
                alpha = np.ones((tile.shape[0], tile.shape[1], 1), dtype=tile.dtype)
                tile = np.concatenate([tile, alpha], axis=2)
            mosaic[iy : iy + 256, ix : ix + 256, :] = tile[:, :, :4]

    # Crop mosaic to exact requested bbox in tile space.
    ox0 = int(round((x0f - x0) * 256.0))
    ox1 = int(round((x1f - x0) * 256.0))
    oy0 = int(round((y0f - y0) * 256.0))
    oy1 = int(round((y1f - y0) * 256.0))
    xmin = max(0, min(ox0, ox1))
    xmax = min(mosaic.shape[1], max(ox0, ox1))
    ymin = max(0, min(oy0, oy1))
    ymax = min(mosaic.shape[0], max(oy0, oy1))
    if xmax - xmin < 10 or ymax - ymin < 10:
        return mosaic
    return mosaic[ymin:ymax, xmin:xmax, :]


def fetch_osm_static_map(min_lon: float, min_lat: float, max_lon: float, max_lat: float, width: int, height: int) -> np.ndarray | None:
    params = {
        "bbox": f"{min_lon:.6f},{min_lat:.6f},{max_lon:.6f},{max_lat:.6f}",
        "size": f"{max(128, width)}x{max(128, height)}",
        "maptype": "mapnik",
    }
    url = "https://staticmap.openstreetmap.de/staticmap.php?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=25.0) as resp:
            data = resp.read()
        img = plt.imread(BytesIO(data), format="png")
        return img
    except Exception as e:
        print(f"Static OSM map endpoint failed ({e}); falling back to tile mosaic...")

    try:
        return fetch_osm_tile_background(min_lon, min_lat, max_lon, max_lat, width=width, height=height)
    except Exception as e:
        print(f"Tile mosaic background failed, continuing without map: {e}")
        return None


def nearest_idx_any_order(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.float64)
    if grid[0] <= grid[-1]:
        ii = np.searchsorted(grid, values, side="left")
        ii = np.clip(ii, 0, len(grid) - 1)
        prev = np.clip(ii - 1, 0, len(grid) - 1)
        ii = np.where(np.abs(grid[prev] - values) < np.abs(grid[ii] - values), prev, ii)
        return ii.astype(np.int32)

    # Descending grid: map through reversed ascending copy, then invert indices.
    rev = grid[::-1]
    jj = np.searchsorted(rev, values, side="left")
    jj = np.clip(jj, 0, len(rev) - 1)
    prev = np.clip(jj - 1, 0, len(rev) - 1)
    jj = np.where(np.abs(rev[prev] - values) < np.abs(rev[jj] - values), prev, jj)
    return (len(grid) - 1 - jj).astype(np.int32)


def main() -> None:
    args = parse_args()
    args.goes_cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.output.suffix.lower() != ".gif":
        print("Output suffix is not .gif; forcing GIF writer output.")

    date_hint = parse_date_opt(args.date_hint)

    print("Loading WFIGS fires...")
    fires = load_wfigs_fires(
        path=args.wfigs_geojson,
        start_year=2016,
        end_year=2026,
        min_acres=1000.0,
    )

    fire = choose_fire(
        fires=fires,
        fire_id=args.fire_id,
        fire_name=args.fire_name,
        center_lat=args.center_lat,
        center_lon=args.center_lon,
        date_hint=date_hint,
    )
    print(f"Selected fire: {fire.fire_id} | {fire.name} | {fire.start_date} -> {fire.end_date} | {fire.state}")

    start_d = parse_date_opt(args.start_date) or fire.start_date
    if args.end_date:
        end_d = parse_date_opt(args.end_date)
    elif args.use_fire_window:
        end_d = fire.end_date
    else:
        end_d = min(fire.end_date, start_d + timedelta(days=max(1, args.days_from_start)))
    if end_d < start_d:
        end_d = start_d

    start_dt = datetime.combine(start_d, time(0, 0), tzinfo=timezone.utc)
    end_dt = datetime.combine(end_d, time(23, 59), tzinfo=timezone.utc)
    print(f"Animation window UTC: {start_dt.isoformat()} to {end_dt.isoformat()}")

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
        raise RuntimeError("No GOES FDCF files found for requested range")

    schedule = build_regular_schedule(
        start_dt=start_dt,
        end_dt=end_dt,
        step_minutes=args.frame_step_minutes,
        max_frames=args.max_frames,
    )
    nearest_keys = nearest_goes_keys_for_schedule(schedule, goes_items, max_abs_minutes=25.0)
    usable = sum(1 for k in nearest_keys if k is not None)
    print(f"Frames: {len(schedule):,} | GOES-matched frames: {usable:,}")

    # Build lon/lat grid.
    lon_grid = np.linspace(minx, maxx, args.grid_nx, dtype=np.float64)
    lat_grid = np.linspace(miny, maxy, args.grid_ny, dtype=np.float64)
    lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)

    # Projection/grid mapping from first available GOES file.
    first_key = next(k for k in nearest_keys if k is not None)
    first_path = ensure_goes_file(args.goes_base_url, first_key, args.goes_cache_dir)
    with Dataset(first_path) as ds:
        xg = ds.variables["x"][:].astype(np.float64)
        yg = ds.variables["y"][:].astype(np.float64)
        proj = ds.variables["goes_imager_projection"]
        sat_h = float(proj.perspective_point_height)
        lon_0 = float(proj.longitude_of_projection_origin)
        semi_major = float(proj.semi_major_axis)
        semi_minor = float(proj.semi_minor_axis)

    crs_geos = pyproj.CRS.from_proj4(
        f"+proj=geos +h={sat_h} +lon_0={lon_0} +sweep=x +a={semi_major} +b={semi_minor} +no_defs"
    )
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs_geos, always_xy=True)
    x_m, y_m = transformer.transform(lon2d.ravel(), lat2d.ravel())
    xr = (np.asarray(x_m, dtype=np.float64) / sat_h).reshape(lat2d.shape)
    yr = (np.asarray(y_m, dtype=np.float64) / sat_h).reshape(lat2d.shape)

    valid = (
        np.isfinite(xr)
        & np.isfinite(yr)
        & (xr >= float(xg.min()))
        & (xr <= float(xg.max()))
        & (yr >= float(yg.min()))
        & (yr <= float(yg.max()))
    )

    xi = nearest_idx_any_order(xg, xr)
    yi = nearest_idx_any_order(yg, yr)

    # Precompute VIIRS held 12-hour bins.
    hold_hours = max(1, int(args.viirs_hold_hours))
    hold_sec = hold_hours * 3600
    n_slots = int(((end_dt - start_dt).total_seconds() // hold_sec) + 1)
    viirs_slot_masks = [np.zeros((args.grid_ny, args.grid_nx), dtype=np.uint8) for _ in range(n_slots)]
    viirs_slot_points_lon = [[] for _ in range(n_slots)]
    viirs_slot_points_lat = [[] for _ in range(n_slots)]

    if not viirs.empty:
        viirs_ts = viirs["acq_datetime"].astype("int64").to_numpy() / 1e9
        t0 = start_dt.timestamp()
        slot_idx = np.floor((viirs_ts - t0) / hold_sec).astype(np.int32)
        slot_idx = np.clip(slot_idx, 0, n_slots - 1)

        vx = viirs["longitude"].to_numpy(dtype=np.float64)
        vy = viirs["latitude"].to_numpy(dtype=np.float64)
        iix = np.clip(np.searchsorted(lon_grid, vx, side="left"), 0, len(lon_grid) - 1)
        iiy = np.clip(np.searchsorted(lat_grid, vy, side="left"), 0, len(lat_grid) - 1)

        for s, ix, iy, lon, lat in zip(slot_idx, iix, iiy, vx, vy, strict=True):
            viirs_slot_masks[int(s)][int(iy), int(ix)] = 1
            viirs_slot_points_lon[int(s)].append(float(lon))
            viirs_slot_points_lat[int(s)].append(float(lat))

    viirs_slot_cum = []
    vac = np.zeros((args.grid_ny, args.grid_nx), dtype=np.uint8)
    for m in viirs_slot_masks:
        vac = np.maximum(vac, m)
        viirs_slot_cum.append(vac.copy())

    frame_slot_idx = []
    for dt in schedule:
        s = int((dt - start_dt).total_seconds() // hold_sec)
        s = min(max(s, 0), n_slots - 1)
        frame_slot_idx.append(s)

    print("Preparing GOES fire masks...")
    goes_cache: dict[str, np.ndarray] = {}
    goes_current: list[np.ndarray] = []

    for i, key in enumerate(nearest_keys, start=1):
        if key is None:
            gmask = np.zeros((args.grid_ny, args.grid_nx), dtype=np.uint8)
        else:
            cached = goes_cache.get(key)
            if cached is None:
                path = ensure_goes_file(args.goes_base_url, key, args.goes_cache_dir)
                with Dataset(path) as ds:
                    if "Mask" not in ds.variables:
                        raise RuntimeError(f"GOES file missing Mask variable: {path}")
                    mask = np.asarray(ds.variables["Mask"][:])
                    vals = mask[yi, xi]
                    fire_like = np.isin(vals, GOES_FIRE_VALUES)
                    if "DQF" in ds.variables:
                        dqf = np.asarray(ds.variables["DQF"][:])
                        q = dqf[yi, xi]
                        fire_like = fire_like & (q <= 1)
                    gmask = (fire_like & valid).astype(np.uint8)
                goes_cache[key] = gmask
            else:
                gmask = cached
        goes_current.append(gmask)

        if i % 40 == 0:
            print(f"Prepared GOES frames: {i}/{len(nearest_keys)}")

    goes_cum = []
    gac = np.zeros((args.grid_ny, args.grid_nx), dtype=np.uint8)
    for m in goes_current:
        gac = np.maximum(gac, m)
        goes_cum.append(gac.copy())

    bg_img = None
    if args.map_background:
        print("Fetching static OSM map background...")
        bg_img = fetch_osm_static_map(minx, miny, maxx, maxy, args.map_width, args.map_height)

    print("Rendering animation...")
    extent = [minx, maxx, miny, maxy]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    ax_go, ax_vi, ax_cum = axes

    for ax in axes:
        if bg_img is not None:
            ax.imshow(bg_img, extent=extent, origin="upper", alpha=args.map_alpha, zorder=0)
        for ext in iter_geometry_exteriors(fire.geometry):
            x, y = ext.xy
            ax.plot(x, y, color="lime", linewidth=1.2, zorder=5)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")

    i0 = 0
    s0 = frame_slot_idx[i0]
    im_go = ax_go.imshow(goes_current[i0], origin="lower", extent=extent, cmap="inferno", vmin=0, vmax=1, alpha=0.62, zorder=2)
    im_vi = ax_vi.imshow(viirs_slot_masks[s0], origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=1, alpha=0.62, zorder=2)
    sc_vi = ax_vi.scatter([], [], s=7, c="#5ef4ff", alpha=0.88, linewidths=0.0, zorder=4)
    im_cg = ax_cum.imshow(goes_cum[i0], origin="lower", extent=extent, cmap="Blues", vmin=0, vmax=1, alpha=0.48, zorder=2)
    im_cv = ax_cum.imshow(viirs_slot_cum[s0], origin="lower", extent=extent, cmap="Reds", vmin=0, vmax=1, alpha=0.48, zorder=3)

    ax_go.set_title("GOES-18 FDCF Fire Mask (15-min)")
    ax_vi.set_title(f"VIIRS Pixels (held {hold_hours}h bins)")
    ax_cum.set_title("Cumulative Overlay (GOES blue, VIIRS red)")

    supt = fig.suptitle("")

    def update(frame_i: int):
        slot_i = frame_slot_idx[frame_i]
        im_go.set_data(goes_current[frame_i])
        im_vi.set_data(viirs_slot_masks[slot_i])
        im_cg.set_data(goes_cum[frame_i])
        im_cv.set_data(viirs_slot_cum[slot_i])

        lon_pts = viirs_slot_points_lon[slot_i]
        lat_pts = viirs_slot_points_lat[slot_i]
        if lon_pts:
            sc_vi.set_offsets(np.column_stack([lon_pts, lat_pts]))
        else:
            sc_vi.set_offsets(np.empty((0, 2), dtype=np.float64))

        slot_start = start_dt + timedelta(hours=slot_i * hold_hours)
        slot_end = min(end_dt, slot_start + timedelta(hours=hold_hours))
        supt.set_text(
            f"{fire.fire_id} | {fire.name} | frame {frame_i + 1}/{len(schedule)} | "
            f"GOES {schedule[frame_i].strftime('%Y-%m-%d %H:%M UTC')} | "
            f"VIIRS bin {slot_start.strftime('%m-%d %H:%M')} to {slot_end.strftime('%m-%d %H:%M')} UTC"
        )
        return im_go, im_vi, im_cg, im_cv, sc_vi, supt

    anim = FuncAnimation(fig, update, frames=len(schedule), interval=100, blit=False)

    out = args.output
    if out.suffix.lower() != ".gif":
        out = out.with_suffix(".gif")
    anim.save(out, writer=PillowWriter(fps=max(1, args.fps)))
    plt.close(fig)

    print(f"Animation written: {out}")
    print(f"GOES cache dir: {args.goes_cache_dir}")


if __name__ == "__main__":
    main()
