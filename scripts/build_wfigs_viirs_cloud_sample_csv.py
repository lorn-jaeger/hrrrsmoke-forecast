#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import zipfile
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
from netCDF4 import Dataset
from shapely import points as shapely_points
from shapely.geometry import Point
from shapely.strtree import STRtree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.mtbs_wfigs_coherence_analysis import DAY0, day_int_from_date  # noqa: E402
from scripts.wfigs_viirs_missing_causes_sample import GoesIndex, ensure_goes_file  # noqa: E402
from scripts.wfigs_viirs_stats import WfigsFire, load_wfigs_fires  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sampled fire-day CSV with VIIRS category counts + cloudiness at two overpasses"
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
    parser.add_argument("--start-date", type=str, default="2021-01-01")
    parser.add_argument("--end-date", type=str, default="2025-12-31")
    parser.add_argument("--min-acres", type=float, default=5000.0)
    parser.add_argument("--sample-rows", type=int, default=250)
    parser.add_argument("--points-per-fire", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=19)
    parser.add_argument(
        "--goes-base-url",
        type=str,
        default="https://noaa-goes18.s3.amazonaws.com",
    )
    parser.add_argument(
        "--goes-prefix",
        type=str,
        default="ABI-L2-ACMC",
        help="Cloud product prefix. Default uses GOES-18 cloud mask.",
    )
    parser.add_argument(
        "--goes-cache-dir",
        type=Path,
        default=Path("/Users/lorn/Code/gribcheck/data/external/goes18_acmc_cache"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/Users/lorn/Code/gribcheck/reports/wfigs_viirs_cloud_sample_2021_2025_5000ac.csv"),
    )
    return parser.parse_args()


def classify_conf(series: pd.Series) -> np.ndarray:
    text = series.astype(str).str.strip().str.lower()
    out = np.full(len(text), "u", dtype="<U1")
    first = text.str.slice(0, 1)
    out[first == "h"] = "h"
    out[first == "n"] = "n"
    out[first == "l"] = "l"
    return out


def sample_fire_days(
    fires: list[WfigsFire],
    sample_rows: int,
    start_date: date,
    end_date: date,
    rng: np.random.Generator,
) -> pd.DataFrame:
    meta = []
    for i, f in enumerate(fires):
        s = max(f.start_date, start_date)
        e = min(f.end_date, end_date)
        if e < s:
            continue
        d = (e - s).days + 1
        if d <= 0:
            continue
        meta.append((i, s, e, d))
    if not meta:
        return pd.DataFrame()

    fire_idx = np.array([m[0] for m in meta], dtype=np.int32)
    s_dates = [m[1] for m in meta]
    durations = np.array([m[3] for m in meta], dtype=np.int64)
    cum = np.cumsum(durations)
    total_days = int(cum[-1])

    selected: set[tuple[int, int]] = set()
    target = int(sample_rows)
    while len(selected) < target:
        draw = rng.integers(0, total_days, size=max(target * 3, 2000), endpoint=False)
        fi_pos = np.searchsorted(cum, draw, side="right")
        prev = np.where(fi_pos > 0, cum[fi_pos - 1], 0)
        offset = draw - prev
        for p, off in zip(fi_pos, offset, strict=True):
            fidx = int(fire_idx[p])
            d = s_dates[p] + timedelta(days=int(off))
            selected.add((fidx, day_int_from_date(d)))
            if len(selected) >= target:
                break
        if len(selected) >= target:
            break

    rows = []
    for fidx, day_int in sorted(selected):
        f = fires[fidx]
        day_d = date.fromisoformat(str(DAY0 + np.timedelta64(day_int, "D")))
        rows.append(
            {
                "fire_idx": fidx,
                "fire_id": f.fire_id,
                "fire_name": f.name,
                "state": f.state,
                "size_acres": f.size_acres,
                "start_date": f.start_date,
                "end_date": f.end_date,
                "day_date": day_d,
                "day_int": int(day_int),
                "days_since_start": (day_d - f.start_date).days,
                "centroid_lon": float(f.geometry.centroid.x),
                "centroid_lat": float(f.geometry.centroid.y),
            }
        )
    return pd.DataFrame(rows).sort_values(["day_date", "fire_id"]).reset_index(drop=True)


def iterate_viirs_chunks(
    viirs_zip: Path,
    min_day: int,
    max_day: int,
    bbox: tuple[float, float, float, float],
    chunksize: int = 300_000,
):
    usecols = ["latitude", "longitude", "acq_date", "confidence", "daynight", "frp"]
    min_lon, min_lat, max_lon, max_lat = bbox
    with zipfile.ZipFile(viirs_zip) as zf:
        member = [n for n in zf.namelist() if n.lower().endswith(".csv") and "archive" in n.lower()][0]
        with zf.open(member) as fp:
            for chunk in pd.read_csv(fp, usecols=usecols, chunksize=chunksize, low_memory=False):
                day_vals = pd.to_datetime(chunk["acq_date"], errors="coerce").to_numpy(dtype="datetime64[D]")
                lat = pd.to_numeric(chunk["latitude"], errors="coerce").to_numpy(dtype=np.float64)
                lon = pd.to_numeric(chunk["longitude"], errors="coerce").to_numpy(dtype=np.float64)
                frp = pd.to_numeric(chunk["frp"], errors="coerce").to_numpy(dtype=np.float64)
                conf = classify_conf(chunk["confidence"])
                dn = chunk["daynight"].astype(str).str.strip().str.upper().str.slice(0, 1).to_numpy(dtype="<U1")

                valid = ~np.isnat(day_vals)
                valid &= np.isfinite(lat) & np.isfinite(lon) & np.isfinite(frp)
                valid &= (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)
                valid &= frp >= 0.0
                if not np.any(valid):
                    continue

                day_int = (day_vals[valid] - DAY0).astype(np.int32)
                keep = (day_int >= min_day) & (day_int <= max_day)
                if not np.any(keep):
                    continue

                yield (
                    lon[valid][keep],
                    lat[valid][keep],
                    day_int[keep],
                    conf[valid][keep],
                    dn[valid][keep],
                    frp[valid][keep],
                )


def assign_viirs_counts(panel: pd.DataFrame, fires: list[WfigsFire], viirs_zip: Path) -> pd.DataFrame:
    fire_ids = sorted(panel["fire_idx"].unique().tolist())
    fire_map = {fi: idx for idx, fi in enumerate(fire_ids)}
    geoms = [fires[fi].geometry for fi in fire_ids]
    areas = np.array([fires[fi].size_acres for fi in fire_ids], dtype=np.float64)
    tree = STRtree(geoms)

    row_lookup = {
        (int(r.fire_idx), int(r.day_int)): int(i)
        for i, r in panel[["fire_idx", "day_int"]].iterrows()
    }

    n = len(panel)
    total = np.zeros(n, dtype=np.int32)
    c_h = np.zeros(n, dtype=np.int32)
    c_n = np.zeros(n, dtype=np.int32)
    c_l = np.zeros(n, dtype=np.int32)
    c_u = np.zeros(n, dtype=np.int32)
    c_day = np.zeros(n, dtype=np.int32)
    c_night = np.zeros(n, dtype=np.int32)
    frp_sum = np.zeros(n, dtype=np.float64)

    min_day = int(panel["day_int"].min())
    max_day = int(panel["day_int"].max())

    # Tight bbox around sampled fires to reduce VIIRS scan.
    b = np.array([fires[fi].geometry.bounds for fi in fire_ids], dtype=np.float64)
    bbox = (float(np.min(b[:, 0])) - 0.2, float(np.min(b[:, 1])) - 0.2, float(np.max(b[:, 2])) + 0.2, float(np.max(b[:, 3])) + 0.2)

    chunk_i = 0
    for lon, lat, day_int, conf, dn, frp in iterate_viirs_chunks(
        viirs_zip=viirs_zip,
        min_day=min_day,
        max_day=max_day,
        bbox=bbox,
    ):
        chunk_i += 1
        pts = shapely_points(lon, lat)
        pairs = tree.query(pts, predicate="within")
        if pairs.shape[1] == 0:
            continue

        p_idx = pairs[0].astype(np.int32)
        f_local = pairs[1].astype(np.int32)
        d = day_int[p_idx]
        order = np.lexsort((areas[f_local], p_idx))
        p_sorted = p_idx[order]
        f_sorted = f_local[order]
        d_sorted = d[order]

        conf_sorted = conf[p_sorted]
        dn_sorted = dn[p_sorted]
        frp_sorted = frp[p_sorted]

        prev = -1
        assigned = False
        for p, fl, di, cf, dd, fr in zip(p_sorted, f_sorted, d_sorted, conf_sorted, dn_sorted, frp_sorted, strict=True):
            if p != prev:
                prev = int(p)
                assigned = False
            if assigned:
                continue
            fi_global = int(fire_ids[int(fl)])
            row = row_lookup.get((fi_global, int(di)))
            if row is None:
                continue
            total[row] += 1
            if cf == "h":
                c_h[row] += 1
            elif cf == "n":
                c_n[row] += 1
            elif cf == "l":
                c_l[row] += 1
            else:
                c_u[row] += 1
            if dd == "D":
                c_day[row] += 1
            elif dd == "N":
                c_night[row] += 1
            frp_sum[row] += float(fr)
            assigned = True

        if chunk_i % 8 == 0:
            print(f"VIIRS chunks processed: {chunk_i}")

    out = panel.copy()
    out["viirs_pixels_total"] = total
    out["viirs_pixels_conf_h"] = c_h
    out["viirs_pixels_conf_n"] = c_n
    out["viirs_pixels_conf_l"] = c_l
    out["viirs_pixels_conf_u"] = c_u
    out["viirs_pixels_day"] = c_day
    out["viirs_pixels_night"] = c_night
    out["viirs_frp_sum_mw"] = frp_sum
    out["viirs_frp_mean_mw"] = np.where(total > 0, frp_sum / total, np.nan)
    return out


def build_fire_point_indices(
    fire_ids: list[int],
    fires: list[WfigsFire],
    points_per_fire: int,
    rng: np.random.Generator,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    transformer: pyproj.Transformer,
    sat_height: float,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fi in fire_ids:
        geom = fires[fi].geometry
        pts = [geom.centroid]
        minx, miny, maxx, maxy = geom.bounds
        tries = 0
        while len(pts) < max(1, points_per_fire) and tries < points_per_fire * 40:
            tries += 1
            px = rng.uniform(minx, maxx)
            py = rng.uniform(miny, maxy)
            p = Point(px, py)
            if geom.contains(p):
                pts.append(p)

        lon = np.array([p.x for p in pts], dtype=np.float64)
        lat = np.array([p.y for p in pts], dtype=np.float64)
        xm, ym = transformer.transform(lon, lat)
        xr = np.asarray(xm, dtype=np.float64) / sat_height
        yr = np.asarray(ym, dtype=np.float64) / sat_height
        valid = (
            np.isfinite(xr)
            & np.isfinite(yr)
            & (xr >= float(x_grid.min()))
            & (xr <= float(x_grid.max()))
            & (yr >= float(y_grid.min()))
            & (yr <= float(y_grid.max()))
        )
        xr = xr[valid]
        yr = yr[valid]
        if xr.size == 0:
            out[fi] = (np.array([], dtype=np.int32), np.array([], dtype=np.int32))
            continue

        xi = np.searchsorted(x_grid, xr, side="left")
        xi = np.clip(xi, 0, len(x_grid) - 1)
        xi_prev = np.clip(xi - 1, 0, len(x_grid) - 1)
        xi = np.where(np.abs(x_grid[xi_prev] - xr) < np.abs(x_grid[xi] - xr), xi_prev, xi).astype(np.int32)

        yi = np.searchsorted(y_grid, yr, side="left")
        yi = np.clip(yi, 0, len(y_grid) - 1)
        yi_prev = np.clip(yi - 1, 0, len(y_grid) - 1)
        yi = np.where(np.abs(y_grid[yi_prev] - yr) < np.abs(y_grid[yi] - yr), yi_prev, yi).astype(np.int32)

        uniq = np.unique(np.column_stack((yi, xi)), axis=0)
        out[fi] = (uniq[:, 1].astype(np.int32), uniq[:, 0].astype(np.int32))
    return out


def add_cloudiness(panel: pd.DataFrame, fires: list[WfigsFire], args: argparse.Namespace, rng: np.random.Generator) -> pd.DataFrame:
    out = panel.copy()

    # Approx NOAA-20 local overpass times.
    lon = out["centroid_lon"].to_numpy(dtype=np.float64)
    day_hour_utc = (13.5 - lon / 15.0) % 24.0
    night_hour_utc = (1.5 - lon / 15.0) % 24.0
    base_dt = [datetime.combine(d, time(0, 0), tzinfo=timezone.utc) for d in out["day_date"].tolist()]
    out["day_overpass_utc"] = [b + timedelta(hours=float(h)) for b, h in zip(base_dt, day_hour_utc, strict=True)]
    out["night_overpass_utc"] = [b + timedelta(hours=float(h)) for b, h in zip(base_dt, night_hour_utc, strict=True)]

    idx = GoesIndex(base_url=args.goes_base_url, product_prefix=args.goes_prefix)
    day_keys, day_diff, night_keys, night_diff = [], [], [], []
    for i, (d1, d2) in enumerate(zip(out["day_overpass_utc"], out["night_overpass_utc"], strict=True), start=1):
        k1, m1 = idx.nearest(d1, max_abs_minutes=45.0)
        k2, m2 = idx.nearest(d2, max_abs_minutes=45.0)
        day_keys.append(k1)
        day_diff.append(m1)
        night_keys.append(k2)
        night_diff.append(m2)
        if i % 100 == 0:
            print(f"GOES nearest-key matched rows: {i}/{len(out)}")
    out["goes_day_key"] = day_keys
    out["goes_day_time_diff_min"] = day_diff
    out["goes_night_key"] = night_keys
    out["goes_night_time_diff_min"] = night_diff

    # Build projection from first available file.
    some_key = out["goes_day_key"].dropna()
    if some_key.empty:
        some_key = out["goes_night_key"].dropna()
    if some_key.empty:
        out["cloud_day_frac"] = np.nan
        out["cloud_night_frac"] = np.nan
        return out

    first_path = ensure_goes_file(args.goes_base_url, str(some_key.iloc[0]), args.goes_cache_dir)
    with Dataset(first_path) as ds:
        x_grid = ds.variables["x"][:].astype(np.float64)
        y_grid = ds.variables["y"][:].astype(np.float64)
        proj = ds.variables["goes_imager_projection"]
        h = float(proj.perspective_point_height)
        lon0 = float(proj.longitude_of_projection_origin)
        a = float(proj.semi_major_axis)
        b = float(proj.semi_minor_axis)

    crs_geos = pyproj.CRS.from_proj4(f"+proj=geos +h={h} +lon_0={lon0} +sweep=x +a={a} +b={b} +no_defs")
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs_geos, always_xy=True)

    fire_ids = sorted(out["fire_idx"].unique().tolist())
    fire_pts = build_fire_point_indices(
        fire_ids=fire_ids,
        fires=fires,
        points_per_fire=args.points_per_fire,
        rng=rng,
        x_grid=x_grid,
        y_grid=y_grid,
        transformer=transformer,
        sat_height=h,
    )

    req: dict[str, set[int]] = {}
    for r in out.itertuples(index=False):
        if isinstance(r.goes_day_key, str):
            req.setdefault(r.goes_day_key, set()).add(int(r.fire_idx))
        if isinstance(r.goes_night_key, str):
            req.setdefault(r.goes_night_key, set()).add(int(r.fire_idx))

    key_fire_cloud: dict[tuple[str, int], float] = {}
    for i, (key, fs) in enumerate(req.items(), start=1):
        path = ensure_goes_file(args.goes_base_url, key, args.goes_cache_dir)
        with Dataset(path) as ds:
            bcm = ds.variables["BCM"][:]
            dqf = ds.variables["DQF"][:]
            for fi in fs:
                xs, ys = fire_pts[fi]
                if xs.size == 0:
                    key_fire_cloud[(key, fi)] = np.nan
                    continue
                bvals = bcm[ys, xs]
                qvals = dqf[ys, xs]
                valid = qvals != 255
                if not np.any(valid):
                    key_fire_cloud[(key, fi)] = np.nan
                    continue
                key_fire_cloud[(key, fi)] = float(np.mean(bvals[valid] == 1))
        if i % 50 == 0:
            print(f"GOES files processed: {i}/{len(req)}")

    day_cloud, night_cloud = [], []
    for r in out.itertuples(index=False):
        fi = int(r.fire_idx)
        dc = key_fire_cloud.get((r.goes_day_key, fi), np.nan) if isinstance(r.goes_day_key, str) else np.nan
        nc = key_fire_cloud.get((r.goes_night_key, fi), np.nan) if isinstance(r.goes_night_key, str) else np.nan
        day_cloud.append(dc)
        night_cloud.append(nc)
    out["cloud_day_frac"] = day_cloud
    out["cloud_night_frac"] = night_cloud
    return out


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.goes_cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    start_d = date.fromisoformat(args.start_date)
    end_d = date.fromisoformat(args.end_date)

    print("Loading WFIGS fires...")
    fires = load_wfigs_fires(
        path=args.wfigs_geojson,
        start_year=start_d.year,
        end_year=end_d.year,
        min_acres=args.min_acres,
    )
    print(f"Loaded fires before footprint filter: {len(fires):,}")

    # GOES-18 CONUS cloud-mask footprint extent from product metadata.
    fires = [
        f for f in fires
        if (-140.7 <= f.geometry.centroid.x <= -49.0) and (14.0 <= f.geometry.centroid.y <= 53.0)
    ]
    print(f"Fires in GOES-18 footprint: {len(fires):,}")
    if not fires:
        raise RuntimeError("No eligible fires after filtering.")

    print("Sampling fire-days...")
    panel = sample_fire_days(
        fires=fires,
        sample_rows=args.sample_rows,
        start_date=start_d,
        end_date=end_d,
        rng=rng,
    )
    if panel.empty:
        raise RuntimeError("No sample rows generated.")
    print(f"Sample rows: {len(panel):,}")

    print("Assigning VIIRS counts by day/category...")
    panel = assign_viirs_counts(panel, fires=fires, viirs_zip=args.viirs_zip)

    print("Computing cloudiness at overpasses...")
    panel = add_cloudiness(panel, fires=fires, args=args, rng=rng)

    # Order columns for easier analysis.
    cols = [
        "fire_id",
        "fire_name",
        "state",
        "size_acres",
        "start_date",
        "end_date",
        "day_date",
        "days_since_start",
        "viirs_pixels_total",
        "viirs_pixels_conf_h",
        "viirs_pixels_conf_n",
        "viirs_pixels_conf_l",
        "viirs_pixels_conf_u",
        "viirs_pixels_day",
        "viirs_pixels_night",
        "viirs_frp_sum_mw",
        "viirs_frp_mean_mw",
        "cloud_day_frac",
        "cloud_night_frac",
        "goes_day_time_diff_min",
        "goes_night_time_diff_min",
        "centroid_lon",
        "centroid_lat",
        "fire_idx",
        "day_int",
    ]
    for c in cols:
        if c not in panel.columns:
            panel[c] = np.nan
    panel = panel[cols].sort_values(["day_date", "fire_id"]).reset_index(drop=True)
    panel.to_csv(args.output_csv, index=False)

    summary = {
        "rows": int(len(panel)),
        "fires": int(panel["fire_id"].nunique()),
        "date_min": str(panel["day_date"].min()),
        "date_max": str(panel["day_date"].max()),
        "mean_viirs_pixels": float(panel["viirs_pixels_total"].mean()),
        "pct_rows_no_viirs": float((panel["viirs_pixels_total"] == 0).mean()),
        "cloud_day_nonnull_frac": float(panel["cloud_day_frac"].notna().mean()),
        "cloud_night_nonnull_frac": float(panel["cloud_night_frac"].notna().mean()),
    }
    with args.output_csv.with_suffix(".summary.json").open("w", encoding="utf-8") as fp:
        import json
        json.dump(summary, fp, indent=2)

    print("Done.")
    print(f"CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
