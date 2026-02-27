#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time as time_module
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
from netCDF4 import Dataset
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from shapely import points as shapely_points
from shapely.geometry import Point
from shapely.strtree import STRtree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.mtbs_wfigs_coherence_analysis import DAY0, day_int_from_date  # noqa: E402
from scripts.wfigs_viirs_stats import WfigsFire, load_wfigs_fires  # noqa: E402


GOES_XML_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


@dataclass
class GoesFileRef:
    key: str
    start_time_utc: datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sampled analysis of missing VIIRS detections: burnout vs cloud cover at overpass times"
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
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--min-acres", type=float, default=1000.0)
    parser.add_argument("--sample-fires", type=int, default=30)
    parser.add_argument("--max-days-per-fire", type=int, default=10)
    parser.add_argument("--points-per-fire", type=int, default=25)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument(
        "--goes-base-url",
        type=str,
        default="https://noaa-goes18.s3.amazonaws.com",
    )
    parser.add_argument(
        "--goes-prefix",
        type=str,
        default="ABI-L2-ACMC",
    )
    parser.add_argument(
        "--goes-cache-dir",
        type=Path,
        default=Path("/Users/lorn/Code/gribcheck/data/external/goes18_acmc_cache"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/lorn/Code/gribcheck/reports/wfigs_viirs_missing_causes_sample"),
    )
    return parser.parse_args()


def size_bucket(acres: float) -> str:
    if acres < 5000:
        return "1k-5k"
    if acres < 20000:
        return "5k-20k"
    if acres < 100000:
        return "20k-100k"
    return "100k+"


def build_sample(
    fires: list[WfigsFire],
    sample_fires: int,
    random_seed: int,
) -> list[WfigsFire]:
    rng = np.random.default_rng(random_seed)
    rows = []
    for i, f in enumerate(fires):
        c = f.geometry.centroid
        rows.append(
            {
                "idx": i,
                "fire": f,
                "bucket": size_bucket(f.size_acres),
                "lon": float(c.x),
                "lat": float(c.y),
            }
        )
    df = pd.DataFrame(rows)

    # GOES18 CONUS full coverage envelope from sample file metadata.
    in_footprint = (
        (df["lon"] >= -140.7)
        & (df["lon"] <= -49.0)
        & (df["lat"] >= 14.0)
        & (df["lat"] <= 53.0)
    )
    df = df[in_footprint].copy()
    if df.empty:
        return []

    groups = {k: g.copy() for k, g in df.groupby("bucket")}
    total = len(df)
    desired = {}
    for k, g in groups.items():
        desired[k] = max(1, int(round(sample_fires * len(g) / total)))

    while sum(desired.values()) > sample_fires:
        k = max(desired, key=lambda x: desired[x])
        if desired[k] > 1:
            desired[k] -= 1
        else:
            break
    while sum(desired.values()) < sample_fires:
        k = max(groups, key=lambda x: len(groups[x]) - desired.get(x, 0))
        if desired.get(k, 0) < len(groups[k]):
            desired[k] = desired.get(k, 0) + 1
        else:
            break

    chosen = []
    for k, g in groups.items():
        n = min(desired.get(k, 0), len(g))
        if n <= 0:
            continue
        sel = g.sample(n=n, random_state=int(rng.integers(0, 1_000_000)))
        chosen.extend(sel["fire"].tolist())
    return chosen


def build_fire_day_table(
    fires: list[WfigsFire],
    max_days_per_fire: int,
) -> pd.DataFrame:
    rows = []
    for i, f in enumerate(fires):
        end = min(f.end_date, f.start_date + timedelta(days=max_days_per_fire - 1))
        d = f.start_date
        while d <= end:
            rows.append(
                {
                    "fire_idx": i,
                    "fire_id": f.fire_id,
                    "state": f.state,
                    "size_acres": f.size_acres,
                    "size_bucket": size_bucket(f.size_acres),
                    "start_date": f.start_date,
                    "day_date": d,
                    "day_int": day_int_from_date(d),
                    "days_since_start": (d - f.start_date).days,
                    "centroid_lon": float(f.geometry.centroid.x),
                    "centroid_lat": float(f.geometry.centroid.y),
                }
            )
            d += timedelta(days=1)
    out = pd.DataFrame(rows)
    out["viirs_pixel_count"] = 0
    return out


def iterate_viirs_chunks(viirs_zip: Path, min_day_int: int, max_day_int: int, chunksize: int = 300_000):
    usecols = ["latitude", "longitude", "acq_date"]
    with zipfile.ZipFile(viirs_zip) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv") and "archive" in n.lower()]
        if not members:
            raise FileNotFoundError(f"No archive CSV in {viirs_zip}")
        member = members[0]
        with zf.open(member) as fp:
            for chunk in pd.read_csv(fp, usecols=usecols, chunksize=chunksize, low_memory=False):
                day_vals = pd.to_datetime(chunk["acq_date"], errors="coerce").to_numpy(dtype="datetime64[D]")
                lat = pd.to_numeric(chunk["latitude"], errors="coerce").to_numpy(dtype=np.float64)
                lon = pd.to_numeric(chunk["longitude"], errors="coerce").to_numpy(dtype=np.float64)
                valid = ~np.isnat(day_vals)
                valid &= np.isfinite(lat) & np.isfinite(lon)
                valid &= (lat >= -90) & (lat <= 90) & (lon >= -180) & (lon <= 180)
                if not np.any(valid):
                    continue
                day_int = (day_vals[valid] - DAY0).astype(np.int32)
                keep = (day_int >= min_day_int) & (day_int <= max_day_int)
                if not np.any(keep):
                    continue
                yield lon[valid][keep], lat[valid][keep], day_int[keep]


def assign_viirs_counts(
    fire_day_df: pd.DataFrame,
    fires: list[WfigsFire],
    viirs_zip: Path,
) -> pd.DataFrame:
    geoms = [f.geometry for f in fires]
    tree = STRtree(geoms)

    start_day = fire_day_df.groupby("fire_idx")["day_int"].min().to_numpy(dtype=np.int32)
    end_day = fire_day_df.groupby("fire_idx")["day_int"].max().to_numpy(dtype=np.int32)
    fire_size = fire_day_df.groupby("fire_idx")["size_acres"].first().to_numpy(dtype=np.float64)
    duration = end_day - start_day + 1

    row_lookup = {
        (int(r.fire_idx), int(r.day_int)): int(idx)
        for idx, r in fire_day_df[["fire_idx", "day_int"]].iterrows()
    }
    counts = np.zeros(len(fire_day_df), dtype=np.int32)

    min_day_int = int(fire_day_df["day_int"].min())
    max_day_int = int(fire_day_df["day_int"].max())

    chunk_i = 0
    for lon, lat, day_int in iterate_viirs_chunks(
        viirs_zip=viirs_zip,
        min_day_int=min_day_int,
        max_day_int=max_day_int,
    ):
        chunk_i += 1
        pts = shapely_points(lon, lat)
        pairs = tree.query(pts, predicate="within")
        if pairs.shape[1] == 0:
            continue

        p_idx = pairs[0].astype(np.int32)
        f_idx = pairs[1].astype(np.int32)
        d = day_int[p_idx]
        keep = (d >= start_day[f_idx]) & (d <= end_day[f_idx])
        if not np.any(keep):
            continue

        p_idx = p_idx[keep]
        f_idx = f_idx[keep]
        d = d[keep]

        # One fire assignment per VIIRS pixel: prefer smaller fire area and shorter duration.
        score = np.round(fire_size[f_idx]).astype(np.int64) * 10000 + duration[f_idx].astype(np.int64)
        order = np.lexsort((score, p_idx))
        p_sorted = p_idx[order]
        f_sorted = f_idx[order]
        d_sorted = d[order]
        first = np.r_[True, p_sorted[1:] != p_sorted[:-1]]
        f_sel = f_sorted[first]
        d_sel = d_sorted[first]

        for fi, di in zip(f_sel, d_sel, strict=True):
            idx = row_lookup.get((int(fi), int(di)))
            if idx is not None:
                counts[idx] += 1

        if chunk_i % 8 == 0:
            print(f"VIIRS chunks processed: {chunk_i}")

    out = fire_day_df.copy()
    out["viirs_pixel_count"] = counts
    out["has_viirs"] = (out["viirs_pixel_count"] > 0).astype(int)
    out["no_viirs"] = 1 - out["has_viirs"]
    return out


def dt_to_year_doy_hour(dt: datetime) -> tuple[int, int, int]:
    d = dt.date()
    return d.year, d.timetuple().tm_yday, dt.hour


def parse_goes_start_from_key(key: str) -> datetime | None:
    m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", key)
    if m is None:
        return None
    year, doy, hh, mm, ss = map(int, m.groups())
    d0 = date(year, 1, 1) + timedelta(days=doy - 1)
    return datetime.combine(d0, time(hh, mm, ss), tzinfo=timezone.utc)


def http_get_bytes(url: str, timeout_sec: float = 25.0, retries: int = 3, backoff_sec: float = 1.5) -> bytes:
    err: Exception | None = None
    for i in range(max(1, retries)):
        try:
            with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
                return resp.read()
        except Exception as e:  # pragma: no cover - transient network faults
            err = e
            if i < retries - 1:
                time_module.sleep(backoff_sec * (i + 1))
    raise RuntimeError(f"Failed HTTP GET after retries: {url}") from err


class GoesIndex:
    def __init__(self, base_url: str, product_prefix: str):
        self.base_url = base_url.rstrip("/")
        self.product_prefix = product_prefix
        self.cache: dict[tuple[int, int, int], list[GoesFileRef]] = {}

    def list_hour(self, year: int, doy: int, hour: int) -> list[GoesFileRef]:
        key = (year, doy, hour)
        if key in self.cache:
            return self.cache[key]

        prefix = f"{self.product_prefix}/{year}/{doy:03d}/{hour:02d}/"
        url = f"{self.base_url}/?prefix={prefix}"
        xml_text = http_get_bytes(url=url, timeout_sec=20.0, retries=3, backoff_sec=1.2)
        root = ET.fromstring(xml_text)
        out: list[GoesFileRef] = []
        for node in root.findall("s3:Contents", GOES_XML_NS):
            k_node = node.find("s3:Key", GOES_XML_NS)
            if k_node is None or not k_node.text:
                continue
            k = k_node.text.strip()
            if not k.endswith(".nc"):
                continue
            ts = parse_goes_start_from_key(k)
            if ts is None:
                continue
            out.append(GoesFileRef(key=k, start_time_utc=ts))
        out.sort(key=lambda r: r.start_time_utc)
        self.cache[key] = out
        return out

    def nearest(self, target_dt: datetime, max_abs_minutes: float = 45.0) -> tuple[str | None, float | None]:
        candidates: list[GoesFileRef] = []
        for shift in (-1, 0, 1):
            dt2 = target_dt + timedelta(hours=shift)
            y, doy, h = dt_to_year_doy_hour(dt2)
            candidates.extend(self.list_hour(y, doy, h))
        if not candidates:
            return None, None
        best = min(candidates, key=lambda r: abs((r.start_time_utc - target_dt).total_seconds()))
        diff_min = abs((best.start_time_utc - target_dt).total_seconds()) / 60.0
        if diff_min > max_abs_minutes:
            return None, None
        return best.key, diff_min


def ensure_goes_file(base_url: str, key: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fn = key.split("/")[-1]
    out_path = cache_dir / fn
    if out_path.exists() and out_path.stat().st_size > 1000:
        return out_path
    url = f"{base_url.rstrip('/')}/{key}"
    data = http_get_bytes(url=url, timeout_sec=35.0, retries=3, backoff_sec=2.0)
    out_path.write_bytes(data)
    return out_path


def build_fire_sample_indices(
    fires: list[WfigsFire],
    points_per_fire: int,
    rng: np.random.Generator,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    transformer: pyproj.Transformer,
    sat_height: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for f in fires:
        geom = f.geometry
        points = [geom.centroid]
        minx, miny, maxx, maxy = geom.bounds
        attempts = 0
        target = max(1, points_per_fire)
        while len(points) < target and attempts < target * 40:
            attempts += 1
            px = rng.uniform(minx, maxx)
            py = rng.uniform(miny, maxy)
            pt = Point(px, py)
            if geom.contains(pt):
                points.append(pt)
        lon = np.array([p.x for p in points], dtype=np.float64)
        lat = np.array([p.y for p in points], dtype=np.float64)
        x_m, y_m = transformer.transform(lon, lat)
        x_rad = np.asarray(x_m, dtype=np.float64) / sat_height
        y_rad = np.asarray(y_m, dtype=np.float64) / sat_height

        valid = (
            np.isfinite(x_rad)
            & np.isfinite(y_rad)
            & (x_rad >= float(x_grid.min()))
            & (x_rad <= float(x_grid.max()))
            & (y_rad >= float(y_grid.min()))
            & (y_rad <= float(y_grid.max()))
        )
        x_rad = x_rad[valid]
        y_rad = y_rad[valid]
        if x_rad.size == 0:
            out.append((np.array([], dtype=np.int32), np.array([], dtype=np.int32)))
            continue

        xi = np.searchsorted(x_grid, x_rad, side="left")
        xi = np.clip(xi, 0, len(x_grid) - 1)
        xi_prev = np.clip(xi - 1, 0, len(x_grid) - 1)
        choose_prev = np.abs(x_grid[xi_prev] - x_rad) < np.abs(x_grid[xi] - x_rad)
        xi = np.where(choose_prev, xi_prev, xi).astype(np.int32)

        yi = np.searchsorted(y_grid, y_rad, side="left")
        yi = np.clip(yi, 0, len(y_grid) - 1)
        yi_prev = np.clip(yi - 1, 0, len(y_grid) - 1)
        choose_prev = np.abs(y_grid[yi_prev] - y_rad) < np.abs(y_grid[yi] - y_rad)
        yi = np.where(choose_prev, yi_prev, yi).astype(np.int32)

        pairs = np.unique(np.column_stack((yi, xi)), axis=0)
        out.append((pairs[:, 1].astype(np.int32), pairs[:, 0].astype(np.int32)))
    return out


def roc_auc_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(y_score)
    sum_pos = float(np.sum(ranks[pos]))
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def fit_logistic(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n, p = X.shape
    X1 = np.column_stack([np.ones(n), X])

    def nll(beta: np.ndarray) -> float:
        z = X1 @ beta
        p_hat = expit(z)
        eps = 1e-12
        return float(-np.sum(y * np.log(p_hat + eps) + (1 - y) * np.log(1 - p_hat + eps)))

    def grad(beta: np.ndarray) -> np.ndarray:
        z = X1 @ beta
        p_hat = expit(z)
        return X1.T @ (p_hat - y)

    beta0 = np.zeros(p + 1, dtype=np.float64)
    res = minimize(nll, beta0, jac=grad, method="BFGS", options={"maxiter": 500, "gtol": 1e-7})
    if not res.success:
        raise RuntimeError(f"Logistic fit failed: {res.message}")
    return res.x


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    args.goes_cache_dir.mkdir(parents=True, exist_ok=True)

    print("Loading WFIGS fires...")
    fires_all = load_wfigs_fires(
        path=args.wfigs_geojson,
        start_year=args.start_year,
        end_year=args.end_year,
        min_acres=args.min_acres,
    )
    print(f"Filtered fires: {len(fires_all):,}")

    sample_fires = build_sample(
        fires=fires_all,
        sample_fires=args.sample_fires,
        random_seed=args.random_seed,
    )
    if not sample_fires:
        raise RuntimeError("No fires in GOES footprint after filtering.")
    print(f"Sampled fires: {len(sample_fires):,}")

    print("Building fire-day panel...")
    panel = build_fire_day_table(sample_fires, max_days_per_fire=args.max_days_per_fire)
    print(f"Fire-day rows: {len(panel):,}")

    print("Assigning VIIRS detections to sampled fire-days...")
    panel = assign_viirs_counts(panel, sample_fires, args.viirs_zip)

    # Overpass times: NOAA-20 approximate local equator crossing ~13:30 and 01:30.
    lon = panel["centroid_lon"].to_numpy(dtype=np.float64)
    day_hour_utc = (13.5 - lon / 15.0) % 24.0
    night_hour_utc = (1.5 - lon / 15.0) % 24.0
    base_dt = [
        datetime.combine(d, time(0, 0), tzinfo=timezone.utc)
        for d in panel["day_date"].tolist()
    ]
    panel["day_overpass_utc"] = [b + timedelta(hours=float(h)) for b, h in zip(base_dt, day_hour_utc, strict=True)]
    panel["night_overpass_utc"] = [b + timedelta(hours=float(h)) for b, h in zip(base_dt, night_hour_utc, strict=True)]

    print("Indexing GOES files by nearest overpass time...")
    goes_index = GoesIndex(base_url=args.goes_base_url, product_prefix=args.goes_prefix)

    day_keys = []
    day_dt_diff = []
    night_keys = []
    night_dt_diff = []
    for i, (dt_day, dt_night) in enumerate(zip(panel["day_overpass_utc"], panel["night_overpass_utc"], strict=True), start=1):
        k1, d1 = goes_index.nearest(dt_day)
        k2, d2 = goes_index.nearest(dt_night)
        day_keys.append(k1)
        day_dt_diff.append(d1)
        night_keys.append(k2)
        night_dt_diff.append(d2)
        if i % 50 == 0:
            print(f"Indexed GOES overpass matches: {i}/{len(panel)}")
    panel["goes_day_key"] = day_keys
    panel["goes_day_time_diff_min"] = day_dt_diff
    panel["goes_night_key"] = night_keys
    panel["goes_night_time_diff_min"] = night_dt_diff

    if panel["goes_day_key"].isna().all() and panel["goes_night_key"].isna().all():
        raise RuntimeError("No GOES files matched the sample overpass times.")

    # Build point->grid index mapping using one sample file.
    first_key = panel["goes_day_key"].dropna().iloc[0] if panel["goes_day_key"].notna().any() else panel["goes_night_key"].dropna().iloc[0]
    first_path = ensure_goes_file(args.goes_base_url, first_key, args.goes_cache_dir)
    with Dataset(first_path) as ds:
        x_grid = ds.variables["x"][:].astype(np.float64)
        y_grid = ds.variables["y"][:].astype(np.float64)
        proj = ds.variables["goes_imager_projection"]
        sat_height = float(proj.perspective_point_height)
        lon_0 = float(proj.longitude_of_projection_origin)
        semi_major = float(proj.semi_major_axis)
        semi_minor = float(proj.semi_minor_axis)

    crs_geos = pyproj.CRS.from_proj4(
        f"+proj=geos +h={sat_height} +lon_0={lon_0} +sweep=x +a={semi_major} +b={semi_minor} +no_defs"
    )
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs_geos, always_xy=True)

    print("Sampling points within fire perimeters and mapping to GOES grid...")
    fire_indices = build_fire_sample_indices(
        fires=sample_fires,
        points_per_fire=args.points_per_fire,
        rng=rng,
        x_grid=x_grid,
        y_grid=y_grid,
        transformer=transformer,
        sat_height=sat_height,
    )

    # Gather all file->fire requests.
    req: dict[str, set[int]] = {}
    for r in panel.itertuples(index=False):
        if isinstance(r.goes_day_key, str):
            req.setdefault(r.goes_day_key, set()).add(int(r.fire_idx))
        if isinstance(r.goes_night_key, str):
            req.setdefault(r.goes_night_key, set()).add(int(r.fire_idx))

    print(f"Computing cloud fractions from GOES files: {len(req):,} unique granules...")
    file_fire_cloud: dict[tuple[str, int], float] = {}
    processed = 0
    for key, fire_set in req.items():
        path = ensure_goes_file(args.goes_base_url, key, args.goes_cache_dir)
        with Dataset(path) as ds:
            bcm = ds.variables["BCM"][:]
            dqf = ds.variables["DQF"][:]
            for fi in fire_set:
                xs, ys = fire_indices[fi]
                if xs.size == 0:
                    file_fire_cloud[(key, fi)] = np.nan
                    continue
                bcm_vals = bcm[ys, xs]
                dqf_vals = dqf[ys, xs]
                valid = dqf_vals != 255
                if not np.any(valid):
                    file_fire_cloud[(key, fi)] = np.nan
                    continue
                cloudy = (bcm_vals[valid] == 1)
                file_fire_cloud[(key, fi)] = float(np.mean(cloudy))
        processed += 1
        if processed % 40 == 0:
            print(f"GOES files processed: {processed}/{len(req)}")

    day_cloud = []
    night_cloud = []
    for r in panel.itertuples(index=False):
        fi = int(r.fire_idx)
        dc = file_fire_cloud.get((r.goes_day_key, fi), np.nan) if isinstance(r.goes_day_key, str) else np.nan
        nc = file_fire_cloud.get((r.goes_night_key, fi), np.nan) if isinstance(r.goes_night_key, str) else np.nan
        day_cloud.append(dc)
        night_cloud.append(nc)
    panel["cloud_day_frac"] = day_cloud
    panel["cloud_night_frac"] = night_cloud
    panel["cloud_mean_frac"] = panel[["cloud_day_frac", "cloud_night_frac"]].mean(axis=1, skipna=True)
    panel["cloud_max_frac"] = panel[["cloud_day_frac", "cloud_night_frac"]].max(axis=1, skipna=True)
    panel["both_overpasses_cloudy_50pct"] = (
        (panel["cloud_day_frac"] >= 0.5) & (panel["cloud_night_frac"] >= 0.5)
    ).astype(int)

    # Modeling subset with both cloud passes available.
    model_df = panel.dropna(subset=["cloud_day_frac", "cloud_night_frac"]).copy()
    X_raw = model_df[["days_since_start", "cloud_day_frac", "cloud_night_frac"]].to_numpy(dtype=np.float64)
    y = model_df["no_viirs"].to_numpy(dtype=np.float64)

    mu = X_raw.mean(axis=0)
    sigma = X_raw.std(axis=0)
    sigma[sigma == 0] = 1.0
    X = (X_raw - mu) / sigma

    beta = fit_logistic(X, y)
    z = beta[0] + X @ beta[1:]
    p_hat = expit(z)
    auc = roc_auc_manual(y.astype(int), p_hat)

    # Counterfactual attribution among missing cases.
    miss_mask = y == 1
    X_miss = X[miss_mask]
    z_full = beta[0] + X_miss @ beta[1:]
    p_full = expit(z_full)

    X_no_cloud = X_miss.copy()
    X_no_cloud[:, 1] = 0.0
    X_no_cloud[:, 2] = 0.0
    p_no_cloud = expit(beta[0] + X_no_cloud @ beta[1:])

    X_no_burn = X_miss.copy()
    X_no_burn[:, 0] = 0.0
    p_no_burn = expit(beta[0] + X_no_burn @ beta[1:])

    cloud_contrib = p_full - p_no_cloud
    burn_contrib = p_full - p_no_burn
    dominant = np.where(cloud_contrib > burn_contrib, "cloud_dominant", "burnout_dominant")

    miss_rows = model_df.loc[miss_mask].copy()
    miss_rows["cloud_contrib"] = cloud_contrib
    miss_rows["burnout_contrib"] = burn_contrib
    miss_rows["dominant_cause"] = dominant

    # Summaries.
    missing_rate_overall = float(model_df["no_viirs"].mean())
    cloudy = model_df["both_overpasses_cloudy_50pct"] == 1
    missing_rate_cloudy = float(model_df.loc[cloudy, "no_viirs"].mean()) if cloudy.any() else np.nan
    missing_rate_not_cloudy = float(model_df.loc[~cloudy, "no_viirs"].mean()) if (~cloudy).any() else np.nan
    odds_ratio_per_sd = np.exp(beta[1:])

    cause_counts = miss_rows["dominant_cause"].value_counts().to_dict()
    n_missing = int(miss_mask.sum())
    cloud_pct = float(cause_counts.get("cloud_dominant", 0) / max(n_missing, 1))
    burn_pct = float(cause_counts.get("burnout_dominant", 0) / max(n_missing, 1))

    summary = {
        "sample_fires": int(len(sample_fires)),
        "sample_fire_days": int(len(panel)),
        "model_rows_with_cloud_data": int(len(model_df)),
        "no_viirs_rate": missing_rate_overall,
        "no_viirs_rate_when_both_overpasses_cloudy_ge_50pct": missing_rate_cloudy,
        "no_viirs_rate_when_not_both_cloudy": missing_rate_not_cloudy,
        "logistic_auc": float(auc),
        "coef_intercept": float(beta[0]),
        "coef_days_since_start_z": float(beta[1]),
        "coef_cloud_day_z": float(beta[2]),
        "coef_cloud_night_z": float(beta[3]),
        "odds_ratio_per_1sd_days_since_start": float(odds_ratio_per_sd[0]),
        "odds_ratio_per_1sd_cloud_day": float(odds_ratio_per_sd[1]),
        "odds_ratio_per_1sd_cloud_night": float(odds_ratio_per_sd[2]),
        "missing_rows_count": n_missing,
        "cloud_dominant_missing_fraction": cloud_pct,
        "burnout_dominant_missing_fraction": burn_pct,
    }

    # Save outputs.
    panel.to_csv(out_dir / "sample_fire_day_panel.csv", index=False)
    model_df.to_csv(out_dir / "sample_fire_day_model_rows.csv", index=False)
    miss_rows.to_csv(out_dir / "missing_day_attribution.csv", index=False)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    lines = []
    lines.append("# Missing VIIRS Cause Analysis (Sampled)")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- WFIGS years: {args.start_year}-{args.end_year}")
    lines.append(f"- Minimum size: {args.min_acres:g} acres")
    lines.append(f"- Sampled fires: {len(sample_fires)}")
    lines.append(f"- Days per fire sampled (max): {args.max_days_per_fire}")
    lines.append("- Cloud product: GOES-18 ABI-L2-ACMC (`BCM`) at nearest times to NOAA-20 day/night overpasses")
    lines.append("")
    lines.append("## Key Results")
    lines.append(f"- Fire-day rows (all sampled): {len(panel)}")
    lines.append(f"- Fire-day rows with both cloud overpasses matched: {len(model_df)}")
    lines.append(f"- Overall no-VIIRS rate (model rows): {missing_rate_overall:.3f}")
    if np.isfinite(missing_rate_cloudy):
        lines.append(f"- No-VIIRS rate when both overpasses are >=50% cloudy: {missing_rate_cloudy:.3f}")
    if np.isfinite(missing_rate_not_cloudy):
        lines.append(f"- No-VIIRS rate when both overpasses are not >=50% cloudy: {missing_rate_not_cloudy:.3f}")
    lines.append(f"- Logistic AUC (days-since-start + cloud-day + cloud-night): {auc:.3f}")
    lines.append(f"- Odds ratio per +1 SD `days_since_start`: {odds_ratio_per_sd[0]:.3f}")
    lines.append(f"- Odds ratio per +1 SD `cloud_day_frac`: {odds_ratio_per_sd[1]:.3f}")
    lines.append(f"- Odds ratio per +1 SD `cloud_night_frac`: {odds_ratio_per_sd[2]:.3f}")
    lines.append("")
    lines.append("## Missing-Day Attribution")
    lines.append(f"- Missing rows attributed cloud-dominant: {cause_counts.get('cloud_dominant', 0)} ({cloud_pct:.3f})")
    lines.append(f"- Missing rows attributed burnout-dominant: {cause_counts.get('burnout_dominant', 0)} ({burn_pct:.3f})")
    lines.append("")
    lines.append("## Notes")
    lines.append("- This is a sample analysis, not full-corpus.")
    lines.append("- Overpass times are approximated from NOAA-20 local crossing times and fire longitude.")
    lines.append("- Cloud fraction is estimated from sampled points within each fire perimeter.")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("Done.")
    print(f"Report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
