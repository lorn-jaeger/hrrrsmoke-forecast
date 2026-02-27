#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from io import BytesIO
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from matplotlib.animation import FuncAnimation, PillowWriter
from netCDF4 import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.wfigs_viirs_missing_causes_sample import GoesIndex, ensure_goes_file, http_get_bytes  # noqa: E402
from scripts.wfigs_viirs_stats import WfigsFire, load_wfigs_fires  # noqa: E402

S3_XML_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


@dataclass
class ViirsFileRef:
    stem: str
    start_time_utc: datetime
    key: str
    size_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Animate what satellites see: GOES-18 RGB (15-min cadence) vs VIIRS true-color RGB "
            "(M5/M4/M3, held 12h bins) for one WFIGS fire."
        )
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
        help="VIIRS FIRMS archive zip (used for slot timing hints only).",
    )

    parser.add_argument("--fire-id", type=str, default=None, help="Exact WFIGS fire ID")
    parser.add_argument("--fire-name", type=str, default=None, help="Case-insensitive incident-name contains match")
    parser.add_argument("--center-lat", type=float, default=None)
    parser.add_argument("--center-lon", type=float, default=None)
    parser.add_argument("--date-hint", type=str, default=None, help="YYYY-MM-DD")

    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--days-from-start", type=int, default=1)
    parser.add_argument("--use-fire-window", action="store_true", help="Use full WFIGS start->end window")

    parser.add_argument("--grid-nx", type=int, default=320)
    parser.add_argument("--grid-ny", type=int, default=220)
    parser.add_argument("--bbox-pad-deg", type=float, default=0.35)
    parser.add_argument("--frame-step-minutes", type=int, default=15)
    parser.add_argument("--viirs-hold-hours", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=0, help="Optional hard cap; <=0 means no cap")
    parser.add_argument("--fps", type=int, default=8)

    parser.add_argument("--goes-base-url", type=str, default="https://noaa-goes18.s3.amazonaws.com")
    parser.add_argument(
        "--goes-prefix",
        type=str,
        default="ABI-L2-CMIPC",
        help="GOES product prefix. ABI-L2-CMIPC (CONUS) is much smaller than CMIPF.",
    )
    parser.add_argument("--goes-max-abs-minutes", type=float, default=25.0)
    parser.add_argument(
        "--goes-cache-dir",
        type=Path,
        default=Path("/Users/lorn/Code/gribcheck/data/external/goes18_cmip_cache"),
    )

    parser.add_argument("--viirs-base-url", type=str, default="https://noaa-nesdis-n20-pds.s3.amazonaws.com")
    parser.add_argument(
        "--viirs-cache-dir",
        type=Path,
        default=Path("/Users/lorn/Code/gribcheck/data/external/viirs_rgb_cache"),
    )
    parser.add_argument("--viirs-search-hours", type=float, default=18.0)
    parser.add_argument("--viirs-min-file-bytes", type=int, default=1_000_000)

    parser.add_argument("--map-width", type=int, default=1280)
    parser.add_argument("--map-height", type=int, default=840)
    parser.add_argument("--map-alpha", type=float, default=0.78)
    parser.set_defaults(map_background=True)
    parser.add_argument("--map-background", dest="map_background", action="store_true")
    parser.add_argument("--no-map-background", dest="map_background", action="store_false")

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Users/lorn/Code/gribcheck/reports/figures/goes_vs_viirs_satellite_rgb.gif"),
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


def list_s3_keys_paginated(base_url: str, prefix: str, suffix: str = "") -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    marker: str | None = None
    base = base_url.rstrip("/") + "/"

    while True:
        params = {"prefix": prefix, "max-keys": "1000"}
        if marker:
            params["marker"] = marker
        url = base + "?" + urllib.parse.urlencode(params)
        xml_bytes = http_get_bytes(url=url, timeout_sec=25.0, retries=3, backoff_sec=1.5)
        root = ET.fromstring(xml_bytes)

        last_key = None
        for node in root.findall("s3:Contents", S3_XML_NS):
            k_node = node.find("s3:Key", S3_XML_NS)
            s_node = node.find("s3:Size", S3_XML_NS)
            if k_node is None or not k_node.text:
                continue
            k = k_node.text.strip()
            if suffix and not k.endswith(suffix):
                continue
            size = int(s_node.text.strip()) if (s_node is not None and s_node.text) else 0
            out.append((k, size))
            last_key = k

        trunc_node = root.find("s3:IsTruncated", S3_XML_NS)
        is_trunc = bool(trunc_node is not None and trunc_node.text and trunc_node.text.strip().lower() == "true")
        if not is_trunc:
            break

        next_marker_node = root.find("s3:NextMarker", S3_XML_NS)
        if next_marker_node is not None and next_marker_node.text:
            marker = next_marker_node.text.strip()
        else:
            marker = last_key
        if not marker:
            break

    return out


def parse_goes_start_from_key(key: str) -> datetime | None:
    m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", key)
    if m is None:
        return None
    year, doy, hh, mm, ss = map(int, m.groups())
    d0 = date(year, 1, 1) + timedelta(days=doy - 1)
    return datetime.combine(d0, time(hh, mm, ss), tzinfo=timezone.utc)


def parse_viirs_stem_and_start(filename: str) -> tuple[str, datetime] | None:
    m = re.match(
        r"^(SVM0[345]|GMODO)_j01_d(\d{8})_t(\d{7})_e(\d{7})_b(\d{5})_c\d+_oeac_ops\.h5$",
        filename,
    )
    if m is None:
        return None
    d8 = m.group(2)
    t7 = m.group(3)
    b5 = m.group(5)
    stem = f"j01_d{d8}_t{t7}_e{m.group(4)}_b{b5}"
    d = datetime.strptime(d8 + t7[:6], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return stem, d


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


def list_goes_rgb_triplets_for_range(idx: GoesIndex, start_dt: datetime, end_dt: datetime) -> list[tuple[datetime, str, str, str]]:
    by_time: dict[datetime, dict[str, str]] = {}
    cursor = start_dt.replace(minute=0, second=0, microsecond=0)
    end_hr = end_dt.replace(minute=0, second=0, microsecond=0)
    while cursor <= end_hr:
        y = cursor.year
        doy = cursor.timetuple().tm_yday
        h = cursor.hour
        for ref in idx.list_hour(y, doy, h):
            dt = ref.start_time_utc
            if not (start_dt <= dt <= end_dt):
                continue
            m = re.search(r"M\dC(\d{2})", ref.key)
            if m is None:
                continue
            ch = m.group(1)
            if ch not in {"01", "02", "03"}:
                continue
            by_time.setdefault(dt, {})[ch] = ref.key
        cursor += timedelta(hours=1)

    out: list[tuple[datetime, str, str, str]] = []
    for dt in sorted(by_time):
        d = by_time[dt]
        if "01" in d and "02" in d and "03" in d:
            out.append((dt, d["01"], d["02"], d["03"]))
    return out


def nearest_goes_triplets_for_schedule(
    schedule: list[datetime],
    triplets: list[tuple[datetime, str, str, str]],
    max_abs_minutes: float,
) -> list[tuple[str, str, str] | None]:
    if not triplets:
        return [None] * len(schedule)

    ts = np.array([dt.timestamp() for dt, _c01, _c02, _c03 in triplets], dtype=np.float64)
    keys = [(c01, c02, c03) for _dt, c01, c02, c03 in triplets]
    out: list[tuple[str, str, str] | None] = []

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
        out.append(keys[best_i] if diff_min <= max_abs_minutes else None)
    return out


class ViirsRgbIndex:
    def __init__(self, base_url: str, min_file_bytes: int):
        self.base_url = base_url.rstrip("/")
        self.min_file_bytes = int(min_file_bytes)
        self.cache: dict[tuple[str, date], dict[str, ViirsFileRef]] = {}

    def _load_day_product(self, product: str, day: date) -> dict[str, ViirsFileRef]:
        ck = (product, day)
        cached = self.cache.get(ck)
        if cached is not None:
            return cached

        prefix = f"{product}/{day:%Y/%m/%d}/"
        rows = list_s3_keys_paginated(self.base_url, prefix=prefix, suffix=".h5")
        out: dict[str, ViirsFileRef] = {}
        for key, size in rows:
            if size < self.min_file_bytes:
                continue
            fn = key.split("/")[-1]
            parsed = parse_viirs_stem_and_start(fn)
            if parsed is None:
                continue
            stem, dt = parsed
            out[stem] = ViirsFileRef(stem=stem, start_time_utc=dt, key=key, size_bytes=size)
        self.cache[ck] = out
        return out

    def nearest_triplets(self, target_dt: datetime, search_hours: float) -> list[tuple[float, ViirsFileRef, ViirsFileRef, ViirsFileRef, ViirsFileRef]]:
        days = sorted(
            {
                (target_dt - timedelta(days=1)).date(),
                target_dt.date(),
                (target_dt + timedelta(days=1)).date(),
            }
        )
        m3_all: list[ViirsFileRef] = []
        for d in days:
            m3_all.extend(self._load_day_product("VIIRS-M3-SDR", d).values())
        if not m3_all:
            return []

        max_diff_sec = float(search_hours) * 3600.0
        out = []
        for m3 in m3_all:
            diff_sec = abs((m3.start_time_utc - target_dt).total_seconds())
            if diff_sec > max_diff_sec:
                continue
            day = m3.start_time_utc.date()
            m4 = self._load_day_product("VIIRS-M4-SDR", day).get(m3.stem)
            m5 = self._load_day_product("VIIRS-M5-SDR", day).get(m3.stem)
            geo = self._load_day_product("VIIRS-MOD-GEO", day).get(m3.stem)
            if m4 is None or m5 is None or geo is None:
                continue
            out.append((diff_sec, m3, m4, m5, geo))
        out.sort(key=lambda r: r[0])
        return out


def ensure_viirs_file(base_url: str, key: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / key.split("/")[-1]
    if out.exists() and out.stat().st_size > 1000:
        return out
    data = http_get_bytes(
        url=f"{base_url.rstrip('/')}/{key}",
        timeout_sec=45.0,
        retries=3,
        backoff_sec=2.0,
    )
    out.write_bytes(data)
    return out


def load_viirs_times_for_bbox_time(
    viirs_zip: Path,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.Series:
    usecols = ["latitude", "longitude", "acq_date", "acq_time"]
    rows = []
    with zipfile.ZipFile(viirs_zip) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv") and "archive" in n.lower()]
        if not members:
            return pd.Series(dtype="datetime64[ns, UTC]")
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
                dt = pd.to_datetime(dt_text, format="%Y-%m-%d %H:%M", errors="coerce", utc=True)
                dt = dt[(dt >= start_dt) & (dt <= end_dt)]
                if len(dt):
                    rows.append(dt)
    if not rows:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.concat(rows, ignore_index=True).sort_values().reset_index(drop=True)


def nearest_idx_any_order(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.float64)
    if grid[0] <= grid[-1]:
        ii = np.searchsorted(grid, values, side="left")
        ii = np.clip(ii, 0, len(grid) - 1)
        prev = np.clip(ii - 1, 0, len(grid) - 1)
        ii = np.where(np.abs(grid[prev] - values) < np.abs(grid[ii] - values), prev, ii)
        return ii.astype(np.int32)

    rev = grid[::-1]
    jj = np.searchsorted(rev, values, side="left")
    jj = np.clip(jj, 0, len(rev) - 1)
    prev = np.clip(jj - 1, 0, len(rev) - 1)
    jj = np.where(np.abs(rev[prev] - values) < np.abs(rev[jj] - values), prev, jj)
    return (len(grid) - 1 - jj).astype(np.int32)


def sample_goes_cmi_to_grid(
    path: Path,
    xr: np.ndarray,
    yr: np.ndarray,
    map_cache: dict[tuple[int, int, float, float, float, float], tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int]],
) -> np.ndarray:
    with Dataset(path) as ds:
        if "CMI" not in ds.variables:
            raise RuntimeError(f"GOES file missing CMI variable: {path}")
        xg = ds.variables["x"][:].astype(np.float64)
        yg = ds.variables["y"][:].astype(np.float64)
        mk = (len(xg), len(yg), float(xg[0]), float(xg[-1]), float(yg[0]), float(yg[-1]))
        mapping = map_cache.get(mk)
        if mapping is None:
            valid_mask = (
                np.isfinite(xr)
                & np.isfinite(yr)
                & (xr >= float(xg.min()))
                & (xr <= float(xg.max()))
                & (yr >= float(yg.min()))
                & (yr <= float(yg.max()))
            )
            if np.any(valid_mask):
                xi = nearest_idx_any_order(xg, xr)
                yi = nearest_idx_any_order(yg, yr)
                x0 = int(np.min(xi[valid_mask]))
                x1 = int(np.max(xi[valid_mask]))
                y0 = int(np.min(yi[valid_mask]))
                y1 = int(np.max(yi[valid_mask]))
            else:
                xi = np.zeros_like(xr, dtype=np.int32)
                yi = np.zeros_like(yr, dtype=np.int32)
                x0 = x1 = y0 = y1 = 0
            mapping = (xi, yi, valid_mask, x0, x1, y0, y1)
            map_cache[mk] = mapping
        xi, yi, valid_mask, x0, x1, y0, y1 = mapping

        var = ds.variables["CMI"]
        var.set_auto_maskandscale(False)
        raw = np.asarray(var[y0 : y1 + 1, x0 : x1 + 1], dtype=np.float32)
        fill = float(getattr(var, "_FillValue", -1.0))
        scale = float(getattr(var, "scale_factor", 1.0))
        offset = float(getattr(var, "add_offset", 0.0))
    raw[raw == fill] = np.nan
    raw[raw < 0] = np.nan
    cmi = raw * scale + offset
    vals = np.full(xr.shape, np.nan, dtype=np.float32)
    if np.any(valid_mask):
        iy = yi[valid_mask] - y0
        ix = xi[valid_mask] - x0
        vals[valid_mask] = cmi[iy, ix]
    return vals


def to_uint8_rgb(rgb_f32: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb_f32, 0.0, 1.0)
    rgb = np.power(rgb, 1.0 / 2.2)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def read_viirs_reflectance(path: Path, band_num: int) -> np.ndarray:
    group = f"All_Data/VIIRS-M{band_num}-SDR_All"
    with h5py.File(path, "r") as f:
        raw = np.asarray(f[f"{group}/Reflectance"][:], dtype=np.float32)
        factors = np.asarray(f[f"{group}/ReflectanceFactors"][:], dtype=np.float32)
    scale = float(factors[0]) if len(factors) >= 1 else 1.0
    offset = float(factors[1]) if len(factors) >= 2 else 0.0
    ref = raw * scale + offset
    # Typical VIIRS SDR invalid/sentinel range.
    invalid = (raw <= 0) | (raw >= 65528)
    ref[invalid] = np.nan
    return ref


def read_viirs_geo(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        lat = np.asarray(f["All_Data/VIIRS-MOD-GEO_All/Latitude"][:], dtype=np.float32)
        lon = np.asarray(f["All_Data/VIIRS-MOD-GEO_All/Longitude"][:], dtype=np.float32)
        sza = np.asarray(f["All_Data/VIIRS-MOD-GEO_All/SolarZenithAngle"][:], dtype=np.float32)
    return lon, lat, sza


def grid_viirs_rgb(
    lon: np.ndarray,
    lat: np.ndarray,
    sza: np.ndarray,
    m3: np.ndarray,
    m4: np.ndarray,
    m5: np.ndarray,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    nx: int,
    ny: int,
) -> tuple[np.ndarray | None, float]:
    cosz = np.cos(np.deg2rad(sza.astype(np.float32)))
    valid = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & np.isfinite(m3)
        & np.isfinite(m4)
        & np.isfinite(m5)
        & np.isfinite(cosz)
        & (cosz > 0.05)
        & (lon >= min_lon)
        & (lon <= max_lon)
        & (lat >= min_lat)
        & (lat <= max_lat)
    )
    if not np.any(valid):
        return None, 0.0

    lon1 = lon[valid].astype(np.float64)
    lat1 = lat[valid].astype(np.float64)
    denom = cosz[valid].astype(np.float32)
    b = m3[valid].astype(np.float32) / denom
    g = m4[valid].astype(np.float32) / denom
    r = m5[valid].astype(np.float32) / denom

    ix = np.floor((lon1 - min_lon) / (max_lon - min_lon) * (nx - 1)).astype(np.int32)
    iy = np.floor((lat1 - min_lat) / (max_lat - min_lat) * (ny - 1)).astype(np.int32)
    keep = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & np.isfinite(r) & np.isfinite(g) & np.isfinite(b)
    if not np.any(keep):
        return None, 0.0
    ix = ix[keep]
    iy = iy[keep]
    r = r[keep]
    g = g[keep]
    b = b[keep]

    count = np.zeros((ny, nx), dtype=np.int32)
    rs = np.zeros((ny, nx), dtype=np.float32)
    gs = np.zeros((ny, nx), dtype=np.float32)
    bs = np.zeros((ny, nx), dtype=np.float32)
    np.add.at(count, (iy, ix), 1)
    np.add.at(rs, (iy, ix), r)
    np.add.at(gs, (iy, ix), g)
    np.add.at(bs, (iy, ix), b)

    rgb = np.zeros((ny, nx, 3), dtype=np.float32)
    hit = count > 0
    rgb[..., 0][hit] = rs[hit] / count[hit]
    rgb[..., 1][hit] = gs[hit] / count[hit]
    rgb[..., 2][hit] = bs[hit] / count[hit]

    coverage = float(np.mean(hit))
    return to_uint8_rgb(np.clip(rgb, 0.0, 1.0)), coverage


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
        return plt.imread(BytesIO(data), format="png")
    except Exception as e:
        print(f"Static OSM map endpoint failed ({e}); falling back to tile mosaic...")

    try:
        return fetch_osm_tile_background(min_lon, min_lat, max_lon, max_lat, width=width, height=height)
    except Exception as e:
        print(f"Tile mosaic background failed, continuing without map: {e}")
        return None


def main() -> None:
    args = parse_args()
    args.goes_cache_dir.mkdir(parents=True, exist_ok=True)
    args.viirs_cache_dir.mkdir(parents=True, exist_ok=True)
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

    lon_grid = np.linspace(minx, maxx, args.grid_nx, dtype=np.float64)
    lat_grid = np.linspace(miny, maxy, args.grid_ny, dtype=np.float64)
    lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)
    extent = [minx, maxx, miny, maxy]

    print("Listing GOES RGB files...")
    goes_idx = GoesIndex(base_url=args.goes_base_url, product_prefix=args.goes_prefix)
    goes_triplets = list_goes_rgb_triplets_for_range(goes_idx, start_dt=start_dt, end_dt=end_dt)
    if not goes_triplets:
        raise RuntimeError(f"No GOES RGB triplets found in range for prefix {args.goes_prefix}")
    print(f"GOES triplets in range: {len(goes_triplets):,}")

    schedule = build_regular_schedule(
        start_dt=start_dt,
        end_dt=end_dt,
        step_minutes=args.frame_step_minutes,
        max_frames=args.max_frames,
    )
    nearest_goes = nearest_goes_triplets_for_schedule(schedule, goes_triplets, max_abs_minutes=args.goes_max_abs_minutes)
    usable_goes = sum(1 for k in nearest_goes if k is not None)
    print(f"Frames: {len(schedule):,} | GOES-matched frames: {usable_goes:,}")

    first_triplet = next(t for t in nearest_goes if t is not None)
    c01_first, c02_first, c03_first = first_triplet
    first_path = ensure_goes_file(args.goes_base_url, c02_first, args.goes_cache_dir)
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

    valid_goes_c02 = (
        np.isfinite(xr)
        & np.isfinite(yr)
        & (xr >= float(xg.min()))
        & (xr <= float(xg.max()))
        & (yr >= float(yg.min()))
        & (yr <= float(yg.max()))
    )
    if not np.any(valid_goes_c02):
        raise RuntimeError("Requested bbox is outside GOES footprint for this product.")

    xi_c02 = nearest_idx_any_order(xg, xr)
    yi_c02 = nearest_idx_any_order(yg, yr)
    x0 = int(np.min(xi_c02[valid_goes_c02]))
    x1 = int(np.max(xi_c02[valid_goes_c02]))
    y0 = int(np.min(yi_c02[valid_goes_c02]))
    y1 = int(np.max(yi_c02[valid_goes_c02]))
    print(f"GOES subset indexes: x[{x0}:{x1}] y[{y0}:{y1}]")

    print("Preparing GOES RGB frames...")
    goes_rgb_cache: dict[tuple[str, str, str], np.ndarray] = {}
    goes_map_cache: dict[tuple[int, int, float, float, float, float], tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int]] = {}
    goes_frames: list[np.ndarray] = []
    for i, trip in enumerate(nearest_goes, start=1):
        if trip is None:
            goes_frames.append(np.zeros((args.grid_ny, args.grid_nx, 3), dtype=np.uint8))
            continue
        cached = goes_rgb_cache.get(trip)
        if cached is None:
            c01_key, c02_key, c03_key = trip
            p01 = ensure_goes_file(args.goes_base_url, c01_key, args.goes_cache_dir)
            p02 = ensure_goes_file(args.goes_base_url, c02_key, args.goes_cache_dir)
            p03 = ensure_goes_file(args.goes_base_url, c03_key, args.goes_cache_dir)

            r = sample_goes_cmi_to_grid(p02, xr, yr, goes_map_cache)
            b = sample_goes_cmi_to_grid(p01, xr, yr, goes_map_cache)
            n = sample_goes_cmi_to_grid(p03, xr, yr, goes_map_cache)

            g = 0.45 * r + 0.10 * n + 0.45 * b
            rgb = np.dstack([r, g, b]).astype(np.float32)
            cached = to_uint8_rgb(np.clip(rgb, 0.0, 1.0))
            goes_rgb_cache[trip] = cached
        goes_frames.append(cached)
        if i % 30 == 0:
            print(f"Prepared GOES frames: {i}/{len(nearest_goes)}")

    hold_hours = max(1, int(args.viirs_hold_hours))
    hold_sec = hold_hours * 3600
    n_slots = int(((end_dt - start_dt).total_seconds() // hold_sec) + 1)
    frame_slot_idx = []
    for dt in schedule:
        s = int((dt - start_dt).total_seconds() // hold_sec)
        s = min(max(s, 0), n_slots - 1)
        frame_slot_idx.append(s)

    print("Loading VIIRS FIRMS timing hints...")
    viirs_times = load_viirs_times_for_bbox_time(
        viirs_zip=args.viirs_zip,
        min_lon=minx,
        min_lat=miny,
        max_lon=maxx,
        max_lat=maxy,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    print(f"VIIRS timing hints in window: {len(viirs_times):,}")

    slot_targets: list[datetime] = []
    slot_start_end: list[tuple[datetime, datetime]] = []
    for s in range(n_slots):
        ss = start_dt + timedelta(seconds=s * hold_sec)
        ee = min(end_dt, ss + timedelta(seconds=hold_sec))
        slot_start_end.append((ss, ee))
        if not viirs_times.empty:
            sub = viirs_times[(viirs_times >= ss) & (viirs_times < ee)]
            if len(sub):
                mid = sub.iloc[len(sub) // 2].to_pydatetime()
                slot_targets.append(mid.astimezone(timezone.utc))
                continue
        slot_targets.append(ss + (ee - ss) / 2)

    print("Preparing VIIRS RGB slot images...")
    viirs_idx = ViirsRgbIndex(base_url=args.viirs_base_url, min_file_bytes=args.viirs_min_file_bytes)
    viirs_slots: list[np.ndarray] = []
    viirs_slot_label: list[str] = []
    viirs_rgb_cache: dict[str, np.ndarray] = {}
    for s, target_dt in enumerate(slot_targets):
        candidates = viirs_idx.nearest_triplets(target_dt=target_dt, search_hours=args.viirs_search_hours)
        slot_img = np.zeros((args.grid_ny, args.grid_nx, 3), dtype=np.uint8)
        label = "No VIIRS RGB"
        for _diff_sec, m3, m4, m5, geo in candidates[:12]:
            if m3.stem in viirs_rgb_cache:
                slot_img = viirs_rgb_cache[m3.stem]
                label = f"VIIRS {m3.start_time_utc.strftime('%Y-%m-%d %H:%M UTC')}"
                break

            p3 = ensure_viirs_file(args.viirs_base_url, m3.key, args.viirs_cache_dir)
            p4 = ensure_viirs_file(args.viirs_base_url, m4.key, args.viirs_cache_dir)
            p5 = ensure_viirs_file(args.viirs_base_url, m5.key, args.viirs_cache_dir)
            pg = ensure_viirs_file(args.viirs_base_url, geo.key, args.viirs_cache_dir)

            try:
                m3_ref = read_viirs_reflectance(p3, 3)
                m4_ref = read_viirs_reflectance(p4, 4)
                m5_ref = read_viirs_reflectance(p5, 5)
                lon, lat, sza = read_viirs_geo(pg)
                rgb_u8, coverage = grid_viirs_rgb(
                    lon=lon,
                    lat=lat,
                    sza=sza,
                    m3=m3_ref,
                    m4=m4_ref,
                    m5=m5_ref,
                    min_lon=minx,
                    min_lat=miny,
                    max_lon=maxx,
                    max_lat=maxy,
                    nx=args.grid_nx,
                    ny=args.grid_ny,
                )
            except Exception:
                rgb_u8, coverage = None, 0.0
            if rgb_u8 is None:
                continue
            if coverage < 0.0008:
                # Swath likely missed this bbox.
                continue
            slot_img = rgb_u8
            label = f"VIIRS {m3.start_time_utc.strftime('%Y-%m-%d %H:%M UTC')}"
            viirs_rgb_cache[m3.stem] = slot_img
            break

        viirs_slots.append(slot_img)
        viirs_slot_label.append(label)
        if (s + 1) % 4 == 0 or (s + 1) == n_slots:
            print(f"Prepared VIIRS slots: {s + 1}/{n_slots}")

    bg_img = None
    if args.map_background:
        print("Fetching static OSM map background...")
        bg_img = fetch_osm_static_map(minx, miny, maxx, maxy, args.map_width, args.map_height)

    print("Rendering animation...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    ax_go, ax_vi = axes
    for ax in axes:
        if bg_img is not None:
            ax.imshow(bg_img, extent=extent, origin="upper", alpha=args.map_alpha, zorder=0)
        for ext in iter_geometry_exteriors(fire.geometry):
            x, y = ext.xy
            ax.plot(x, y, color="yellow", linewidth=1.0, zorder=5)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")

    i0 = 0
    s0 = frame_slot_idx[i0]
    im_go = ax_go.imshow(goes_frames[i0], origin="lower", extent=extent, zorder=2)
    im_vi = ax_vi.imshow(viirs_slots[s0], origin="lower", extent=extent, zorder=2)
    ax_go.set_title("GOES-18 RGB (C02 + synthetic G + C01)")
    ax_vi.set_title(f"VIIRS True Color (M5/M4/M3, held {hold_hours}h)")
    supt = fig.suptitle("")

    def update(frame_i: int):
        slot_i = frame_slot_idx[frame_i]
        im_go.set_data(goes_frames[frame_i])
        im_vi.set_data(viirs_slots[slot_i])
        ss, ee = slot_start_end[slot_i]
        supt.set_text(
            f"{fire.fire_id} | {fire.name} | frame {frame_i + 1}/{len(schedule)} | "
            f"GOES {schedule[frame_i].strftime('%Y-%m-%d %H:%M UTC')} | "
            f"{viirs_slot_label[slot_i]} | slot {ss.strftime('%m-%d %H:%M')} - {ee.strftime('%m-%d %H:%M')} UTC"
        )
        return im_go, im_vi, supt

    anim = FuncAnimation(fig, update, frames=len(schedule), interval=100, blit=False)
    out = args.output
    if out.suffix.lower() != ".gif":
        out = out.with_suffix(".gif")
    anim.save(out, writer=PillowWriter(fps=max(1, args.fps)))
    plt.close(fig)

    print(f"Animation written: {out}")
    print(f"GOES cache dir: {args.goes_cache_dir}")
    print(f"VIIRS cache dir: {args.viirs_cache_dir}")


if __name__ == "__main__":
    main()
