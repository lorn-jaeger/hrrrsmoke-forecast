#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from shapely import contains, points as shapely_points
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.wfigs_viirs_stats import WfigsFire, load_wfigs_fires  # noqa: E402


S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
TIME_RE = re.compile(r"_s(\d{15})_e(\d{15})_c")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find VIIRS overpass drop-and-rebound events and test whether VIIRS cloud mask "
            "inside fire polygons spikes during the drop."
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
    )
    parser.add_argument("--start-date", type=str, default="2023-01-01")
    parser.add_argument("--end-date", type=str, default="2025-12-31")
    parser.add_argument("--min-acres", type=float, default=5000.0)
    parser.add_argument("--event-gap-minutes", type=int, default=45)
    parser.add_argument("--max-gap-hours-triplet", type=float, default=40.0)
    parser.add_argument("--min-prev-pixels", type=int, default=15)
    parser.add_argument("--drop-frac-max", type=float, default=0.35)
    parser.add_argument("--rebound-factor-min", type=float, default=2.0)
    parser.add_argument("--rebound-prev-frac-min", type=float, default=0.5)
    parser.add_argument("--max-fires", type=int, default=60)
    parser.add_argument("--events-per-fire", type=int, default=1)
    parser.add_argument("--max-events", type=int, default=0, help="Optional hard cap on selected events (0 = no cap).")
    parser.add_argument("--max-cloud-time-diff-min", type=float, default=30.0)
    parser.add_argument(
        "--cloud-base-url",
        type=str,
        default="https://noaa-nesdis-n20-pds.s3.amazonaws.com",
    )
    parser.add_argument(
        "--cloud-prefix",
        type=str,
        default="VIIRS-JRR-CloudMask",
    )
    parser.add_argument(
        "--cloud-cache-dir",
        type=Path,
        default=Path("/Users/lorn/Code/gribcheck/data/external/viirs_jrr_cloudmask_cache"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/lorn/Code/gribcheck/reports/viirs_drop_rebound_viirs_clouds"),
    )
    return parser.parse_args()


def parse_viirs_datetimes(acq_date: pd.Series, acq_time: pd.Series) -> np.ndarray:
    time_str = acq_time.astype(str).str.strip().str.zfill(4)
    dt = pd.to_datetime(
        acq_date.astype(str) + " " + time_str.str.slice(0, 2) + ":" + time_str.str.slice(2, 4),
        errors="coerce",
        utc=True,
    )
    return dt.to_numpy(dtype="datetime64[m]")


def iterate_viirs_chunks(viirs_zip: Path, chunksize: int = 300_000):
    usecols = ["latitude", "longitude", "acq_date", "acq_time"]
    with zipfile.ZipFile(viirs_zip) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv") and "archive" in n.lower()]
        if not members:
            raise FileNotFoundError(f"No archive CSV found in {viirs_zip}")
        member = members[0]
        with zf.open(member) as fp:
            for chunk in pd.read_csv(fp, usecols=usecols, chunksize=chunksize, low_memory=False):
                lat = pd.to_numeric(chunk["latitude"], errors="coerce").to_numpy(dtype=np.float64)
                lon = pd.to_numeric(chunk["longitude"], errors="coerce").to_numpy(dtype=np.float64)
                dt_m = parse_viirs_datetimes(chunk["acq_date"], chunk["acq_time"])
                valid = np.isfinite(lat) & np.isfinite(lon) & ~np.isnat(dt_m)
                if not np.any(valid):
                    continue
                yield lon[valid], lat[valid], dt_m[valid]


def build_fire_minute_counts(
    fires: list[WfigsFire],
    viirs_zip: Path,
    start_dt: np.datetime64,
    end_dt: np.datetime64,
) -> pd.DataFrame:
    geoms = [f.geometry for f in fires]
    areas = np.array([f.size_acres for f in fires], dtype=np.float64)
    fire_ids = np.arange(len(fires), dtype=np.int32)
    tree = STRtree(geoms)

    b = np.array([g.bounds for g in geoms], dtype=np.float64)
    min_lon = float(np.min(b[:, 0]) - 0.2)
    min_lat = float(np.min(b[:, 1]) - 0.2)
    max_lon = float(np.max(b[:, 2]) + 0.2)
    max_lat = float(np.max(b[:, 3]) + 0.2)

    counts: dict[tuple[int, int], int] = {}
    chunk_i = 0

    for lon, lat, dt_m in iterate_viirs_chunks(viirs_zip):
        chunk_i += 1
        keep = (dt_m >= start_dt) & (dt_m <= end_dt)
        keep &= (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)
        if not np.any(keep):
            continue

        lon2 = lon[keep]
        lat2 = lat[keep]
        dt2 = dt_m[keep]
        minute_int = dt2.astype("int64")

        pts = shapely_points(lon2, lat2)
        pairs = tree.query(pts, predicate="within")
        if pairs.shape[1] == 0:
            continue

        p_idx = pairs[0].astype(np.int32)
        f_local = pairs[1].astype(np.int32)

        # One assignment per VIIRS point: resolve overlaps by choosing smaller fire area.
        order = np.lexsort((areas[f_local], p_idx))
        p_sorted = p_idx[order]
        f_sorted = f_local[order]

        first = np.ones(len(order), dtype=bool)
        first[1:] = p_sorted[1:] != p_sorted[:-1]

        p_take = p_sorted[first]
        f_take = f_sorted[first]
        fire_take = fire_ids[f_take]
        minute_take = minute_int[p_take]

        key_arr = np.column_stack((fire_take.astype(np.int32), minute_take.astype(np.int64)))
        uniq, cnt = np.unique(key_arr, axis=0, return_counts=True)
        for (fi, mi), c in zip(uniq.tolist(), cnt.tolist(), strict=True):
            key = (int(fi), int(mi))
            counts[key] = counts.get(key, 0) + int(c)

        if chunk_i % 8 == 0:
            print(f"VIIRS chunks processed: {chunk_i}")

    rows = [
        {"fire_idx": fi, "minute_int": mi, "pixel_count": c}
        for (fi, mi), c in counts.items()
    ]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["fire_idx", "minute_int"]).reset_index(drop=True)
    return out


def build_overpass_events(minute_counts: pd.DataFrame, gap_minutes: int) -> pd.DataFrame:
    rows = []
    for fire_idx, g in minute_counts.groupby("fire_idx", sort=False):
        m = g["minute_int"].to_numpy(dtype=np.int64)
        c = g["pixel_count"].to_numpy(dtype=np.int32)
        order = np.argsort(m)
        m = m[order]
        c = c[order]

        split = np.where(np.diff(m) > gap_minutes)[0] + 1
        starts = np.r_[0, split]
        ends = np.r_[split, len(m)]

        for event_i, (s, e) in enumerate(zip(starts, ends, strict=True)):
            mm = m[s:e]
            cc = c[s:e]
            total = int(np.sum(cc))
            if total <= 0:
                continue
            weighted_min = int(np.round(np.average(mm, weights=cc)))
            rows.append(
                {
                    "fire_idx": int(fire_idx),
                    "event_idx": int(event_i),
                    "event_start_minute": int(mm.min()),
                    "event_end_minute": int(mm.max()),
                    "event_mid_minute": weighted_min,
                    "event_pixel_count": total,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["event_time_utc"] = pd.to_datetime(out["event_mid_minute"], unit="m", utc=True)
    out = out.sort_values(["fire_idx", "event_mid_minute"]).reset_index(drop=True)
    return out


def detect_drop_rebound_events(
    events: pd.DataFrame,
    min_prev_pixels: int,
    drop_frac_max: float,
    rebound_factor_min: float,
    rebound_prev_frac_min: float,
    max_gap_hours_triplet: float,
) -> pd.DataFrame:
    rows = []
    for fire_idx, g in events.groupby("fire_idx", sort=False):
        g = g.sort_values("event_mid_minute").reset_index(drop=True)
        if len(g) < 3:
            continue
        m = g["event_mid_minute"].to_numpy(dtype=np.int64)
        p = g["event_pixel_count"].to_numpy(dtype=np.int32)

        for i in range(1, len(g) - 1):
            prev_c = int(p[i - 1])
            cur_c = int(p[i])
            next_c = int(p[i + 1])

            if prev_c < min_prev_pixels:
                continue
            if cur_c > prev_c * drop_frac_max:
                continue
            if next_c < cur_c * rebound_factor_min:
                continue
            if next_c < prev_c * rebound_prev_frac_min:
                continue

            gap1_h = float((m[i] - m[i - 1]) / 60.0)
            gap2_h = float((m[i + 1] - m[i]) / 60.0)
            if gap1_h > max_gap_hours_triplet or gap2_h > max_gap_hours_triplet:
                continue

            drop_ratio = float(cur_c / max(prev_c, 1))
            rebound_ratio = float(next_c / max(cur_c, 1))
            score = float((prev_c - cur_c) + (next_c - cur_c))

            rows.append(
                {
                    "fire_idx": int(fire_idx),
                    "prev_event_idx": int(g.loc[i - 1, "event_idx"]),
                    "drop_event_idx": int(g.loc[i, "event_idx"]),
                    "next_event_idx": int(g.loc[i + 1, "event_idx"]),
                    "prev_time_utc": g.loc[i - 1, "event_time_utc"],
                    "drop_time_utc": g.loc[i, "event_time_utc"],
                    "next_time_utc": g.loc[i + 1, "event_time_utc"],
                    "prev_pixels": prev_c,
                    "drop_pixels": cur_c,
                    "next_pixels": next_c,
                    "gap_prev_to_drop_h": gap1_h,
                    "gap_drop_to_next_h": gap2_h,
                    "drop_ratio": drop_ratio,
                    "drop_pct": float(1.0 - drop_ratio),
                    "rebound_ratio": rebound_ratio,
                    "score": score,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out


@dataclass
class CloudGranule:
    key: str
    start_utc: datetime
    end_utc: datetime

    @property
    def mid_utc(self) -> datetime:
        return self.start_utc + (self.end_utc - self.start_utc) / 2


def parse_granule_time(code15: str) -> datetime:
    # Format: YYYYMMDDHHMMSSd (tenths of second)
    base = datetime.strptime(code15[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    tenth = int(code15[14])
    return base + timedelta(milliseconds=100 * tenth)


class ViirsCloudIndex:
    def __init__(self, base_url: str, prefix: str):
        self.base_url = base_url.rstrip("/")
        self.prefix = prefix.strip("/")
        self.day_cache: dict[date, list[CloudGranule]] = {}

    def list_day(self, d: date) -> list[CloudGranule]:
        if d in self.day_cache:
            return self.day_cache[d]

        pref = f"{self.prefix}/{d.year:04d}/{d.month:02d}/{d.day:02d}/"
        url = f"{self.base_url}/?prefix={pref}&max-keys=4000"
        out: list[CloudGranule] = []
        try:
            xml = urllib.request.urlopen(url, timeout=30).read()
            root = ET.fromstring(xml)
            for node in root.findall("s3:Contents", S3_NS):
                k_node = node.find("s3:Key", S3_NS)
                if k_node is None or not k_node.text:
                    continue
                key = k_node.text.strip()
                if not key.endswith(".nc"):
                    continue
                m = TIME_RE.search(key)
                if m is None:
                    continue
                st = parse_granule_time(m.group(1))
                en = parse_granule_time(m.group(2))
                out.append(CloudGranule(key=key, start_utc=st, end_utc=en))
        except Exception:
            out = []

        out.sort(key=lambda x: x.start_utc)
        self.day_cache[d] = out
        return out

    def nearest(self, t: datetime, max_diff_min: float) -> tuple[CloudGranule | None, float | None]:
        cands: list[CloudGranule] = []
        for shift in (-1, 0, 1):
            cands.extend(self.list_day((t + timedelta(days=shift)).date()))
        if not cands:
            return None, None

        # Prefer granules that cover the time; otherwise use nearest mid-point.
        covering = [g for g in cands if g.start_utc <= t <= g.end_utc]
        if covering:
            best = min(covering, key=lambda g: abs((g.mid_utc - t).total_seconds()))
            diff = abs((best.mid_utc - t).total_seconds()) / 60.0
            if diff <= max_diff_min:
                return best, diff
            return None, None

        best = min(cands, key=lambda g: abs((g.mid_utc - t).total_seconds()))
        diff = abs((best.mid_utc - t).total_seconds()) / 60.0
        if diff > max_diff_min:
            return None, None
        return best, diff


def ensure_cloud_file(base_url: str, key: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / key.split("/")[-1]
    if path.exists() and path.stat().st_size > 1000:
        return path
    url = f"{base_url.rstrip('/')}/{key}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


def cloud_fraction_for_geometry(
    lat: np.ndarray,
    lon: np.ndarray,
    cloud_binary: np.ndarray,
    geom: BaseGeometry,
) -> tuple[float, int]:
    minx, miny, maxx, maxy = geom.bounds
    in_box = (lat >= miny) & (lat <= maxy) & (lon >= minx) & (lon <= maxx)
    if not np.any(in_box):
        return np.nan, 0

    ys, xs = np.where(in_box)
    lons = lon[ys, xs]
    lats = lat[ys, xs]
    vals = cloud_binary[ys, xs]

    pts = shapely_points(lons, lats)
    inside = contains(geom, pts)
    if not np.any(inside):
        return np.nan, 0

    v = vals[inside]
    if v.size == 0:
        return np.nan, 0
    frac = float(np.mean(v == 1))
    return frac, int(v.size)


def compute_cloud_fractions(
    selected: pd.DataFrame,
    fires: list[WfigsFire],
    base_url: str,
    prefix: str,
    cache_dir: Path,
    max_diff_min: float,
) -> pd.DataFrame:
    idx = ViirsCloudIndex(base_url=base_url, prefix=prefix)
    out = selected.copy()

    numeric_cols = [
        "prev_cloud_pct",
        "drop_cloud_pct",
        "next_cloud_pct",
        "prev_cloud_pixels",
        "drop_cloud_pixels",
        "next_cloud_pixels",
        "prev_cloud_time_diff_min",
        "drop_cloud_time_diff_min",
        "next_cloud_time_diff_min",
    ]
    for col in numeric_cols:
        out[col] = np.nan

    key_cols = ["prev_cloud_key", "drop_cloud_key", "next_cloud_key"]
    for col in key_cols:
        out[col] = pd.Series([None] * len(out), dtype="object")

    # Resolve cloud granules for each slot.
    req: dict[str, set[int]] = {}
    slot_map: dict[tuple[int, str], tuple[str | None, float | None]] = {}
    for i, row in out.iterrows():
        for slot, tcol in [("prev", "prev_time_utc"), ("drop", "drop_time_utc"), ("next", "next_time_utc")]:
            t = pd.Timestamp(row[tcol]).to_pydatetime()
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            gran, diff = idx.nearest(t=t, max_diff_min=max_diff_min)
            key = gran.key if gran is not None else None
            slot_map[(int(i), slot)] = (key, diff)
            if key is not None:
                req.setdefault(key, set()).add(int(row["fire_idx"]))

    # Compute cloud fraction per (key, fire_idx).
    key_fire_cloud: dict[tuple[str, int], tuple[float, int]] = {}
    for k_i, (key, fire_set) in enumerate(req.items(), start=1):
        try:
            path = ensure_cloud_file(base_url=base_url, key=key, cache_dir=cache_dir)
            with Dataset(path) as ds:
                lat = np.asarray(ds.variables["Latitude"][:], dtype=np.float32)
                lon = np.asarray(ds.variables["Longitude"][:], dtype=np.float32)
                if "CloudMaskBinary" in ds.variables:
                    cb = np.asarray(ds.variables["CloudMaskBinary"][:], dtype=np.int8)
                    valid = np.isfinite(lat) & np.isfinite(lon) & ((cb == 0) | (cb == 1))
                    cb = np.where(valid, cb, -1)
                else:
                    cm = np.asarray(ds.variables["CloudMask"][:], dtype=np.int8)
                    valid = np.isfinite(lat) & np.isfinite(lon)
                    # 0 clear, 1 probably clear, 2 probably cloudy, 3 cloudy
                    cb = np.where(valid, (cm >= 2).astype(np.int8), -1)

                for fi in fire_set:
                    geom = fires[fi].geometry
                    frac, n_pix = cloud_fraction_for_geometry(lat, lon, cb, geom)
                    key_fire_cloud[(key, fi)] = (frac, n_pix)
        except Exception:
            for fi in fire_set:
                key_fire_cloud[(key, fi)] = (np.nan, 0)

        if k_i % 30 == 0:
            print(f"Cloud granules processed: {k_i}/{len(req)}")

    # Fill slot columns.
    for i, row in out.iterrows():
        fi = int(row["fire_idx"])
        for slot, prefix_col in [("prev", "prev"), ("drop", "drop"), ("next", "next")]:
            key, diff = slot_map[(int(i), slot)]
            out.at[i, f"{prefix_col}_cloud_key"] = key
            out.at[i, f"{prefix_col}_cloud_time_diff_min"] = diff
            if key is None:
                continue
            frac, n_pix = key_fire_cloud.get((key, fi), (np.nan, 0))
            out.at[i, f"{prefix_col}_cloud_pct"] = frac * 100.0 if np.isfinite(frac) else np.nan
            out.at[i, f"{prefix_col}_cloud_pixels"] = n_pix

    return out


def write_report(df: pd.DataFrame, all_candidates: pd.DataFrame, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    valid3 = df[["prev_cloud_pct", "drop_cloud_pct", "next_cloud_pct"]].notna().all(axis=1)
    d = df[valid3].copy()
    if not d.empty:
        d["cloud_spike_vs_prev"] = d["drop_cloud_pct"] - d["prev_cloud_pct"]
        d["cloud_spike_vs_next"] = d["drop_cloud_pct"] - d["next_cloud_pct"]
        d["cloud_spike_vs_mean_neighbors"] = d["drop_cloud_pct"] - (d["prev_cloud_pct"] + d["next_cloud_pct"]) / 2.0
        d["cloud_supported_drop"] = (
            (d["cloud_spike_vs_prev"] >= 15.0) & (d["cloud_spike_vs_next"] >= 15.0)
        )
        corr = float(d["drop_pct"].corr(d["cloud_spike_vs_mean_neighbors"], method="spearman"))
        support_rate = float(d["cloud_supported_drop"].mean())
    else:
        corr = np.nan
        support_rate = np.nan

    summary = {
        "selected_events": int(len(df)),
        "selected_fires": int(df["fire_id"].nunique()) if not df.empty else 0,
        "candidate_triplets_before_sampling": int(len(all_candidates)),
        "events_with_all_three_cloud_matches": int(valid3.sum()) if len(df) else 0,
        "all_three_cloud_match_rate": float(valid3.mean()) if len(df) else np.nan,
        "median_drop_pct": float(df["drop_pct"].median()) if len(df) else np.nan,
        "median_rebound_ratio": float(df["rebound_ratio"].median()) if len(df) else np.nan,
        "cloud_supported_drop_rate": support_rate,
        "spearman_drop_pct_vs_cloud_spike": corr,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    lines = []
    lines.append("# VIIRS Drop-Rebound vs VIIRS Cloud Mask")
    lines.append("")
    lines.append("- Method: find overpass triplets with `prev -> sharp drop -> rebound` in VIIRS fire pixels.")
    lines.append("- Cloud source: NOAA-20 `VIIRS-JRR-CloudMask` (`CloudMaskBinary`) at nearest overpass times.")
    lines.append("- Cloud metric: percent cloudy pixels inside the fire polygon (cloudy/probably-cloudy binary).")
    lines.append("")
    lines.append("## Summary")
    for k, v in summary.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    if not df.empty:
        top = df.sort_values("score", ascending=False).head(20).copy()
        cols = [
            "fire_id",
            "fire_name",
            "state",
            "drop_time_utc",
            "prev_pixels",
            "drop_pixels",
            "next_pixels",
            "drop_pct",
            "rebound_ratio",
            "prev_cloud_pct",
            "drop_cloud_pct",
            "next_cloud_pct",
        ]
        top = top[cols]
        lines.append("## Top Events")
        lines.append("```csv")
        lines.append(top.to_csv(index=False))
        lines.append("```")

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cloud_cache_dir.mkdir(parents=True, exist_ok=True)

    start_d = date.fromisoformat(args.start_date)
    end_d = date.fromisoformat(args.end_date)
    start_dt = np.datetime64(datetime.combine(start_d, datetime.min.time()), "m")
    end_dt = np.datetime64(datetime.combine(end_d, datetime.max.time()), "m")

    print("Loading WFIGS fires...")
    fires = load_wfigs_fires(
        path=args.wfigs_geojson,
        start_year=start_d.year,
        end_year=end_d.year,
        min_acres=args.min_acres,
    )
    print(f"WFIGS fires loaded: {len(fires):,}")
    if not fires:
        raise RuntimeError("No fires matched WFIGS filters.")

    print("Assigning VIIRS detections to fires and building per-minute counts...")
    minute_counts = build_fire_minute_counts(
        fires=fires,
        viirs_zip=args.viirs_zip,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    if minute_counts.empty:
        raise RuntimeError("No VIIRS detections matched the filtered fires/date range.")
    minute_counts.to_csv(args.output_dir / "fire_minute_counts.csv", index=False)
    print(f"Fire-minute rows: {len(minute_counts):,}")

    print("Building overpass events...")
    events = build_overpass_events(minute_counts=minute_counts, gap_minutes=args.event_gap_minutes)
    if events.empty:
        raise RuntimeError("No overpass events were built.")
    events.to_csv(args.output_dir / "fire_overpass_events.csv", index=False)
    print(f"Overpass events: {len(events):,}")

    print("Detecting drop-rebound triplets...")
    candidates = detect_drop_rebound_events(
        events=events,
        min_prev_pixels=args.min_prev_pixels,
        drop_frac_max=args.drop_frac_max,
        rebound_factor_min=args.rebound_factor_min,
        rebound_prev_frac_min=args.rebound_prev_frac_min,
        max_gap_hours_triplet=args.max_gap_hours_triplet,
    )
    if candidates.empty:
        raise RuntimeError("No drop-rebound triplets found with current thresholds.")
    candidates.to_csv(args.output_dir / "drop_rebound_candidates.csv", index=False)
    print(f"Drop-rebound candidates: {len(candidates):,}")

    # Select top N fires by best score, then keep K events per fire.
    cand = candidates.sort_values("score", ascending=False).copy()
    fire_rank = cand.groupby("fire_idx", as_index=False)["score"].max().sort_values("score", ascending=False)
    keep_fires = fire_rank["fire_idx"].tolist()
    if int(args.max_fires) > 0:
        keep_fires = keep_fires[: int(args.max_fires)]
    cand = cand[cand["fire_idx"].isin(keep_fires)]
    selected = (
        cand.sort_values("score", ascending=False)
        .groupby("fire_idx", as_index=False)
        .head(max(1, int(args.events_per_fire)))
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    if int(args.max_events) > 0:
        selected = selected.head(int(args.max_events)).reset_index(drop=True)
    print(
        f"Selected events for cloud check: {len(selected):,} "
        f"across {selected['fire_idx'].nunique():,} fires "
        f"(events_per_fire={int(args.events_per_fire)}, max_events={int(args.max_events)})"
    )

    # Attach fire metadata.
    selected["fire_id"] = [fires[int(fi)].fire_id for fi in selected["fire_idx"]]
    selected["fire_name"] = [fires[int(fi)].name for fi in selected["fire_idx"]]
    selected["state"] = [fires[int(fi)].state for fi in selected["fire_idx"]]
    selected["size_acres"] = [fires[int(fi)].size_acres for fi in selected["fire_idx"]]
    selected["fire_start_date"] = [fires[int(fi)].start_date for fi in selected["fire_idx"]]
    selected["fire_end_date"] = [fires[int(fi)].end_date for fi in selected["fire_idx"]]

    print("Matching VIIRS cloud-mask granules and computing cloud fractions...")
    selected_cloud = compute_cloud_fractions(
        selected=selected,
        fires=fires,
        base_url=args.cloud_base_url,
        prefix=args.cloud_prefix,
        cache_dir=args.cloud_cache_dir,
        max_diff_min=args.max_cloud_time_diff_min,
    )

    selected_cloud.to_csv(args.output_dir / "selected_drop_rebound_with_clouds.csv", index=False)
    summary = write_report(selected_cloud, candidates, args.output_dir)

    print("Done.")
    print(f"Output dir: {args.output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
