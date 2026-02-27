#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import ijson
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile  # pyshp
from rapidfuzz import fuzz
from shapely import points as shapely_points
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree


DAY0 = np.datetime64("1970-01-01", "D")


@dataclass
class FireIncident:
    dataset: str
    fire_id: str
    irwin_id: str | None
    name: str
    state: str | None
    start_date: date
    end_date: date | None
    end_date_is_proxy: bool
    size_acres: float
    geometry: BaseGeometry


@dataclass
class FireDetectionMetrics:
    total_window_detections: int = 0
    in_window_detections: int = 0
    pre_start_detections: int = 0
    post_end_detections: int = 0
    high_nom_total_window: int = 0
    high_nom_in_window: int = 0
    low_total_window: int = 0
    first_day_int: int | None = None
    last_day_int: int | None = None
    first_in_day_int: int | None = None
    last_in_day_int: int | None = None
    daily_total_in_window: Counter[int] = field(default_factory=Counter)
    daily_hn_in_window: Counter[int] = field(default_factory=Counter)


def day_int_from_date(value: date) -> int:
    return int((np.datetime64(value.isoformat(), "D") - DAY0).astype(int))


def date_from_day_int(day_int: int | None) -> date | None:
    if day_int is None:
        return None
    return date.fromisoformat(str(DAY0 + np.timedelta64(int(day_int), "D")))


def normalize_irwin(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().strip("{}").lower()
    if not text:
        return None
    return text


def normalize_name(value: str | None) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def to_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_wfigs_dt(value) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        dt = parsedate_to_datetime(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    try:
        dt2 = pd.to_datetime(text, utc=True, errors="coerce")
        if pd.isna(dt2):
            return None
        return dt2.to_pydatetime().astimezone(timezone.utc)
    except Exception:
        return None


def choose_wfigs_end(props: dict) -> datetime | None:
    keys = [
        "attr_FireOutDateTime",
        "attr_ContainmentDateTime",
        "attr_ControlDateTime",
        "poly_PolygonDateTime",
    ]
    for key in keys:
        dt = parse_wfigs_dt(props.get(key))
        if dt is not None:
            return dt
    return None


def parse_mtbs_post_date(post_id: str | None) -> date | None:
    if post_id is None:
        return None
    text = str(post_id).strip()
    if text == "":
        return None
    m = re.search(r"(\d{8})$", text)
    if m is None:
        return None
    s = m.group(1)
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        return None


def repair_geometry(geom: BaseGeometry) -> BaseGeometry | None:
    if geom.is_empty:
        return None
    if geom.is_valid:
        return geom
    fixed = geom.buffer(0)
    if fixed.is_empty:
        return None
    return fixed


def load_wfigs_fires(path: Path, start_year: int, end_year: int, min_acres: float) -> list[FireIncident]:
    start_limit = date(start_year, 1, 1)
    end_limit = date(end_year, 12, 31)
    by_fire_id: dict[str, FireIncident] = {}

    with path.open("rb") as fp:
        for feature in ijson.items(fp, "features.item"):
            props = feature.get("properties", {})
            if str(props.get("attr_IncidentTypeCategory") or "").strip() != "WF":
                continue

            start_dt = parse_wfigs_dt(props.get("attr_FireDiscoveryDateTime"))
            if start_dt is None:
                continue
            start_d = start_dt.date()
            if start_d < start_limit or start_d > end_limit:
                continue

            size = to_float(props.get("attr_IncidentSize"))
            if size is None or size <= 0:
                size = to_float(props.get("poly_GISAcres"))
            if size is None or size < min_acres:
                continue

            fire_id = (
                str(props.get("attr_UniqueFireIdentifier") or "").strip()
                or str(props.get("poly_IRWINID") or "").strip()
                or str(props.get("attr_LocalIncidentIdentifier") or "").strip()
            )
            if fire_id == "":
                continue

            geom_obj = feature.get("geometry")
            if geom_obj is None:
                continue
            geom = repair_geometry(shape(geom_obj))
            if geom is None:
                continue

            end_dt = choose_wfigs_end(props)
            end_d = end_dt.date() if end_dt is not None else start_d
            if end_d < start_d:
                end_d = start_d

            rec = FireIncident(
                dataset="WFIGS",
                fire_id=fire_id,
                irwin_id=normalize_irwin(props.get("attr_IrwinID") or props.get("poly_IRWINID")),
                name=str(props.get("attr_IncidentName") or props.get("poly_IncidentName") or "Unknown").strip(),
                state=str(props.get("attr_POOState") or "").strip() or None,
                start_date=start_d,
                end_date=end_d,
                end_date_is_proxy=False,
                size_acres=float(size),
                geometry=geom,
            )

            prev = by_fire_id.get(fire_id)
            if prev is None:
                by_fire_id[fire_id] = rec
            else:
                if rec.size_acres > prev.size_acres:
                    geom_keep = rec.geometry
                    size_keep = rec.size_acres
                else:
                    geom_keep = prev.geometry
                    size_keep = prev.size_acres
                by_fire_id[fire_id] = FireIncident(
                    dataset="WFIGS",
                    fire_id=fire_id,
                    irwin_id=rec.irwin_id or prev.irwin_id,
                    name=(rec.name if len(rec.name) >= len(prev.name) else prev.name),
                    state=rec.state or prev.state,
                    start_date=min(rec.start_date, prev.start_date),
                    end_date=max(rec.end_date or rec.start_date, prev.end_date or prev.start_date),
                    end_date_is_proxy=False,
                    size_acres=size_keep,
                    geometry=geom_keep,
                )

    fires = list(by_fire_id.values())
    fires.sort(key=lambda r: (r.start_date, -r.size_acres, r.fire_id))
    return fires


def load_mtbs_fires(zip_path: Path, start_year: int, end_year: int, min_acres: float) -> list[FireIncident]:
    start_limit = date(start_year, 1, 1)
    end_limit = date(end_year, 12, 31)
    reader = shapefile.Reader(str(zip_path))
    fields = [f[0] for f in reader.fields[1:]]
    idx = {name: i for i, name in enumerate(fields)}
    by_fire_id: dict[str, FireIncident] = {}

    for sr in reader.iterShapeRecords():
        rec = sr.record
        if str(rec[idx["Incid_Type"]] or "").strip().lower() != "wildfire":
            continue
        ig_d = rec[idx["Ig_Date"]]
        if ig_d is None or ig_d < start_limit or ig_d > end_limit:
            continue

        size = to_float(rec[idx["BurnBndAc"]])
        if size is None or size < min_acres:
            continue

        fire_id = str(rec[idx["Event_ID"]] or "").strip()
        if fire_id == "":
            continue

        geom = repair_geometry(shape(sr.shape.__geo_interface__))
        if geom is None:
            continue

        post_date = parse_mtbs_post_date(rec[idx["Post_ID"]])
        if post_date is not None and post_date < ig_d:
            post_date = None

        state = fire_id[:2] if len(fire_id) >= 2 else None
        rec_out = FireIncident(
            dataset="MTBS",
            fire_id=fire_id,
            irwin_id=normalize_irwin(rec[idx["irwinID"]]),
            name=str(rec[idx["Incid_Name"]] or "Unknown").strip(),
            state=state,
            start_date=ig_d,
            end_date=post_date,
            end_date_is_proxy=True,
            size_acres=float(size),
            geometry=geom,
        )

        prev = by_fire_id.get(fire_id)
        if prev is None:
            by_fire_id[fire_id] = rec_out
        else:
            if rec_out.size_acres > prev.size_acres:
                geom_keep = rec_out.geometry
                size_keep = rec_out.size_acres
            else:
                geom_keep = prev.geometry
                size_keep = prev.size_acres
            end_keep = rec_out.end_date or prev.end_date
            if rec_out.end_date and prev.end_date:
                end_keep = max(rec_out.end_date, prev.end_date)
            by_fire_id[fire_id] = FireIncident(
                dataset="MTBS",
                fire_id=fire_id,
                irwin_id=rec_out.irwin_id or prev.irwin_id,
                name=(rec_out.name if len(rec_out.name) >= len(prev.name) else prev.name),
                state=rec_out.state or prev.state,
                start_date=min(rec_out.start_date, prev.start_date),
                end_date=end_keep,
                end_date_is_proxy=True,
                size_acres=size_keep,
                geometry=geom_keep,
            )

    fires = list(by_fire_id.values())
    fires.sort(key=lambda r: (r.start_date, -r.size_acres, r.fire_id))
    return fires


def classify_confidence(series: pd.Series) -> np.ndarray:
    text = series.astype(str).str.strip().str.lower()
    out = np.full(len(text), "u", dtype="<U1")
    first = text.str.slice(0, 1)
    out[first == "h"] = "h"
    out[first == "n"] = "n"
    out[first == "l"] = "l"
    numeric = pd.to_numeric(text, errors="coerce")
    unknown = out == "u"
    if unknown.any():
        out[unknown & numeric.notna() & (numeric >= 80)] = "h"
        out[unknown & numeric.notna() & (numeric >= 30) & (numeric < 80)] = "n"
        out[unknown & numeric.notna() & (numeric < 30)] = "l"
    return out


def iterate_viirs_chunks(
    viirs_zip: Path,
    min_day_int: int,
    max_day_int: int,
    chunksize: int = 250_000,
):
    usecols = ["latitude", "longitude", "acq_date", "confidence"]
    with zipfile.ZipFile(viirs_zip) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv") and "archive" in n.lower()]
        if not members:
            raise FileNotFoundError(f"No archive CSV found in {viirs_zip}")
        for member in members:
            with zf.open(member) as fp:
                for chunk in pd.read_csv(fp, usecols=usecols, chunksize=chunksize, low_memory=False):
                    day_vals = pd.to_datetime(chunk["acq_date"], errors="coerce").to_numpy(dtype="datetime64[D]")
                    lat_vals = pd.to_numeric(chunk["latitude"], errors="coerce").to_numpy(dtype=np.float64)
                    lon_vals = pd.to_numeric(chunk["longitude"], errors="coerce").to_numpy(dtype=np.float64)
                    conf_vals = classify_confidence(chunk["confidence"])

                    valid = ~np.isnat(day_vals)
                    valid &= np.isfinite(lat_vals) & np.isfinite(lon_vals)
                    valid &= (lat_vals >= -90) & (lat_vals <= 90) & (lon_vals >= -180) & (lon_vals <= 180)
                    if not np.any(valid):
                        continue

                    day_int = (day_vals[valid] - DAY0).astype(np.int32)
                    keep = (day_int >= min_day_int) & (day_int <= max_day_int)
                    if not np.any(keep):
                        continue

                    yield (
                        lon_vals[valid][keep],
                        lat_vals[valid][keep],
                        day_int[keep],
                        conf_vals[valid][keep],
                    )


def compute_detection_metrics(
    fires: list[FireIncident],
    viirs_zip: Path,
    window_pad_days: int,
    start_year: int,
    end_year: int,
    label: str,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    n = len(fires)
    geoms = [f.geometry for f in fires]
    tree = STRtree(geoms)

    start_day = np.array([day_int_from_date(f.start_date) for f in fires], dtype=np.int32)
    end_day = np.array([day_int_from_date(f.end_date or f.start_date) for f in fires], dtype=np.int32)
    active_days = np.maximum(end_day - start_day + 1, 1).astype(np.int32)
    duration_days = active_days.copy()
    window_start = start_day - int(window_pad_days)
    window_end = end_day + int(window_pad_days)

    viirs_min = day_int_from_date(date(start_year, 1, 1) - timedelta(days=window_pad_days + 2))
    viirs_max = day_int_from_date(date(end_year, 12, 31) + timedelta(days=450))

    metrics = [FireDetectionMetrics() for _ in fires]
    chunk_i = 0
    matched_points_total = 0

    for lon_vals, lat_vals, day_vals, conf_vals in iterate_viirs_chunks(
        viirs_zip=viirs_zip,
        min_day_int=viirs_min,
        max_day_int=viirs_max,
    ):
        chunk_i += 1
        pts = shapely_points(lon_vals, lat_vals)
        pairs = tree.query(pts, predicate="within")
        if pairs.shape[1] == 0:
            if chunk_i % 10 == 0:
                print(f"[{label}] processed chunk {chunk_i}: no spatial matches")
            continue

        p_idx = pairs[0].astype(np.int32)
        f_idx = pairs[1].astype(np.int32)

        d = day_vals[p_idx]
        keep = (d >= window_start[f_idx]) & (d <= window_end[f_idx])
        if not np.any(keep):
            if chunk_i % 10 == 0:
                print(f"[{label}] processed chunk {chunk_i}: no date-window matches")
            continue

        p_idx = p_idx[keep]
        f_idx = f_idx[keep]
        d = d[keep]

        s = start_day[f_idx]
        e = end_day[f_idx]
        pre = d < s
        post = d > e
        off = np.where(pre, s - d, np.where(post, d - e, 0))
        score = off.astype(np.int64) * 10_000 + np.minimum(duration_days[f_idx], 9_999).astype(np.int64)

        order = np.lexsort((f_idx, score, p_idx))
        p_sorted = p_idx[order]
        f_sorted = f_idx[order]
        d_sorted = d[order]

        first_mask = np.r_[True, p_sorted[1:] != p_sorted[:-1]]
        p_sel = p_sorted[first_mask]
        f_sel = f_sorted[first_mask]
        d_sel = d_sorted[first_mask]
        conf_sel = conf_vals[p_sel]

        matched_points_total += len(p_sel)

        for fire_i, day_i, conf_i in zip(f_sel, d_sel, conf_sel, strict=True):
            fire_idx = int(fire_i)
            day_int = int(day_i)
            conf = str(conf_i)
            m = metrics[fire_idx]

            m.total_window_detections += 1
            if conf in ("h", "n"):
                m.high_nom_total_window += 1
            elif conf == "l":
                m.low_total_window += 1

            if m.first_day_int is None or day_int < m.first_day_int:
                m.first_day_int = day_int
            if m.last_day_int is None or day_int > m.last_day_int:
                m.last_day_int = day_int

            s_i = int(start_day[fire_idx])
            e_i = int(end_day[fire_idx])
            if day_int < s_i:
                m.pre_start_detections += 1
            elif day_int > e_i:
                m.post_end_detections += 1
            else:
                m.in_window_detections += 1
                m.daily_total_in_window[day_int] += 1
                if conf in ("h", "n"):
                    m.high_nom_in_window += 1
                    m.daily_hn_in_window[day_int] += 1
                if m.first_in_day_int is None or day_int < m.first_in_day_int:
                    m.first_in_day_int = day_int
                if m.last_in_day_int is None or day_int > m.last_in_day_int:
                    m.last_in_day_int = day_int

        if chunk_i % 10 == 0:
            print(f"[{label}] processed chunk {chunk_i}: cumulative matched points={matched_points_total:,}")

    rows = []
    profile_sum = np.zeros(10, dtype=np.float64)
    profile_n = 0

    for i, fire in enumerate(fires):
        m = metrics[i]
        start_i = int(start_day[i])
        end_i = int(end_day[i])
        active_i = int(active_days[i])

        days_any_in = len(m.daily_total_in_window)
        days_hn_nonzero = len(m.daily_hn_in_window)
        days_hn_gt3 = sum(1 for c in m.daily_hn_in_window.values() if c > 3)
        days_hn_1to3 = sum(1 for c in m.daily_hn_in_window.values() if 1 <= c <= 3)

        few_days_le3 = active_i - days_hn_gt3
        pct_days_few_hn_le3 = (few_days_le3 / active_i) if active_i > 0 else np.nan
        pct_days_hn_1to3 = (days_hn_1to3 / active_i) if active_i > 0 else np.nan
        pct_days_with_any_hn = (days_hn_nonzero / active_i) if active_i > 0 else np.nan
        pct_days_with_any_detection = (days_any_in / active_i) if active_i > 0 else np.nan

        if m.daily_total_in_window:
            min_in = min(m.daily_total_in_window)
            max_in = max(m.daily_total_in_window)
            span_days = (max_in - min_in) + 1
            peak_day = min(m.daily_total_in_window, key=lambda d: (-m.daily_total_in_window[d], d))
            denom = max(active_i - 1, 1)
            peak_rel = (peak_day - start_i) / denom
            span_rel = span_days / active_i
        else:
            span_days = 0
            peak_rel = np.nan
            span_rel = np.nan

        if m.daily_hn_in_window:
            total_hn = sum(m.daily_hn_in_window.values())
            denom = max(active_i - 1, 1)
            for day_i2, count in m.daily_hn_in_window.items():
                rel = (day_i2 - start_i) / denom
                b = int(np.clip(np.floor(rel * 10.0), 0, 9))
                profile_sum[b] += count / total_hn
            profile_n += 1

        in_share = (
            m.in_window_detections / m.total_window_detections
            if m.total_window_detections > 0
            else np.nan
        )
        noise_share = (
            (m.pre_start_detections + m.post_end_detections) / m.total_window_detections
            if m.total_window_detections > 0
            else np.nan
        )

        first_lead_days = (
            (start_i - m.first_day_int)
            if (m.first_day_int is not None and m.first_day_int < start_i)
            else 0
        )
        last_lag_days = (
            (m.last_day_int - end_i)
            if (m.last_day_int is not None and m.last_day_int > end_i)
            else 0
        )

        rows.append(
            {
                "dataset": fire.dataset,
                "fire_index": i,
                "fire_id": fire.fire_id,
                "irwin_id": fire.irwin_id,
                "name": fire.name,
                "state": fire.state,
                "start_date": fire.start_date,
                "end_date": fire.end_date or fire.start_date,
                "end_date_is_proxy": fire.end_date_is_proxy,
                "size_acres": fire.size_acres,
                "active_window_days": active_i,
                "total_window_detections": m.total_window_detections,
                "in_window_detections": m.in_window_detections,
                "pre_start_detections": m.pre_start_detections,
                "post_end_detections": m.post_end_detections,
                "high_nom_total_window": m.high_nom_total_window,
                "high_nom_in_window": m.high_nom_in_window,
                "low_total_window": m.low_total_window,
                "in_window_share": in_share,
                "noise_share": noise_share,
                "days_with_any_detection_in_window": days_any_in,
                "days_with_any_hn_in_window": days_hn_nonzero,
                "pct_active_days_with_any_detection": pct_days_with_any_detection,
                "pct_active_days_with_any_hn": pct_days_with_any_hn,
                "pct_active_days_few_hn_le3": pct_days_few_hn_le3,
                "pct_active_days_hn_1to3": pct_days_hn_1to3,
                "first_detection_date": date_from_day_int(m.first_day_int),
                "last_detection_date": date_from_day_int(m.last_day_int),
                "first_in_window_detection_date": date_from_day_int(m.first_in_day_int),
                "last_in_window_detection_date": date_from_day_int(m.last_in_day_int),
                "days_before_start_first_detection": first_lead_days,
                "days_after_end_last_detection": last_lag_days,
                "in_window_detection_span_days": span_days,
                "in_window_detection_span_frac": span_rel,
                "in_window_peak_relative_position": peak_rel,
            }
        )

    out_df = pd.DataFrame(rows)
    profile_avg = (profile_sum / profile_n) if profile_n > 0 else np.zeros(10, dtype=np.float64)
    profile_df = pd.DataFrame(
        {
            "dataset": label,
            "bin_index": np.arange(10, dtype=int),
            "bin_start_rel": np.arange(10) / 10.0,
            "bin_end_rel": (np.arange(10) + 1) / 10.0,
            "avg_profile_share": profile_avg,
        }
    )

    summary = {
        "dataset": label,
        "fire_count": int(len(out_df)),
        "mean_size_acres": float(out_df["size_acres"].mean()),
        "median_size_acres": float(out_df["size_acres"].median()),
        "mean_active_window_days": float(out_df["active_window_days"].mean()),
        "median_active_window_days": float(out_df["active_window_days"].median()),
        "fires_with_any_window_detection": int((out_df["total_window_detections"] > 0).sum()),
        "fires_with_any_in_window_detection": int((out_df["in_window_detections"] > 0).sum()),
        "total_window_detections": int(out_df["total_window_detections"].sum()),
        "total_in_window_detections": int(out_df["in_window_detections"].sum()),
        "total_pre_start_detections": int(out_df["pre_start_detections"].sum()),
        "total_post_end_detections": int(out_df["post_end_detections"].sum()),
        "in_window_detection_share_total": float(
            out_df["in_window_detections"].sum()
            / max(out_df["total_window_detections"].sum(), 1)
        ),
        "noise_share_total": float(
            (out_df["pre_start_detections"].sum() + out_df["post_end_detections"].sum())
            / max(out_df["total_window_detections"].sum(), 1)
        ),
        "mean_in_window_share_per_fire": float(out_df["in_window_share"].mean(skipna=True)),
        "mean_noise_share_per_fire": float(out_df["noise_share"].mean(skipna=True)),
        "pct_active_days_with_any_hn_weighted": float(
            out_df["days_with_any_hn_in_window"].sum() / max(out_df["active_window_days"].sum(), 1)
        ),
        "pct_active_days_few_hn_le3_weighted": float(
            (
                out_df["active_window_days"].sum()
                - (out_df["pct_active_days_few_hn_le3"].rsub(1.0) * out_df["active_window_days"]).sum()
            )
            / max(out_df["active_window_days"].sum(), 1)
        ),
        "median_days_before_start_first_detection": float(
            out_df["days_before_start_first_detection"].median()
        ),
        "median_days_after_end_last_detection": float(
            out_df["days_after_end_last_detection"].median()
        ),
    }
    return out_df, summary, profile_df


def safe_overlap_ratio(a: BaseGeometry, b: BaseGeometry) -> float:
    try:
        inter = a.intersection(b).area
        if inter <= 0:
            return 0.0
        denom = min(max(a.area, 1e-12), max(b.area, 1e-12))
        return float(inter / denom)
    except Exception:
        return 0.0


def match_mtbs_wfigs(
    mtbs_fires: list[FireIncident],
    wfigs_fires: list[FireIncident],
    max_start_diff_days: int = 90,
) -> pd.DataFrame:
    mtbs_used: set[int] = set()
    wfigs_used: set[int] = set()
    rows: list[dict] = []

    wfigs_irwin: defaultdict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(wfigs_fires):
        if f.irwin_id:
            wfigs_irwin[f.irwin_id].append(i)

    for i, m in enumerate(mtbs_fires):
        if not m.irwin_id:
            continue
        candidates = [j for j in wfigs_irwin.get(m.irwin_id, []) if j not in wfigs_used]
        if not candidates:
            continue
        candidates.sort(key=lambda j: abs((wfigs_fires[j].start_date - m.start_date).days))
        j = candidates[0]
        w = wfigs_fires[j]
        overlap = safe_overlap_ratio(m.geometry, w.geometry)
        rows.append(
            {
                "mtbs_index": i,
                "wfigs_index": j,
                "match_method": "irwin_exact",
                "match_score": 1.0,
                "overlap_ratio": overlap,
            }
        )
        mtbs_used.add(i)
        wfigs_used.add(j)

    unmatched_mtbs = [i for i in range(len(mtbs_fires)) if i not in mtbs_used]
    unmatched_wfigs = [j for j in range(len(wfigs_fires)) if j not in wfigs_used]

    if unmatched_mtbs and unmatched_wfigs:
        geom_list = [wfigs_fires[j].geometry for j in unmatched_wfigs]
        tree = STRtree(geom_list)
        candidate_rows: list[tuple[float, float, int, int, int]] = []

        for i in unmatched_mtbs:
            m = mtbs_fires[i]
            cand_pos = tree.query(m.geometry, predicate="intersects")
            if len(cand_pos) == 0:
                continue
            m_name = normalize_name(m.name)
            for pos in cand_pos:
                j = unmatched_wfigs[int(pos)]
                if j in wfigs_used:
                    continue
                w = wfigs_fires[j]
                start_diff = abs((w.start_date - m.start_date).days)
                if start_diff > max_start_diff_days:
                    continue
                overlap = safe_overlap_ratio(m.geometry, w.geometry)
                if overlap < 0.1:
                    continue
                w_name = normalize_name(w.name)
                name_score = fuzz.ratio(m_name, w_name) / 100.0 if (m_name and w_name) else 0.0
                time_score = max(0.0, 1.0 - (start_diff / max_start_diff_days))
                score = (0.65 * overlap) + (0.2 * time_score) + (0.15 * name_score)
                if score < 0.35:
                    continue
                candidate_rows.append((score, overlap, start_diff, i, j))

        candidate_rows.sort(reverse=True, key=lambda x: (x[0], x[1], -x[2]))
        for score, overlap, _start_diff, i, j in candidate_rows:
            if i in mtbs_used or j in wfigs_used:
                continue
            mtbs_used.add(i)
            wfigs_used.add(j)
            rows.append(
                {
                    "mtbs_index": i,
                    "wfigs_index": j,
                    "match_method": "spatial_temporal",
                    "match_score": score,
                    "overlap_ratio": overlap,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "mtbs_index",
                "wfigs_index",
                "match_method",
                "match_score",
                "overlap_ratio",
            ]
        )
    return pd.DataFrame(rows)


def add_pairwise_differences(
    match_df: pd.DataFrame,
    mtbs_fires: list[FireIncident],
    wfigs_fires: list[FireIncident],
) -> pd.DataFrame:
    if match_df.empty:
        return match_df.copy()

    rows = []
    for row in match_df.itertuples(index=False):
        m = mtbs_fires[int(row.mtbs_index)]
        w = wfigs_fires[int(row.wfigs_index)]
        m_end = m.end_date or m.start_date
        w_end = w.end_date or w.start_date
        rows.append(
            {
                "mtbs_index": int(row.mtbs_index),
                "wfigs_index": int(row.wfigs_index),
                "match_method": row.match_method,
                "match_score": float(row.match_score),
                "overlap_ratio": float(row.overlap_ratio),
                "mtbs_fire_id": m.fire_id,
                "wfigs_fire_id": w.fire_id,
                "mtbs_name": m.name,
                "wfigs_name": w.name,
                "mtbs_state": m.state,
                "wfigs_state": w.state,
                "mtbs_start_date": m.start_date,
                "wfigs_start_date": w.start_date,
                "start_date_diff_days_wfigs_minus_mtbs": (w.start_date - m.start_date).days,
                "mtbs_end_proxy_date": m.end_date,
                "wfigs_end_date": w.end_date,
                "end_date_diff_days_wfigs_minus_mtbs_proxy": (w_end - m_end).days,
                "mtbs_size_acres": m.size_acres,
                "wfigs_size_acres": w.size_acres,
                "size_diff_acres_wfigs_minus_mtbs": w.size_acres - m.size_acres,
                "size_ratio_wfigs_over_mtbs": (w.size_acres / m.size_acres) if m.size_acres > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_matches(pair_df: pd.DataFrame, mtbs_count: int, wfigs_count: int) -> dict:
    if pair_df.empty:
        return {
            "matched_pair_count": 0,
            "pct_mtbs_matched": 0.0,
            "pct_wfigs_matched": 0.0,
        }
    return {
        "matched_pair_count": int(len(pair_df)),
        "pct_mtbs_matched": float(len(pair_df) / max(mtbs_count, 1)),
        "pct_wfigs_matched": float(len(pair_df) / max(wfigs_count, 1)),
        "irwin_exact_pairs": int((pair_df["match_method"] == "irwin_exact").sum()),
        "spatial_temporal_pairs": int((pair_df["match_method"] == "spatial_temporal").sum()),
        "median_overlap_ratio": float(pair_df["overlap_ratio"].median()),
        "median_abs_start_date_diff_days": float(
            pair_df["start_date_diff_days_wfigs_minus_mtbs"].abs().median()
        ),
        "median_abs_size_pct_diff": float(
            ((pair_df["wfigs_size_acres"] - pair_df["mtbs_size_acres"]).abs() / pair_df["mtbs_size_acres"]).median()
        ),
    }


def create_figures(
    mtbs_df: pd.DataFrame,
    wfigs_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Fire size distribution.
    plt.figure(figsize=(8, 5))
    mtbs_log = np.log10(mtbs_df["size_acres"].clip(lower=1))
    wfigs_log = np.log10(wfigs_df["size_acres"].clip(lower=1))
    bins = np.linspace(min(mtbs_log.min(), wfigs_log.min()), max(mtbs_log.max(), wfigs_log.max()), 40)
    plt.hist(mtbs_log, bins=bins, alpha=0.55, label="MTBS")
    plt.hist(wfigs_log, bins=bins, alpha=0.55, label="WFIGS")
    plt.xlabel("log10(Fire size acres)")
    plt.ylabel("Count")
    plt.title("Fire Size Distribution (2021-2023, >=1000 acres)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "fire_size_distribution_log10.png", dpi=140)
    plt.close()

    # In-window detection share.
    plt.figure(figsize=(8, 5))
    data = [
        mtbs_df["in_window_share"].dropna().to_numpy(),
        wfigs_df["in_window_share"].dropna().to_numpy(),
    ]
    plt.boxplot(data, labels=["MTBS", "WFIGS"], showfliers=False)
    plt.ylabel("VIIRS detection share in reported window")
    plt.title("Temporal Coherence via In-Window Detection Share")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(figures_dir / "in_window_detection_share_boxplot.png", dpi=140)
    plt.close()

    # Relative profile.
    plt.figure(figsize=(8, 5))
    for dataset, grp in profile_df.groupby("dataset"):
        x = grp["bin_start_rel"].to_numpy() + 0.05
        y = grp["avg_profile_share"].to_numpy()
        plt.plot(x, y, marker="o", label=dataset)
    plt.xlabel("Relative position inside reported fire window")
    plt.ylabel("Avg normalized high/nominal VIIRS share")
    plt.title("Where Detections Occur Inside the Reported Window")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "relative_daily_profile_hn.png", dpi=140)
    plt.close()


def write_markdown_report(
    output_path: Path,
    mtbs_summary: dict,
    wfigs_summary: dict,
    match_summary: dict,
    pair_df: pd.DataFrame,
) -> None:
    comparison_df = pd.DataFrame(
        [
            {
                "dataset": "MTBS",
                "fire_count": mtbs_summary["fire_count"],
                "mean_size_acres": mtbs_summary["mean_size_acres"],
                "median_size_acres": mtbs_summary["median_size_acres"],
                "mean_window_days": mtbs_summary["mean_active_window_days"],
                "in_window_share_total": mtbs_summary["in_window_detection_share_total"],
                "noise_share_total": mtbs_summary["noise_share_total"],
                "pct_days_with_any_hn": mtbs_summary["pct_active_days_with_any_hn_weighted"],
                "pct_days_few_hn_le3": mtbs_summary["pct_active_days_few_hn_le3_weighted"],
                "total_in_window_detections": mtbs_summary["total_in_window_detections"],
                "pre_start_detections": mtbs_summary["total_pre_start_detections"],
                "post_end_detections": mtbs_summary["total_post_end_detections"],
            },
            {
                "dataset": "WFIGS",
                "fire_count": wfigs_summary["fire_count"],
                "mean_size_acres": wfigs_summary["mean_size_acres"],
                "median_size_acres": wfigs_summary["median_size_acres"],
                "mean_window_days": wfigs_summary["mean_active_window_days"],
                "in_window_share_total": wfigs_summary["in_window_detection_share_total"],
                "noise_share_total": wfigs_summary["noise_share_total"],
                "pct_days_with_any_hn": wfigs_summary["pct_active_days_with_any_hn_weighted"],
                "pct_days_few_hn_le3": wfigs_summary["pct_active_days_few_hn_le3_weighted"],
                "total_in_window_detections": wfigs_summary["total_in_window_detections"],
                "pre_start_detections": wfigs_summary["total_pre_start_detections"],
                "post_end_detections": wfigs_summary["total_post_end_detections"],
            },
        ]
    )

    def _fmt_value(x):
        if isinstance(x, (float, np.floating)):
            return f"{float(x):.4f}"
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        return str(x)

    def _to_markdown_table(df: pd.DataFrame) -> str:
        headers = list(df.columns)
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        body = []
        for row in df.itertuples(index=False):
            body.append("| " + " | ".join(_fmt_value(v) for v in row) + " |")
        return "\n".join([header_line, sep_line, *body])

    lines = []
    lines.append("# MTBS vs WFIGS Coherence Analysis (2021-2023)")
    lines.append("")
    lines.append("## Scope and Method")
    lines.append("- Fires filtered to 2021-01-01 through 2023-12-31 by reported start date.")
    lines.append("- Size floor applied to both datasets: **>= 1000 acres**.")
    lines.append("- VIIRS source: archive CSV inside `DL_FIRE_J1V-C2_718831.zip`.")
    lines.append("- Spatial linkage: point-in-polygon assignment to fire perimeters.")
    lines.append("- Temporal noise window for pre/post counts: +/- 30 days around each fire window.")
    lines.append("- MTBS caveat: MTBS perimeter data includes ignition date (`Ig_Date`) but no true fire-out date.")
    lines.append("  MTBS end-like comparisons therefore use `Post_ID` image date as a proxy.")
    lines.append("")
    lines.append("## Requested Metrics (Dataset-Level)")
    lines.append(_to_markdown_table(comparison_df))
    lines.append("")
    lines.append("## Shared Fires (MTBS vs WFIGS)")
    lines.append(f"- Matched pairs: **{match_summary.get('matched_pair_count', 0)}**")
    lines.append(f"- MTBS matched: **{match_summary.get('pct_mtbs_matched', 0.0) * 100:.1f}%**")
    lines.append(f"- WFIGS matched: **{match_summary.get('pct_wfigs_matched', 0.0) * 100:.1f}%**")
    if pair_df.empty:
        lines.append("- No shared pairs found with current matching thresholds.")
    else:
        lines.append(f"- Median overlap ratio (intersection/min-area): **{match_summary.get('median_overlap_ratio', float('nan')):.3f}**")
        lines.append(
            f"- Median |start-date difference| (WFIGS - MTBS): **{match_summary.get('median_abs_start_date_diff_days', float('nan')):.1f} days**"
        )
        lines.append(
            f"- Median absolute size difference: **{match_summary.get('median_abs_size_pct_diff', float('nan')) * 100:.1f}%**"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- Higher `in_window_share_total` means detections align better with reported fire windows.")
    lines.append("- Higher `noise_share_total` means more detections fall before start or after end (less coherent).")
    lines.append("- Higher `pct_days_few_hn_le3` means many low-signal days inside the reported window.")
    lines.append("- MTBS end-date related metrics should be treated as proxy-based, not literal fire-out behavior.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MTBS vs WFIGS coherence analysis with VIIRS linkage")
    parser.add_argument(
        "--wfigs-geojson",
        type=Path,
        default=Path("/Users/lorn/Downloads/WFIGS_Interagency_Perimeters_6781836551080060975.geojson"),
    )
    parser.add_argument(
        "--mtbs-zip",
        type=Path,
        default=Path("/Users/lorn/Code/gribcheck/data/external/mtbs_perimeter_data.zip"),
    )
    parser.add_argument(
        "--viirs-zip",
        type=Path,
        default=Path("/Users/lorn/Downloads/DL_FIRE_J1V-C2_718831.zip"),
    )
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--min-acres", type=float, default=1000.0)
    parser.add_argument("--window-pad-days", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/lorn/Code/gribcheck/reports/mtbs_wfigs_coherence_2021_2023"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading WFIGS fires...")
    wfigs_fires = load_wfigs_fires(
        path=args.wfigs_geojson,
        start_year=args.start_year,
        end_year=args.end_year,
        min_acres=args.min_acres,
    )
    print(f"WFIGS filtered fires: {len(wfigs_fires):,}")

    print("Loading MTBS fires...")
    mtbs_fires = load_mtbs_fires(
        zip_path=args.mtbs_zip,
        start_year=args.start_year,
        end_year=args.end_year,
        min_acres=args.min_acres,
    )
    print(f"MTBS filtered fires: {len(mtbs_fires):,}")

    print("Computing WFIGS VIIRS coherence metrics...")
    wfigs_df, wfigs_summary, wfigs_profile = compute_detection_metrics(
        fires=wfigs_fires,
        viirs_zip=args.viirs_zip,
        window_pad_days=args.window_pad_days,
        start_year=args.start_year,
        end_year=args.end_year,
        label="WFIGS",
    )

    print("Computing MTBS VIIRS coherence metrics...")
    mtbs_df, mtbs_summary, mtbs_profile = compute_detection_metrics(
        fires=mtbs_fires,
        viirs_zip=args.viirs_zip,
        window_pad_days=args.window_pad_days,
        start_year=args.start_year,
        end_year=args.end_year,
        label="MTBS",
    )

    print("Matching MTBS and WFIGS fires...")
    match_seed_df = match_mtbs_wfigs(mtbs_fires=mtbs_fires, wfigs_fires=wfigs_fires)
    pair_df = add_pairwise_differences(match_seed_df, mtbs_fires=mtbs_fires, wfigs_fires=wfigs_fires)
    match_summary = summarize_matches(pair_df, mtbs_count=len(mtbs_fires), wfigs_count=len(wfigs_fires))

    # Join coherence metrics on matched pairs for direct comparison.
    mtbs_for_join = mtbs_df.add_prefix("mtbs_")
    wfigs_for_join = wfigs_df.add_prefix("wfigs_")
    pair_metrics_df = (
        pair_df.merge(mtbs_for_join, left_on="mtbs_index", right_on="mtbs_fire_index", how="left")
        .merge(wfigs_for_join, left_on="wfigs_index", right_on="wfigs_fire_index", how="left")
    )

    # Additional "which is better" discriminator on matched pairs.
    if not pair_metrics_df.empty:
        pair_metrics_df["delta_in_window_share_wfigs_minus_mtbs"] = (
            pair_metrics_df["wfigs_in_window_share"] - pair_metrics_df["mtbs_in_window_share"]
        )
        pair_metrics_df["delta_noise_share_wfigs_minus_mtbs"] = (
            pair_metrics_df["wfigs_noise_share"] - pair_metrics_df["mtbs_noise_share"]
        )
        pair_metrics_df["delta_pct_days_with_any_hn_wfigs_minus_mtbs"] = (
            pair_metrics_df["wfigs_pct_active_days_with_any_hn"] - pair_metrics_df["mtbs_pct_active_days_with_any_hn"]
        )

    profile_df = pd.concat([mtbs_profile, wfigs_profile], ignore_index=True)

    print("Writing outputs...")
    mtbs_df.to_csv(out_dir / "mtbs_fire_metrics.csv", index=False)
    wfigs_df.to_csv(out_dir / "wfigs_fire_metrics.csv", index=False)
    pair_df.to_csv(out_dir / "mtbs_wfigs_matched_pairs.csv", index=False)
    pair_metrics_df.to_csv(out_dir / "mtbs_wfigs_matched_pairs_with_metrics.csv", index=False)
    profile_df.to_csv(out_dir / "relative_daily_profile_hn.csv", index=False)

    summary_df = pd.DataFrame([mtbs_summary, wfigs_summary])
    summary_df.to_csv(out_dir / "dataset_summary.csv", index=False)
    with (out_dir / "dataset_summary.json").open("w", encoding="utf-8") as fp:
        json.dump({"MTBS": mtbs_summary, "WFIGS": wfigs_summary, "match_summary": match_summary}, fp, indent=2)

    create_figures(mtbs_df=mtbs_df, wfigs_df=wfigs_df, profile_df=profile_df, figures_dir=fig_dir)
    write_markdown_report(
        output_path=out_dir / "coherence_report.md",
        mtbs_summary=mtbs_summary,
        wfigs_summary=wfigs_summary,
        match_summary=match_summary,
        pair_df=pair_df,
    )

    print("Done.")
    print(f"Report: {out_dir / 'coherence_report.md'}")


if __name__ == "__main__":
    main()
