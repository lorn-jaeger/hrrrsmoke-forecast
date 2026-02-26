from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import ijson
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from gribcheck.config import PipelineConfig
from gribcheck.date_utils import parse_optional_datetime
from gribcheck.geo_utils import bbox_radius_km, haversine_km, nested_coordinates_bounds
from gribcheck.models import FireRecord

LOGGER = logging.getLogger(__name__)


def _to_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def _choose_fire_end(props: dict) -> datetime | None:
    fields = [
        "attr_FireOutDateTime",
        "attr_ContainmentDateTime",
        "attr_ControlDateTime",
        "poly_PolygonDateTime",
    ]
    for key in fields:
        dt = parse_optional_datetime(props.get(key))
        if dt is not None:
            return dt
    return None


def _choose_fire_start(props: dict) -> datetime | None:
    return parse_optional_datetime(props.get("attr_FireDiscoveryDateTime"))


def _fire_size_acres(props: dict) -> float | None:
    primary = _to_float(props.get("attr_IncidentSize"))
    if primary is not None and primary > 0:
        return primary
    fallback = _to_float(props.get("poly_GISAcres"))
    if fallback is not None and fallback > 0:
        return fallback
    return None


def _feature_to_fire_record(
    feature: dict,
    min_size_acres: float,
    run_start: date,
    run_end: date,
    incident_type: str,
) -> FireRecord | None:
    props = feature.get("properties", {})
    geometry = feature.get("geometry", {})

    if str(props.get("attr_IncidentTypeCategory", "")).strip() != incident_type:
        return None

    start_dt = _choose_fire_start(props)
    if start_dt is None:
        return None

    if not (run_start <= start_dt.date() <= run_end):
        return None

    end_dt = _choose_fire_end(props)
    if end_dt is None:
        return None

    if end_dt < start_dt:
        return None

    size_acres = _fire_size_acres(props)
    if size_acres is None or size_acres < min_size_acres:
        return None

    coords = geometry.get("coordinates")
    if coords is None:
        return None

    try:
        min_lon, min_lat, max_lon, max_lat = nested_coordinates_bounds(coords)
    except Exception:
        return None

    unique_fire_id = (
        str(props.get("attr_UniqueFireIdentifier") or "").strip()
        or str(props.get("poly_IRWINID") or "").strip()
        or str(props.get("attr_LocalIncidentIdentifier") or "").strip()
    )
    if not unique_fire_id:
        return None

    incident_name = str(props.get("attr_IncidentName") or props.get("poly_IncidentName") or "Unknown").strip()
    state = str(props.get("attr_POOState") or "").strip()

    return FireRecord(
        unique_fire_id=unique_fire_id,
        incident_name=incident_name,
        incident_type=incident_type,
        state=state,
        start_time_utc=start_dt.astimezone(timezone.utc),
        end_time_utc=end_dt.astimezone(timezone.utc),
        start_date=start_dt.date(),
        end_date=end_dt.date(),
        size_acres=float(size_acres),
        min_lon=float(min_lon),
        min_lat=float(min_lat),
        max_lon=float(max_lon),
        max_lat=float(max_lat),
    )


def load_filtered_fire_records(config: PipelineConfig) -> list[FireRecord]:
    path = config.paths.wildfire_perimeter_geojson
    records: list[FireRecord] = []

    LOGGER.info("Streaming wildfire features from %s", path)
    with Path(path).open("rb") as f:
        for feature in ijson.items(f, "features.item"):
            rec = _feature_to_fire_record(
                feature=feature,
                min_size_acres=config.wildfire.min_size_acres,
                run_start=config.run.start_date,
                run_end=config.run.end_date,
                incident_type=config.wildfire.incident_type,
            )
            if rec is not None:
                records.append(rec)

    LOGGER.info("Loaded %d wildfire records after filtering", len(records))
    return records


def build_daily_fire_index(records: list[FireRecord]) -> dict[date, list[FireRecord]]:
    index: dict[date, list[FireRecord]] = defaultdict(list)
    for rec in records:
        day = rec.start_date
        while day <= rec.end_date:
            index[day].append(rec)
            day += timedelta(days=1)
    return dict(index)


def assign_fire_proximity_bins(
    df: pd.DataFrame,
    daily_fires: dict[date, list[FireRecord]],
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    date_col: str = "date_local",
) -> pd.Series:
    out = pd.Series(index=df.index, dtype="object")

    grouped = df.groupby(date_col, sort=False)
    for local_day, day_df in grouped:
        fires = daily_fires.get(local_day, [])
        if not fires:
            out.loc[day_df.index] = ">100"
            continue

        centers = np.array([[rec.center_lat, rec.center_lon] for rec in fires], dtype=np.float64)
        radii = np.array([bbox_radius_km((rec.min_lon, rec.min_lat, rec.max_lon, rec.max_lat)) for rec in fires])

        lat_arr = day_df[lat_col].to_numpy(dtype=np.float64)
        lon_arr = day_df[lon_col].to_numpy(dtype=np.float64)

        # Use a fast nearest-neighbor prefilter, then exact haversine on few candidates.
        lat0 = np.deg2rad(float(np.mean(centers[:, 0])))
        scale_lon = np.cos(lat0)
        centers_xy = np.column_stack((centers[:, 1] * scale_lon, centers[:, 0]))
        station_xy = np.column_stack((lon_arr * scale_lon, lat_arr))
        tree = cKDTree(centers_xy)
        k = min(8, len(centers))
        _, nn_idx = tree.query(station_xy, k=k)
        nn_idx = np.atleast_2d(nn_idx)
        if nn_idx.shape[0] != len(lat_arr):
            nn_idx = nn_idx.T

        day_bins: list[str] = []
        for i, (lat, lon) in enumerate(zip(lat_arr, lon_arr)):
            candidate_idx = np.atleast_1d(nn_idx[i]).astype(int)
            c_subset = centers[candidate_idx]
            r_subset = radii[candidate_idx]
            center_dist = np.array([haversine_km(lat, lon, c_lat, c_lon) for c_lat, c_lon in c_subset])
            perimeter_dist = np.maximum(center_dist - r_subset, 0.0)
            min_dist = float(np.min(perimeter_dist))
            if min_dist <= 30.0:
                day_bins.append("<=30")
            elif min_dist <= 100.0:
                day_bins.append("30-100")
            else:
                day_bins.append(">100")

        out.loc[day_df.index] = day_bins

    out = out.fillna(">100")
    out.name = "fire_proximity_bin"
    return out
