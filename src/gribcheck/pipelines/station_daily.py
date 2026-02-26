from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder

from gribcheck.config import PipelineConfig
from gribcheck.geo_utils import hrrr_transformer_to_xy
from gribcheck.hrrr import HRRRAnalysisReader

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StationDailyStats:
    station_count: int
    hour_count_attempted: int
    joined_rows: int


def _expected_hours_for_local_day(local_day: datetime.date, tz_name: str) -> int:
    tz = ZoneInfo(tz_name)
    start_local = datetime.combine(local_day, time(0, 0), tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)
    return int((end_utc - start_utc).total_seconds() // 3600)


def _resolve_station_timezones(stations: pd.DataFrame) -> pd.Series:
    finder = TimezoneFinder(in_memory=True)
    tz_values: list[str] = []
    for lat, lon in zip(stations["latitude"].to_numpy(), stations["longitude"].to_numpy()):
        tz_name = finder.timezone_at(lat=float(lat), lng=float(lon))
        if not tz_name:
            tz_name = finder.closest_timezone_at(lat=float(lat), lng=float(lon))
        tz_values.append(tz_name or "UTC")
    return pd.Series(tz_values, index=stations.index, name="timezone")


def _utc_hour_range(start_date, end_date) -> pd.DatetimeIndex:
    start_utc = datetime.combine(start_date, time(0, 0), tzinfo=timezone.utc) - timedelta(days=1)
    end_utc = datetime.combine(end_date, time(23, 0), tzinfo=timezone.utc) + timedelta(days=1)
    return pd.date_range(start=start_utc, end=end_utc, freq="1h", tz="UTC")


def run(
    config: PipelineConfig,
    max_hours: int | None = None,
    station_limit: int | None = None,
) -> StationDailyStats:
    pm_df = pd.read_parquet(config.paths.pm_output)
    if pm_df.empty:
        raise ValueError("PM dataset is empty; run ingest-pm first")

    pm_df["date_local"] = pd.to_datetime(pm_df["date_local"]).dt.date

    stations = (
        pm_df[["station_id", "latitude", "longitude", "state_name"]]
        .drop_duplicates(subset=["station_id"])
        .reset_index(drop=True)
    )
    if station_limit is not None:
        stations = stations.head(int(station_limit)).reset_index(drop=True)
        pm_df = pm_df[pm_df["station_id"].isin(stations["station_id"])].copy()
    stations["timezone"] = _resolve_station_timezones(stations)

    transformer = hrrr_transformer_to_xy()
    x_vals, y_vals = transformer.transform(stations["longitude"].to_numpy(), stations["latitude"].to_numpy())
    stations["x"] = x_vals
    stations["y"] = y_vals

    reader = HRRRAnalysisReader(config.hrrr)
    utc_hours = _utc_hour_range(config.run.start_date, config.run.end_date)
    if max_hours is not None:
        utc_hours = utc_hours[: int(max_hours)]

    by_tz: dict[str, np.ndarray] = {}
    tz_arr = stations["timezone"].to_numpy()
    for tz_name in sorted(set(tz_arr)):
        by_tz[tz_name] = np.where(tz_arr == tz_name)[0]

    station_ids = stations["station_id"].to_numpy()
    x_points = stations["x"].to_numpy(dtype=np.float64)
    y_points = stations["y"].to_numpy(dtype=np.float64)

    variable_specs = list(config.hrrr.station_variables)
    if len(variable_specs) != 2:
        raise ValueError("Station daily pipeline expects exactly 2 HRRR station variables")

    output_names = [spec.output_column or spec.variable.lower() for spec in variable_specs]

    # key -> [sum_var0, sum_var1, hour_count]
    agg: dict[tuple[str, datetime.date], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])

    LOGGER.info("Sampling %d UTC hours for %d stations", len(utc_hours), len(stations))

    for i, utc_hour in enumerate(utc_hours):
        if i % 200 == 0:
            LOGGER.info("Processing hour %d / %d (%s)", i + 1, len(utc_hours), utc_hour)

        sampled_vars: list[np.ndarray] = []
        missing_field = False
        for spec in variable_specs:
            da = reader.load_field(utc_hour.to_pydatetime(), spec)
            if da is None:
                missing_field = True
                break
            sampled = reader.bilinear_sample(da, x_points=x_points, y_points=y_points)
            sampled_vars.append(sampled)

        if missing_field:
            continue

        valid_mask = np.isfinite(sampled_vars[0]) & np.isfinite(sampled_vars[1])

        for tz_name, idxs in by_tz.items():
            if len(idxs) == 0:
                continue
            local_day = utc_hour.to_pydatetime().astimezone(ZoneInfo(tz_name)).date()
            for idx in idxs:
                if not valid_mask[idx]:
                    continue
                key = (station_ids[idx], local_day)
                rec = agg[key]
                rec[0] += float(sampled_vars[0][idx])
                rec[1] += float(sampled_vars[1][idx])
                rec[2] += 1.0

    rows: list[dict[str, object]] = []
    for (station_id, local_day), (sum0, sum1, count) in agg.items():
        if count <= 0:
            continue
        rows.append(
            {
                "station_id": station_id,
                "date_local": local_day,
                output_names[0]: sum0 / count,
                output_names[1]: sum1 / count,
                "hrrr_hours_found": int(count),
            }
        )

    if rows:
        hrrr_daily = pd.DataFrame(rows)
    else:
        hrrr_daily = pd.DataFrame(
            columns=["station_id", "date_local", *output_names, "hrrr_hours_found"]
        )

    station_tz = stations[["station_id", "timezone"]].copy()

    joined = pm_df.merge(hrrr_daily, on=["station_id", "date_local"], how="left")
    joined = joined.merge(station_tz, on="station_id", how="left")

    joined["hrrr_hours_found"] = joined["hrrr_hours_found"].fillna(0).astype(int)
    joined["hrrr_hours_expected"] = joined.apply(
        lambda row: _expected_hours_for_local_day(row["date_local"], row["timezone"]),
        axis=1,
    )
    joined["hrrr_day_complete"] = joined["hrrr_hours_found"] == joined["hrrr_hours_expected"]

    for col in output_names:
        if col not in joined.columns:
            joined[col] = np.nan

    joined = joined.sort_values(["date_local", "station_id"]).reset_index(drop=True)

    config.paths.station_daily_output.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(config.paths.station_daily_output, index=False)

    LOGGER.info("Station daily join complete: %d rows written", len(joined))

    return StationDailyStats(
        station_count=len(stations),
        hour_count_attempted=len(utc_hours),
        joined_rows=len(joined),
    )
