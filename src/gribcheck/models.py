from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class VariableSpec:
    variable: str
    level: str
    channel_name: str | None = None
    output_column: str | None = None


@dataclass(frozen=True)
class PMDailyRecord:
    state_code: str
    county_code: str
    site_num: str
    station_id: str
    date_local: date
    latitude: float
    longitude: float
    pm25_value: float
    pm_source_code: str
    pm_source_poc: str
    pm_source_method: str


@dataclass(frozen=True)
class StationDailyHRRR:
    station_id: str
    date_local: date
    massden_daily_mean: float
    colmd_daily_mean: float
    hrrr_hours_found: int
    hrrr_hours_expected: int
    hrrr_day_complete: bool


@dataclass(frozen=True)
class RasterSampleIndex:
    sample_id: int
    fire_id: str
    run_time_utc: datetime
    split: str
    bbox_min_lon: float
    bbox_min_lat: float
    bbox_max_lon: float
    bbox_max_lat: float
    label_t_plus_12h_available: bool
    label_t_plus_24h_available: bool


@dataclass(frozen=True)
class FireRecord:
    unique_fire_id: str
    incident_name: str
    incident_type: str
    state: str
    start_time_utc: datetime
    end_time_utc: datetime
    start_date: date
    end_date: date
    size_acres: float
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    @property
    def center_lon(self) -> float:
        return (self.min_lon + self.max_lon) / 2.0

    @property
    def center_lat(self) -> float:
        return (self.min_lat + self.max_lat) / 2.0
