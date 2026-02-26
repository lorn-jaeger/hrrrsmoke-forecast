from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from gribcheck.config import PipelineConfig
from gribcheck.date_utils import parse_pm_date

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PMIngestStats:
    rows_primary_raw: int
    rows_fallback_raw: int
    rows_primary_dedup: int
    rows_fallback_dedup: int
    rows_final: int


def _read_pm_zip(zip_path: Path, parameter_code: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        csv_candidates = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_candidates:
            raise ValueError(f"No CSV found in {zip_path}")
        csv_name = csv_candidates[0]
        with zf.open(csv_name) as fp:
            df = pd.read_csv(fp, dtype=str, low_memory=False)

    if "Parameter Code" in df.columns:
        df = df[df["Parameter Code"].astype(str).str.strip('"') == parameter_code]

    df["pm_source_code"] = parameter_code
    return df


def _normalize_pm(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    rename_map = {
        "State Code": "state_code",
        "County Code": "county_code",
        "Site Num": "site_num",
        "POC": "poc",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Parameter Name": "parameter_name",
        "Sample Duration": "sample_duration",
        "Pollutant Standard": "pollutant_standard",
        "Date Local": "date_local_raw",
        "Units of Measure": "units",
        "Event Type": "event_type",
        "Observation Count": "observation_count",
        "Observation Percent": "observation_percent",
        "Arithmetic Mean": "arithmetic_mean",
        "AQI": "aqi",
        "Method Code": "method_code",
        "Method Name": "method_name",
        "State Name": "state_name",
        "County Name": "county_name",
        "City Name": "city_name",
        "CBSA Name": "cbsa_name",
        "Date of Last Change": "date_last_change_raw",
    }

    missing = [col for col in rename_map if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected PM columns: {missing}")

    out = df.rename(columns=rename_map)[list(rename_map.values()) + ["pm_source_code"]].copy()

    out["state_code"] = out["state_code"].fillna("").astype(str).str.strip().str.strip('"').str.zfill(2)
    out["county_code"] = out["county_code"].fillna("").astype(str).str.strip().str.strip('"').str.zfill(3)
    out["site_num"] = out["site_num"].fillna("").astype(str).str.strip().str.strip('"').str.zfill(4)
    out["station_id"] = out["state_code"] + "-" + out["county_code"] + "-" + out["site_num"]

    out["date_local"] = out["date_local_raw"].map(parse_pm_date)
    out = out[(out["date_local"] >= start_date) & (out["date_local"] <= end_date)].copy()

    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out["observation_count"] = pd.to_numeric(out["observation_count"], errors="coerce")
    out["observation_percent"] = pd.to_numeric(out["observation_percent"], errors="coerce")
    out["arithmetic_mean"] = pd.to_numeric(out["arithmetic_mean"], errors="coerce")
    out["aqi"] = pd.to_numeric(out["aqi"], errors="coerce")
    out["poc_numeric"] = pd.to_numeric(out["poc"], errors="coerce")
    out["date_last_change"] = pd.to_datetime(out["date_last_change_raw"], errors="coerce")

    out = out.dropna(subset=["latitude", "longitude", "arithmetic_mean"]).copy()

    return out


def _dedupe_within_code(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    sort_cols = [
        "station_id",
        "date_local",
        "observation_percent",
        "poc_numeric",
        "date_last_change",
    ]

    ranked = df.sort_values(
        by=sort_cols,
        ascending=[True, True, False, True, False],
        na_position="last",
    )
    dedup = ranked.drop_duplicates(subset=["station_id", "date_local"], keep="first").copy()

    dedup["pm_source_poc"] = dedup["poc"].astype(str)
    dedup["pm_source_method"] = dedup["method_name"].astype(str)
    dedup["pm_source_parameter_name"] = dedup["parameter_name"].astype(str)

    return dedup


def _combine_primary_fallback(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    primary = primary.copy()
    fallback = fallback.copy()
    primary["source_priority"] = 0
    fallback["source_priority"] = 1

    combined = pd.concat([primary, fallback], ignore_index=True)
    combined = combined.sort_values(
        by=["station_id", "date_local", "source_priority", "observation_percent", "poc_numeric", "date_last_change"],
        ascending=[True, True, True, False, True, False],
        na_position="last",
    )

    combined = combined.drop_duplicates(subset=["station_id", "date_local"], keep="first").copy()

    keep_cols = [
        "state_code",
        "county_code",
        "site_num",
        "station_id",
        "date_local",
        "latitude",
        "longitude",
        "arithmetic_mean",
        "units",
        "observation_count",
        "observation_percent",
        "aqi",
        "event_type",
        "state_name",
        "county_name",
        "city_name",
        "cbsa_name",
        "pm_source_code",
        "pm_source_poc",
        "pm_source_method",
        "pm_source_parameter_name",
        "pollutant_standard",
        "sample_duration",
        "date_last_change",
    ]

    return combined[keep_cols].rename(columns={"arithmetic_mean": "pm25_value"}).reset_index(drop=True)


def _collect_pm_files(downloads_dir: Path, file_glob: str, start_date: date, end_date: date) -> list[Path]:
    files = sorted(downloads_dir.glob(file_glob))
    selected: list[Path] = []
    for path in files:
        stem = path.stem
        parts = stem.split("_")
        if not parts:
            continue
        year_txt = parts[-1]
        try:
            year = int(year_txt)
        except ValueError:
            continue
        if start_date.year <= year <= end_date.year:
            selected.append(path)
    return selected


def run(config: PipelineConfig) -> PMIngestStats:
    files_primary = _collect_pm_files(
        config.paths.downloads_dir,
        config.pm.file_glob_primary,
        config.run.start_date,
        config.run.end_date,
    )
    files_fallback = _collect_pm_files(
        config.paths.downloads_dir,
        config.pm.file_glob_fallback,
        config.run.start_date,
        config.run.end_date,
    )

    if not files_primary:
        raise FileNotFoundError(f"No primary PM files found with pattern {config.pm.file_glob_primary}")
    if not files_fallback:
        raise FileNotFoundError(f"No fallback PM files found with pattern {config.pm.file_glob_fallback}")

    LOGGER.info("Reading %d primary PM zip files", len(files_primary))
    df_primary_raw = pd.concat(
        [_read_pm_zip(path, config.pm.code_primary) for path in files_primary],
        ignore_index=True,
    )

    LOGGER.info("Reading %d fallback PM zip files", len(files_fallback))
    df_fallback_raw = pd.concat(
        [_read_pm_zip(path, config.pm.code_fallback) for path in files_fallback],
        ignore_index=True,
    )

    df_primary = _normalize_pm(df_primary_raw, config.run.start_date, config.run.end_date)
    df_fallback = _normalize_pm(df_fallback_raw, config.run.start_date, config.run.end_date)

    primary_dedup = _dedupe_within_code(df_primary)
    fallback_dedup = _dedupe_within_code(df_fallback)

    final = _combine_primary_fallback(primary_dedup, fallback_dedup)

    config.paths.pm_output.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(config.paths.pm_output, index=False)

    LOGGER.info("PM ingest complete: %d rows written to %s", len(final), config.paths.pm_output)

    return PMIngestStats(
        rows_primary_raw=len(df_primary_raw),
        rows_fallback_raw=len(df_fallback_raw),
        rows_primary_dedup=len(primary_dedup),
        rows_fallback_dedup=len(fallback_dedup),
        rows_final=len(final),
    )
