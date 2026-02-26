from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from gribcheck.models import VariableSpec


@dataclass(frozen=True)
class RunConfig:
    start_date: date
    end_date: date


@dataclass(frozen=True)
class PathsConfig:
    downloads_dir: Path
    wildfire_perimeter_geojson: Path
    intermediate_dir: Path
    processed_dir: Path
    reports_dir: Path
    figures_dir: Path
    qa_tiff_dir: Path
    pm_output: Path
    station_daily_output: Path
    accuracy_summary_output: Path
    accuracy_report_output: Path
    wildfire_zarr_output: Path
    wildfire_index_output: Path
    dataset_build_log_output: Path


@dataclass(frozen=True)
class PMConfig:
    code_primary: str
    code_fallback: str
    file_glob_primary: str
    file_glob_fallback: str


@dataclass(frozen=True)
class HRRRConfig:
    bucket: str
    product: str
    anonymous: bool
    station_variables: tuple[VariableSpec, ...]


@dataclass(frozen=True)
class ViirsConfig:
    file_glob: str
    cache_parquet: Path


@dataclass(frozen=True)
class WildfireConfig:
    incident_type: str
    min_size_acres: float
    buffer_km: float
    patch_size: tuple[int, int]
    frp_channel_name: str
    analysis_variables: tuple[VariableSpec, ...]
    label_variable: VariableSpec
    label_lead_hours: tuple[int, ...]


@dataclass(frozen=True)
class StorageConfig:
    budget_gb: float
    dtype: str
    compressor: str
    projection_check_interval: int
    drop_upper_air_and_dzdt: bool
    two_hour_cadence_after_72h: bool
    cap_samples_per_fire: int


@dataclass(frozen=True)
class SplitConfig:
    train_start: date
    train_end: date
    val_start: date
    val_end: date
    test_start: date
    test_end: date


@dataclass(frozen=True)
class PipelineConfig:
    run: RunConfig
    paths: PathsConfig
    pm: PMConfig
    hrrr: HRRRConfig
    viirs: ViirsConfig
    wildfire: WildfireConfig
    storage: StorageConfig
    split: SplitConfig


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_variable_spec(raw: dict[str, Any]) -> VariableSpec:
    return VariableSpec(
        variable=str(raw["variable"]),
        level=str(raw["level"]),
        channel_name=(str(raw["channel_name"]) if raw.get("channel_name") else None),
        output_column=(str(raw["output_column"]) if raw.get("output_column") else None),
    )


def load_config(path: str | Path) -> PipelineConfig:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    run = RunConfig(
        start_date=_parse_date(raw["run"]["start_date"]),
        end_date=_parse_date(raw["run"]["end_date"]),
    )

    paths_raw = raw["paths"]
    paths = PathsConfig(
        downloads_dir=Path(paths_raw["downloads_dir"]).expanduser(),
        wildfire_perimeter_geojson=Path(paths_raw["wildfire_perimeter_geojson"]).expanduser(),
        intermediate_dir=Path(paths_raw["intermediate_dir"]).expanduser(),
        processed_dir=Path(paths_raw["processed_dir"]).expanduser(),
        reports_dir=Path(paths_raw["reports_dir"]).expanduser(),
        figures_dir=Path(paths_raw["figures_dir"]).expanduser(),
        qa_tiff_dir=Path(paths_raw["qa_tiff_dir"]).expanduser(),
        pm_output=Path(paths_raw["pm_output"]).expanduser(),
        station_daily_output=Path(paths_raw["station_daily_output"]).expanduser(),
        accuracy_summary_output=Path(paths_raw["accuracy_summary_output"]).expanduser(),
        accuracy_report_output=Path(paths_raw["accuracy_report_output"]).expanduser(),
        wildfire_zarr_output=Path(paths_raw["wildfire_zarr_output"]).expanduser(),
        wildfire_index_output=Path(paths_raw["wildfire_index_output"]).expanduser(),
        dataset_build_log_output=Path(paths_raw["dataset_build_log_output"]).expanduser(),
    )

    pm = PMConfig(
        code_primary=str(raw["pm"]["code_primary"]),
        code_fallback=str(raw["pm"]["code_fallback"]),
        file_glob_primary=str(raw["pm"]["file_glob_primary"]),
        file_glob_fallback=str(raw["pm"]["file_glob_fallback"]),
    )

    hrrr = HRRRConfig(
        bucket=str(raw["hrrr"]["bucket"]),
        product=str(raw["hrrr"]["product"]),
        anonymous=bool(raw["hrrr"]["anonymous"]),
        station_variables=tuple(
            _parse_variable_spec(item) for item in raw["hrrr"]["station_variables"]
        ),
    )

    viirs = ViirsConfig(
        file_glob=str(raw["viirs"]["file_glob"]),
        cache_parquet=Path(raw["viirs"]["cache_parquet"]).expanduser(),
    )

    wildfire = WildfireConfig(
        incident_type=str(raw["wildfire"]["incident_type"]),
        min_size_acres=float(raw["wildfire"]["min_size_acres"]),
        buffer_km=float(raw["wildfire"]["buffer_km"]),
        patch_size=tuple(int(v) for v in raw["wildfire"]["patch_size"]),
        frp_channel_name=str(raw["wildfire"]["frp_channel_name"]),
        analysis_variables=tuple(
            _parse_variable_spec(item) for item in raw["wildfire"]["analysis_variables"]
        ),
        label_variable=VariableSpec(
            variable=str(raw["wildfire"]["label_variable"]["variable"]),
            level=str(raw["wildfire"]["label_variable"]["level"]),
            channel_name="target_massden",
        ),
        label_lead_hours=tuple(int(v) for v in raw["wildfire"]["label_variable"]["lead_hours"]),
    )

    storage = StorageConfig(
        budget_gb=float(raw["storage"]["budget_gb"]),
        dtype=str(raw["storage"]["dtype"]),
        compressor=str(raw["storage"]["compressor"]),
        projection_check_interval=int(raw["storage"]["projection_check_interval"]),
        drop_upper_air_and_dzdt=bool(raw["storage"]["reduction_order"]["drop_upper_air_and_dzdt"]),
        two_hour_cadence_after_72h=bool(raw["storage"]["reduction_order"]["two_hour_cadence_after_72h"]),
        cap_samples_per_fire=int(raw["storage"]["reduction_order"]["cap_samples_per_fire"]),
    )

    split = SplitConfig(
        train_start=_parse_date(raw["split"]["train_start"]),
        train_end=_parse_date(raw["split"]["train_end"]),
        val_start=_parse_date(raw["split"]["val_start"]),
        val_end=_parse_date(raw["split"]["val_end"]),
        test_start=_parse_date(raw["split"]["test_start"]),
        test_end=_parse_date(raw["split"]["test_end"]),
    )

    return PipelineConfig(
        run=run,
        paths=paths,
        pm=pm,
        hrrr=hrrr,
        viirs=viirs,
        wildfire=wildfire,
        storage=storage,
        split=split,
    )


def ensure_output_dirs(config: PipelineConfig) -> None:
    dirs = [
        config.paths.intermediate_dir,
        config.paths.processed_dir,
        config.paths.reports_dir,
        config.paths.figures_dir,
        config.paths.qa_tiff_dir,
    ]
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)
