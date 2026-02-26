from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from gribcheck.config import load_config
from gribcheck.pipelines import pm_ingest, station_daily


def _write_pm_zip(path: Path, parameter_code: str):
    csv_name = f"daily_{parameter_code}_2021.csv"
    content = "\n".join(
        [
            "State Code,County Code,Site Num,Parameter Code,POC,Latitude,Longitude,Datum,Parameter Name,Sample Duration,Pollutant Standard,Date Local,Units of Measure,Event Type,Observation Count,Observation Percent,Arithmetic Mean,1st Max Value,1st Max Hour,AQI,Method Code,Method Name,Local Site Name,Address,State Name,County Name,City Name,CBSA Name,Date of Last Change",
            f"01,001,0001,{parameter_code},1,0.2,0.2,NAD83,PM,24 HOUR,std,2021-01-01,ug/m3,None,1,100,10,10,0,20,1,M,site,a,Alabama,X,Y,Z,2024-01-01",
        ]
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, content)


def test_station_resume_skips_processed_hours(monkeypatch, tmp_path: Path):
    downloads = tmp_path / "downloads"
    downloads.mkdir()

    _write_pm_zip(downloads / "daily_88101_2021.zip", "88101")
    _write_pm_zip(downloads / "daily_88502_2021.zip", "88502")

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"""
run:
  start_date: "2021-01-01"
  end_date: "2021-01-01"
paths:
  downloads_dir: "{downloads}"
  wildfire_perimeter_geojson: "{tmp_path / 'wf.geojson'}"
  intermediate_dir: "{tmp_path / 'intermediate'}"
  processed_dir: "{tmp_path / 'processed'}"
  reports_dir: "{tmp_path / 'reports'}"
  figures_dir: "{tmp_path / 'reports' / 'figures'}"
  qa_tiff_dir: "{tmp_path / 'qa'}"
  pm_output: "{tmp_path / 'intermediate' / 'pm.parquet'}"
  station_daily_output: "{tmp_path / 'processed' / 'station.parquet'}"
  accuracy_summary_output: "{tmp_path / 'processed' / 'accuracy.parquet'}"
  accuracy_report_output: "{tmp_path / 'reports' / 'accuracy.md'}"
  wildfire_zarr_output: "{tmp_path / 'processed' / 'wf.zarr'}"
  wildfire_index_output: "{tmp_path / 'processed' / 'wf.parquet'}"
  dataset_build_log_output: "{tmp_path / 'processed' / 'wf.json'}"
pm:
  code_primary: "88101"
  code_fallback: "88502"
  file_glob_primary: "daily_88101_*.zip"
  file_glob_fallback: "daily_88502_*.zip"
hrrr:
  bucket: "hrrrzarr"
  product: "sfc"
  anonymous: true
  station_variables:
    - variable: "MASSDEN"
      level: "8m_above_ground"
      output_column: "massden_daily_mean"
    - variable: "COLMD"
      level: "entire_atmosphere_single_layer"
      output_column: "colmd_daily_mean"
viirs:
  file_glob: "DL_FIRE_J1*.zip"
  cache_parquet: "{tmp_path / 'intermediate' / 'viirs.parquet'}"
wildfire:
  incident_type: "WF"
  min_size_acres: 50
  buffer_km: 30
  patch_size: [256, 256]
  frp_channel_name: "VIIRS_FRP_1h"
  analysis_variables:
    - variable: "MASSDEN"
      level: "8m_above_ground"
      channel_name: "MASSDEN_8m"
  label_variable:
    variable: "MASSDEN"
    level: "8m_above_ground"
    lead_hours: [12, 24]
storage:
  budget_gb: 1
  dtype: "float16"
  compressor: "zstd"
  projection_check_interval: 100
  reduction_order:
    drop_upper_air_and_dzdt: true
    two_hour_cadence_after_72h: true
    cap_samples_per_fire: 10
split:
  train_start: "2021-01-01"
  train_end: "2021-01-01"
  val_start: "2021-01-02"
  val_end: "2021-01-03"
  test_start: "2021-01-04"
  test_end: "2021-01-05"
        """,
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    pm_ingest.run(cfg)

    class DummyReader:
        calls = 0

        def __init__(self, _cfg):
            pass

        def load_field(self, _run_time, spec):
            DummyReader.calls += 1
            v = 1.0 if spec.variable == "MASSDEN" else 2.0
            return xr.DataArray(
                np.full((2, 2), v, dtype=np.float32),
                dims=("y", "x"),
                coords={"x": np.array([0.0, 1.0]), "y": np.array([0.0, 1.0])},
            )

        @staticmethod
        def bilinear_sample(field, x_points, y_points):
            return np.full(len(x_points), float(field.values.mean()), dtype=np.float32)

    class DummyTransformer:
        def transform(self, lon, lat):
            return lon, lat

    monkeypatch.setattr(station_daily, "HRRRAnalysisReader", DummyReader)
    monkeypatch.setattr(station_daily, "hrrr_transformer_to_xy", lambda: DummyTransformer())
    monkeypatch.setattr(
        station_daily,
        "_resolve_station_timezones",
        lambda stations: pd.Series(["UTC"] * len(stations), index=stations.index),
    )

    station_daily.run(cfg, max_hours=1, workers=1, resume=True, checkpoint_flush_hours=1)
    assert DummyReader.calls == 2

    station_daily.run(cfg, max_hours=2, workers=1, resume=True, checkpoint_flush_hours=1)
    assert DummyReader.calls == 4

    station_daily.run(cfg, max_hours=2, workers=1, resume=True, checkpoint_flush_hours=1)
    assert DummyReader.calls == 4
