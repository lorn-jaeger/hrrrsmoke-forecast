from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from gribcheck.config import load_config
from gribcheck.models import FireRecord
from gribcheck.pipelines import wildfire_raster


def test_wildfire_resume_skips_completed_samples(monkeypatch, tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"""
run:
  start_date: "2021-01-01"
  end_date: "2021-01-31"
paths:
  downloads_dir: "{tmp_path / 'downloads'}"
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
  patch_size: [4, 4]
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
  projection_check_interval: 10
  reduction_order:
    drop_upper_air_and_dzdt: true
    two_hour_cadence_after_72h: true
    cap_samples_per_fire: 100
split:
  train_start: "2021-01-01"
  train_end: "2021-01-15"
  val_start: "2021-01-16"
  val_end: "2021-01-20"
  test_start: "2021-01-21"
  test_end: "2021-01-31"
        """,
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)

    rec = FireRecord(
        unique_fire_id="fire-1",
        incident_name="Test Fire",
        incident_type="WF",
        state="CA",
        start_time_utc=datetime(2021, 1, 1, 0, tzinfo=timezone.utc),
        end_time_utc=datetime(2021, 1, 3, 23, tzinfo=timezone.utc),
        start_date=date(2021, 1, 1),
        end_date=date(2021, 1, 3),
        size_acres=9999.0,
        min_lon=0.0,
        min_lat=0.0,
        max_lon=1.0,
        max_lat=1.0,
    )

    class DummyReader:
        calls = 0

        def __init__(self, _cfg, max_cache_entries: int = 0):
            self.cache_hits = 0
            self.cache_misses = 0

        def load_field(self, _run_time, _spec):
            DummyReader.calls += 1
            self.cache_misses += 1
            return xr.DataArray(
                np.full((2, 2), 1.0, dtype=np.float32),
                dims=("y", "x"),
                coords={"x": np.array([0.0, 1.0]), "y": np.array([0.0, 1.0])},
            )

    class DummyTransformer:
        def transform(self, lon, lat):
            return lon, lat

    class DummyRasterizer:
        @staticmethod
        def patch_for_hour(*args, **kwargs):
            _ = args, kwargs
            return np.zeros((4, 4), dtype=np.float32)

    monkeypatch.setattr(wildfire_raster, "load_filtered_fire_records", lambda _cfg: [rec])
    monkeypatch.setattr(wildfire_raster, "HRRRAnalysisReader", DummyReader)
    monkeypatch.setattr(wildfire_raster, "hrrr_transformer_to_xy", lambda: DummyTransformer())
    monkeypatch.setattr(wildfire_raster, "load_or_build_viirs_hourly_points", lambda _cfg: pd.DataFrame())
    monkeypatch.setattr(wildfire_raster, "build_viirs_rasterizer", lambda _df: DummyRasterizer())

    stats1 = wildfire_raster.run(
        cfg,
        max_samples_total=1,
        workers=1,
        sample_hours_utc=(0, 6, 12, 18),
        next_day_only=True,
        daily_aggregate=True,
        resume=True,
        checkpoint_flush_samples=1,
    )
    assert stats1.sample_count == 1
    calls_after_first = DummyReader.calls

    stats2 = wildfire_raster.run(
        cfg,
        max_samples_total=2,
        workers=1,
        sample_hours_utc=(0, 6, 12, 18),
        next_day_only=True,
        daily_aggregate=True,
        resume=True,
        checkpoint_flush_samples=1,
    )
    assert stats2.sample_count == 2
    calls_after_second = DummyReader.calls
    assert calls_after_second > calls_after_first

    stats3 = wildfire_raster.run(
        cfg,
        max_samples_total=2,
        workers=1,
        sample_hours_utc=(0, 6, 12, 18),
        next_day_only=True,
        daily_aggregate=True,
        resume=True,
        checkpoint_flush_samples=1,
    )
    assert stats3.sample_count == 2
    assert DummyReader.calls == calls_after_second

    index_df = pd.read_parquet(cfg.paths.wildfire_index_output)
    assert len(index_df) == 2
    assert index_df["sample_key"].nunique() == 2

    checkpoint_path = cfg.paths.wildfire_index_output.with_suffix(".checkpoint.jsonl")
    assert checkpoint_path.exists()

    zarr_group = zarr.open_group(str(cfg.paths.wildfire_zarr_output), mode="r")
    assert int(zarr_group["inputs/VIIRS_FRP_1h"].shape[0]) == 2
    assert int(zarr_group["labels/t_plus_24h"].shape[0]) == 2
