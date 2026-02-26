from __future__ import annotations

import logging
import zipfile
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from gribcheck.config import PipelineConfig
from gribcheck.geo_utils import hrrr_transformer_to_xy

LOGGER = logging.getLogger(__name__)


REQUIRED_COLUMNS = ["latitude", "longitude", "acq_date", "acq_time", "frp"]


class VIIRSFRPRasterizer:
    def __init__(self, hour_to_points_xyf: dict[datetime, np.ndarray]):
        self.hour_to_points_xyf = hour_to_points_xyf

    def patch_for_hour(
        self,
        run_time_utc: datetime,
        bounds_xy: tuple[float, float, float, float],
        patch_size: tuple[int, int],
    ) -> np.ndarray:
        key = run_time_utc.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        pts = self.hour_to_points_xyf.get(key)
        h, w = patch_size
        raster = np.zeros((h, w), dtype=np.float32)

        if pts is None or pts.size == 0:
            return raster

        xmin, ymin, xmax, ymax = bounds_xy
        if xmax <= xmin or ymax <= ymin:
            return raster

        x = pts[:, 0]
        y = pts[:, 1]
        frp = pts[:, 2]

        in_bounds = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        if not np.any(in_bounds):
            return raster

        x = x[in_bounds]
        y = y[in_bounds]
        frp = frp[in_bounds]

        x_norm = (x - xmin) / (xmax - xmin)
        y_norm = (y - ymin) / (ymax - ymin)

        ix = np.clip(np.rint(x_norm * (w - 1)).astype(np.int32), 0, w - 1)
        iy = np.clip(np.rint(y_norm * (h - 1)).astype(np.int32), 0, h - 1)

        np.add.at(raster, (iy, ix), frp.astype(np.float32, copy=False))
        return raster


def _parse_viirs_datetime(acq_date: pd.Series, acq_time: pd.Series) -> pd.Series:
    time_str = acq_time.astype(str).str.strip().str.zfill(4)
    dt_text = acq_date.astype(str).str.strip() + " " + time_str.str.slice(0, 2) + ":" + time_str.str.slice(2, 4)
    return pd.to_datetime(dt_text, format="%Y-%m-%d %H:%M", errors="coerce", utc=True)


def _read_viirs_zip(zip_path: Path, window_start: datetime, window_end: datetime) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as zf:
        csv_members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        for member in csv_members:
            with zf.open(member) as fp:
                for chunk in pd.read_csv(fp, usecols=REQUIRED_COLUMNS, chunksize=300_000, low_memory=False):
                    chunk["acq_datetime"] = _parse_viirs_datetime(chunk["acq_date"], chunk["acq_time"])
                    chunk = chunk.dropna(subset=["acq_datetime", "latitude", "longitude", "frp"]).copy()
                    chunk = chunk[
                        (chunk["acq_datetime"] >= window_start)
                        & (chunk["acq_datetime"] <= window_end)
                    ]
                    if chunk.empty:
                        continue
                    chunk["hour_utc"] = chunk["acq_datetime"].dt.floor("h")
                    pieces.append(chunk[["latitude", "longitude", "frp", "hour_utc"]])

    if not pieces:
        return pd.DataFrame(columns=["latitude", "longitude", "frp", "hour_utc"])

    return pd.concat(pieces, ignore_index=True)


def load_or_build_viirs_hourly_points(config: PipelineConfig) -> pd.DataFrame:
    cache_path = config.viirs.cache_parquet
    if cache_path.exists():
        LOGGER.info("Loading VIIRS FRP cache from %s", cache_path)
        df = pd.read_parquet(cache_path)
        df["hour_utc"] = pd.to_datetime(df["hour_utc"], utc=True)
        return df

    viirs_files = sorted(config.paths.downloads_dir.glob(config.viirs.file_glob))
    if not viirs_files:
        raise FileNotFoundError(
            f"No VIIRS files found in {config.paths.downloads_dir} with pattern {config.viirs.file_glob}"
        )

    window_start = datetime.combine(config.run.start_date, time(0, 0), tzinfo=timezone.utc) - timedelta(days=2)
    window_end = datetime.combine(config.run.end_date, time(23, 59), tzinfo=timezone.utc) + timedelta(days=64)

    LOGGER.info("Building VIIRS FRP cache from %d file(s)", len(viirs_files))
    frames = [_read_viirs_zip(path, window_start=window_start, window_end=window_end) for path in viirs_files]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["latitude", "longitude", "frp", "hour_utc"])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    LOGGER.info("VIIRS FRP cache written: %d rows", len(df))
    return df


def build_viirs_rasterizer(df: pd.DataFrame) -> VIIRSFRPRasterizer:
    if df.empty:
        return VIIRSFRPRasterizer(hour_to_points_xyf={})

    transformer = hrrr_transformer_to_xy()
    x, y = transformer.transform(df["longitude"].to_numpy(), df["latitude"].to_numpy())

    work = pd.DataFrame(
        {
            "hour_utc": pd.to_datetime(df["hour_utc"], utc=True),
            "x": x,
            "y": y,
            "frp": pd.to_numeric(df["frp"], errors="coerce"),
        }
    ).dropna(subset=["x", "y", "frp"])

    hour_to_points: dict[datetime, np.ndarray] = {}
    for hour_utc, grp in work.groupby("hour_utc", sort=False):
        key = hour_utc.to_pydatetime().astimezone(timezone.utc)
        hour_to_points[key] = grp[["x", "y", "frp"]].to_numpy(dtype=np.float32)

    LOGGER.info("VIIRS hourly FRP groups prepared: %d hours", len(hour_to_points))
    return VIIRSFRPRasterizer(hour_to_points_xyf=hour_to_points)
