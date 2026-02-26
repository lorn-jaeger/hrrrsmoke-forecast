from __future__ import annotations

from collections import OrderedDict
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock

import numpy as np
import s3fs
import xarray as xr

from gribcheck.config import HRRRConfig
from gribcheck.models import VariableSpec

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HRRRFieldKey:
    run_time_utc: datetime
    variable: str
    level: str


class HRRRAnalysisReader:
    def __init__(self, config: HRRRConfig, max_cache_entries: int = 0):
        self.config = config
        self.fs = s3fs.S3FileSystem(anon=config.anonymous)
        self.max_cache_entries = max(0, int(max_cache_entries))
        self._field_cache: OrderedDict[tuple[str, str, str], xr.DataArray] = OrderedDict()
        self._cache_lock = Lock()
        self.cache_hits = 0
        self.cache_misses = 0

    @staticmethod
    def _cache_key(run_time_utc: datetime, variable: str, level: str) -> tuple[str, str, str]:
        if run_time_utc.tzinfo is None:
            run_time_utc = run_time_utc.replace(tzinfo=timezone.utc)
        run_time_utc = run_time_utc.astimezone(timezone.utc)
        return (run_time_utc.strftime("%Y%m%d%H"), variable, level)

    def _field_store_path(self, run_time_utc: datetime, variable: str, level: str) -> str:
        if run_time_utc.tzinfo is None:
            run_time_utc = run_time_utc.replace(tzinfo=timezone.utc)
        run_time_utc = run_time_utc.astimezone(timezone.utc)
        ymd = run_time_utc.strftime("%Y%m%d")
        hh = run_time_utc.strftime("%H")
        return (
            f"s3://{self.config.bucket}/{self.config.product}/{ymd}/{ymd}_{hh}z_anl.zarr/"
            f"{level}/{variable}/{level}"
        )

    def _field_meta_store_path(self, run_time_utc: datetime, variable: str, level: str) -> str:
        if run_time_utc.tzinfo is None:
            run_time_utc = run_time_utc.replace(tzinfo=timezone.utc)
        run_time_utc = run_time_utc.astimezone(timezone.utc)
        ymd = run_time_utc.strftime("%Y%m%d")
        hh = run_time_utc.strftime("%H")
        return (
            f"s3://{self.config.bucket}/{self.config.product}/{ymd}/{ymd}_{hh}z_anl.zarr/"
            f"{level}/{variable}"
        )

    def load_field(self, run_time_utc: datetime, spec: VariableSpec) -> xr.DataArray | None:
        cache_key = self._cache_key(run_time_utc, spec.variable, spec.level)
        if self.max_cache_entries > 0:
            with self._cache_lock:
                cached = self._field_cache.get(cache_key)
                if cached is not None:
                    self.cache_hits += 1
                    self._field_cache.move_to_end(cache_key)
                    return cached
                self.cache_misses += 1

        store_path = self._field_store_path(run_time_utc, spec.variable, spec.level)
        mapper = s3fs.S3Map(store_path, s3=self.fs, check=False)
        meta_mapper = s3fs.S3Map(
            self._field_meta_store_path(run_time_utc, spec.variable, spec.level),
            s3=self.fs,
            check=False,
        )

        try:
            ds = xr.open_zarr(mapper, consolidated=False, decode_timedelta=False)
            meta_ds = xr.open_zarr(meta_mapper, consolidated=False, decode_timedelta=False)
        except Exception as exc:
            LOGGER.warning(
                "Failed to open HRRR field %s/%s at %s (%s)",
                spec.variable,
                spec.level,
                run_time_utc.isoformat(),
                exc,
            )
            return None

        if spec.variable in ds.data_vars:
            da = ds[spec.variable]
        elif len(ds.data_vars) == 1:
            da = next(iter(ds.data_vars.values()))
        else:
            LOGGER.warning(
                "Variable %s not found in dataset for %s; available vars=%s",
                spec.variable,
                run_time_utc,
                list(ds.data_vars),
            )
            return None

        # Analysis files should be a single time, but reduce defensively.
        for dim in ("time", "reference_time", "step"):
            if dim in da.dims:
                da = da.isel({dim: 0})

        # Pull x/y coordinates from metadata group when data subgroup is coordinate-less.
        if "projection_x_coordinate" in da.dims and "projection_x_coordinate" in meta_ds.coords:
            da = da.assign_coords(
                projection_x_coordinate=meta_ds.coords["projection_x_coordinate"].values
            )
        if "projection_y_coordinate" in da.dims and "projection_y_coordinate" in meta_ds.coords:
            da = da.assign_coords(
                projection_y_coordinate=meta_ds.coords["projection_y_coordinate"].values
            )

        # Ensure common dimension names.
        rename_dims: dict[str, str] = {}
        if "projection_x_coordinate" in da.dims:
            rename_dims["projection_x_coordinate"] = "x"
        if "projection_y_coordinate" in da.dims:
            rename_dims["projection_y_coordinate"] = "y"
        if rename_dims:
            da = da.rename(rename_dims)
        if "projection_x_coordinate" in da.coords and "x" not in da.coords:
            da = da.rename({"projection_x_coordinate": "x"})
        if "projection_y_coordinate" in da.coords and "y" not in da.coords:
            da = da.rename({"projection_y_coordinate": "y"})

        if "x" not in da.coords or "y" not in da.coords:
            LOGGER.warning("Field %s at %s missing x/y coordinates", spec.variable, run_time_utc)
            return None

        out = da.astype(np.float32)
        if self.max_cache_entries > 0:
            out = out.load()
            with self._cache_lock:
                self._field_cache[cache_key] = out
                self._field_cache.move_to_end(cache_key)
                while len(self._field_cache) > self.max_cache_entries:
                    self._field_cache.popitem(last=False)
        return out

    @staticmethod
    def bilinear_sample(
        field: xr.DataArray,
        x_points: np.ndarray,
        y_points: np.ndarray,
    ) -> np.ndarray:
        sampled = field.interp(
            x=xr.DataArray(x_points, dims="station"),
            y=xr.DataArray(y_points, dims="station"),
            method="linear",
        )
        return sampled.values.astype(np.float32, copy=False)
