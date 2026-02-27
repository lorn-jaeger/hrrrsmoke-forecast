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
        self._coord_cache: OrderedDict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._coord_cache_entries = max(16, self.max_cache_entries)
        self._missing_field_cache: set[tuple[str, str, str]] = set()
        self._warned_missing_field_cache: set[tuple[str, str, str]] = set()
        self._field_exists_cache: dict[tuple[str, str, str], bool] = {}
        self._run_availability_cache: dict[str, bool] = {}
        self._cache_lock = Lock()
        self.cache_hits = 0
        self.cache_misses = 0
        self.missing_field_skips = 0
        self.missing_run_skips = 0

    @staticmethod
    def _cache_key(run_time_utc: datetime, variable: str, level: str) -> tuple[str, str, str]:
        if run_time_utc.tzinfo is None:
            run_time_utc = run_time_utc.replace(tzinfo=timezone.utc)
        run_time_utc = run_time_utc.astimezone(timezone.utc)
        return (run_time_utc.strftime("%Y%m%d%H"), variable, level)

    @staticmethod
    def _run_key(run_time_utc: datetime) -> str:
        if run_time_utc.tzinfo is None:
            run_time_utc = run_time_utc.replace(tzinfo=timezone.utc)
        run_time_utc = run_time_utc.astimezone(timezone.utc)
        return run_time_utc.strftime("%Y%m%d%H")

    @staticmethod
    def _coord_key(run_time_utc: datetime, level: str) -> tuple[str, str]:
        if run_time_utc.tzinfo is None:
            run_time_utc = run_time_utc.replace(tzinfo=timezone.utc)
        run_time_utc = run_time_utc.astimezone(timezone.utc)
        return (run_time_utc.strftime("%Y%m%d%H"), level)

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

    def _run_root_store_path(self, run_time_utc: datetime) -> str:
        if run_time_utc.tzinfo is None:
            run_time_utc = run_time_utc.replace(tzinfo=timezone.utc)
        run_time_utc = run_time_utc.astimezone(timezone.utc)
        ymd = run_time_utc.strftime("%Y%m%d")
        hh = run_time_utc.strftime("%H")
        return f"s3://{self.config.bucket}/{self.config.product}/{ymd}/{ymd}_{hh}z_anl.zarr"

    def _is_run_available(self, run_time_utc: datetime) -> bool:
        run_key = self._run_key(run_time_utc)
        with self._cache_lock:
            cached = self._run_availability_cache.get(run_key)
        if cached is not None:
            return bool(cached)

        try:
            exists = bool(self.fs.exists(self._run_root_store_path(run_time_utc)))
        except Exception as exc:
            # Treat errors as unavailable to avoid repeated slow missing-field probes.
            LOGGER.debug("Run-root availability check failed at %s (%s)", run_time_utc.isoformat(), exc)
            exists = False

        with self._cache_lock:
            self._run_availability_cache[run_key] = exists
        return exists

    def run_available(self, run_time_utc: datetime) -> bool:
        return self._is_run_available(run_time_utc)

    def _mark_missing_field(self, cache_key: tuple[str, str, str], spec: VariableSpec, run_time_utc: datetime, reason: str) -> None:
        with self._cache_lock:
            self._missing_field_cache.add(cache_key)
            first_warn = cache_key not in self._warned_missing_field_cache
            if first_warn:
                self._warned_missing_field_cache.add(cache_key)
        if first_warn:
            LOGGER.warning(
                "Failed to open HRRR field %s/%s at %s (%s)",
                spec.variable,
                spec.level,
                run_time_utc.isoformat(),
                reason,
            )
        else:
            LOGGER.debug(
                "Skipping previously missing HRRR field %s/%s at %s",
                spec.variable,
                spec.level,
                run_time_utc.isoformat(),
            )

    def load_field(self, run_time_utc: datetime, spec: VariableSpec) -> xr.DataArray | None:
        if run_time_utc.tzinfo is None:
            run_time_utc = run_time_utc.replace(tzinfo=timezone.utc)
        run_time_utc = run_time_utc.astimezone(timezone.utc)

        if not self._is_run_available(run_time_utc):
            self.missing_run_skips += 1
            return None

        cache_key = self._cache_key(run_time_utc, spec.variable, spec.level)
        with self._cache_lock:
            if cache_key in self._missing_field_cache:
                self.missing_field_skips += 1
                return None
            cached_exists = self._field_exists_cache.get(cache_key)
            if self.max_cache_entries > 0:
                cached = self._field_cache.get(cache_key)
                if cached is not None:
                    self.cache_hits += 1
                    self._field_cache.move_to_end(cache_key)
                    return cached
                self.cache_misses += 1

        store_path = self._field_store_path(run_time_utc, spec.variable, spec.level)
        if cached_exists is None:
            try:
                field_exists = bool(self.fs.exists(store_path))
            except Exception:
                field_exists = True
            with self._cache_lock:
                self._field_exists_cache[cache_key] = field_exists
        else:
            field_exists = bool(cached_exists)
        if not field_exists:
            self._mark_missing_field(cache_key, spec, run_time_utc, "field path does not exist")
            return None

        mapper = s3fs.S3Map(store_path, s3=self.fs, check=False)

        try:
            ds = xr.open_zarr(mapper, consolidated=False, decode_timedelta=False)
        except Exception as exc:
            exc_text = str(exc)
            missing_like = "group not found at path" in exc_text or "NoSuchKey" in exc_text
            if missing_like:
                self._mark_missing_field(cache_key, spec, run_time_utc, exc_text)
            else:
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
            with self._cache_lock:
                self._missing_field_cache.add(cache_key)
            return None

        # Analysis files should be a single time, but reduce defensively.
        for dim in ("time", "reference_time", "step"):
            if dim in da.dims:
                da = da.isel({dim: 0})

        # Pull x/y coordinates from metadata group when data subgroup is coordinate-less.
        need_x = "projection_x_coordinate" in da.dims and "projection_x_coordinate" not in da.coords
        need_y = "projection_y_coordinate" in da.dims and "projection_y_coordinate" not in da.coords
        if need_x or need_y:
            coord_key = self._coord_key(run_time_utc, spec.level)
            with self._cache_lock:
                coord_cached = self._coord_cache.get(coord_key)
                if coord_cached is not None:
                    self._coord_cache.move_to_end(coord_key)

            if coord_cached is None:
                meta_mapper = s3fs.S3Map(
                    self._field_meta_store_path(run_time_utc, spec.variable, spec.level),
                    s3=self.fs,
                    check=False,
                )
                try:
                    meta_ds = xr.open_zarr(meta_mapper, consolidated=False, decode_timedelta=False)
                    x_coord = (
                        meta_ds.coords["projection_x_coordinate"].values
                        if "projection_x_coordinate" in meta_ds.coords
                        else np.array([], dtype=np.float32)
                    )
                    y_coord = (
                        meta_ds.coords["projection_y_coordinate"].values
                        if "projection_y_coordinate" in meta_ds.coords
                        else np.array([], dtype=np.float32)
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "Failed to open HRRR coordinate metadata for %s/%s at %s (%s)",
                        spec.variable,
                        spec.level,
                        run_time_utc.isoformat(),
                        exc,
                    )
                    x_coord = np.array([], dtype=np.float32)
                    y_coord = np.array([], dtype=np.float32)

                coord_cached = (x_coord, y_coord)
                with self._cache_lock:
                    self._coord_cache[coord_key] = coord_cached
                    self._coord_cache.move_to_end(coord_key)
                    while len(self._coord_cache) > self._coord_cache_entries:
                        self._coord_cache.popitem(last=False)

            x_coord, y_coord = coord_cached
            if need_x and x_coord.size > 0:
                da = da.assign_coords(projection_x_coordinate=x_coord)
            if need_y and y_coord.size > 0:
                da = da.assign_coords(projection_y_coordinate=y_coord)

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
            with self._cache_lock:
                self._missing_field_cache.add(cache_key)
            return None

        out = da.astype(np.float32)
        if self.max_cache_entries > 0:
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
