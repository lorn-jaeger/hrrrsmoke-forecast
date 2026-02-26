from __future__ import annotations

import hashlib
import json
import logging
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from threading import local
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder

from gribcheck.config import PipelineConfig
from gribcheck.geo_utils import hrrr_transformer_to_xy
from gribcheck.hrrr import HRRRAnalysisReader

LOGGER = logging.getLogger(__name__)
_THREAD_LOCAL = local()


@dataclass(frozen=True)
class StationDailyStats:
    station_count: int
    hour_count_attempted: int
    hour_count_processed_total: int
    joined_rows: int
    checkpoint_dir: str


@dataclass(frozen=True)
class CheckpointPaths:
    root: Path
    metadata_json: Path
    progress_json: Path
    sum0_bin: Path
    sum1_bin: Path
    count_bin: Path
    processed_mask_bin: Path


def _default_workers(workers: int | None) -> int:
    if workers is not None:
        return max(1, int(workers))
    cpu_count = os.cpu_count() or 2
    return max(1, min(8, cpu_count))


def _default_checkpoint_dir(config: PipelineConfig, station_limit: int | None) -> Path:
    suffix = f"_stations_{int(station_limit)}" if station_limit is not None else ""
    return config.paths.intermediate_dir / f"station_daily_checkpoint{suffix}"


def _checkpoint_paths(checkpoint_dir: Path) -> CheckpointPaths:
    return CheckpointPaths(
        root=checkpoint_dir,
        metadata_json=checkpoint_dir / "metadata.json",
        progress_json=checkpoint_dir / "progress.json",
        sum0_bin=checkpoint_dir / "sum0.f32.bin",
        sum1_bin=checkpoint_dir / "sum1.f32.bin",
        count_bin=checkpoint_dir / "count.u16.bin",
        processed_mask_bin=checkpoint_dir / "processed_mask.u8.bin",
    )


def _expected_hours_for_local_day(local_day: date, tz_name: str) -> int:
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


def _utc_hour_range(start_date: date, end_date: date) -> pd.DatetimeIndex:
    start_utc = datetime.combine(start_date, time(0, 0), tzinfo=timezone.utc) - timedelta(days=1)
    end_utc = datetime.combine(end_date, time(23, 0), tzinfo=timezone.utc) + timedelta(days=1)
    return pd.date_range(start=start_utc, end=end_utc, freq="1h", tz="UTC")


def _thread_reader(config: PipelineConfig) -> HRRRAnalysisReader:
    reader = getattr(_THREAD_LOCAL, "hrrr_reader", None)
    if reader is None:
        reader = HRRRAnalysisReader(config.hrrr)
        _THREAD_LOCAL.hrrr_reader = reader
    return reader


def _sample_two_fields(
    reader: HRRRAnalysisReader,
    run_time_utc: datetime,
    variable_specs,
    x_points: np.ndarray,
    y_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    sampled_vars: list[np.ndarray] = []
    for spec in variable_specs:
        da = reader.load_field(run_time_utc, spec)
        if da is None:
            return None
        sampled = reader.bilinear_sample(da, x_points=x_points, y_points=y_points)
        sampled_vars.append(sampled)
    return sampled_vars[0], sampled_vars[1]


def _sample_two_fields_threaded(
    config: PipelineConfig,
    run_time_utc: datetime,
    variable_specs,
    x_points: np.ndarray,
    y_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    reader = _thread_reader(config)
    return _sample_two_fields(
        reader=reader,
        run_time_utc=run_time_utc,
        variable_specs=variable_specs,
        x_points=x_points,
        y_points=y_points,
    )


def _build_station_day_lookup(pm_df: pd.DataFrame) -> dict[str, dict[date, int]]:
    lookup: dict[str, dict[date, int]] = {}
    duplicate = 0
    for row_idx, (station_id, local_day) in enumerate(
        zip(pm_df["station_id"].to_numpy(), pm_df["date_local"].to_numpy())
    ):
        by_day = lookup.setdefault(str(station_id), {})
        if local_day in by_day:
            duplicate += 1
            continue
        by_day[local_day] = row_idx

    if duplicate > 0:
        raise ValueError(
            f"PM table has {duplicate} duplicate station/day keys; expected deduplicated site-day rows"
        )
    return lookup


def _accumulate_hour_arrays(
    sum0_arr: np.memmap,
    sum1_arr: np.memmap,
    count_arr: np.memmap,
    station_ids: np.ndarray,
    station_day_lookup: dict[str, dict[date, int]],
    by_tz: dict[str, np.ndarray],
    tz_cache: dict[str, ZoneInfo],
    utc_hour: datetime,
    sampled_0: np.ndarray,
    sampled_1: np.ndarray,
) -> None:
    valid_mask = np.isfinite(sampled_0) & np.isfinite(sampled_1)

    for tz_name, idxs in by_tz.items():
        if len(idxs) == 0:
            continue
        local_day = utc_hour.astimezone(tz_cache[tz_name]).date()
        valid_idxs = idxs[valid_mask[idxs]]
        for station_idx in valid_idxs:
            station_id = str(station_ids[station_idx])
            row_idx = station_day_lookup.get(station_id, {}).get(local_day)
            if row_idx is None:
                continue
            sum0_arr[row_idx] += float(sampled_0[station_idx])
            sum1_arr[row_idx] += float(sampled_1[station_idx])
            count_arr[row_idx] = np.uint16(int(count_arr[row_idx]) + 1)


def _atomic_write_parquet(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(output_path)


def _prepare_inputs(
    config: PipelineConfig,
    station_limit: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[date, int]]]:
    pm_df = pd.read_parquet(config.paths.pm_output)
    if pm_df.empty:
        raise ValueError("PM dataset is empty; run ingest-pm first")

    pm_df["date_local"] = pd.to_datetime(pm_df["date_local"]).dt.date

    stations = (
        pm_df[["station_id", "latitude", "longitude", "state_name"]]
        .drop_duplicates(subset=["station_id"])
        .sort_values("station_id")
        .reset_index(drop=True)
    )
    if station_limit is not None:
        stations = stations.head(int(station_limit)).reset_index(drop=True)
        pm_df = pm_df[pm_df["station_id"].isin(stations["station_id"])].copy()

    stations["timezone"] = _resolve_station_timezones(stations)

    pm_df = pm_df.sort_values(["date_local", "station_id"]).reset_index(drop=True)
    station_day_lookup = _build_station_day_lookup(pm_df)

    return pm_df, stations, station_day_lookup


def _build_joined_from_arrays(
    pm_df: pd.DataFrame,
    station_tz: pd.DataFrame,
    sum0_arr: np.ndarray,
    sum1_arr: np.ndarray,
    count_arr: np.ndarray,
    output_names: list[str],
) -> pd.DataFrame:
    joined = pm_df.copy()

    counts = np.asarray(count_arr, dtype=np.int32)
    sum0 = np.asarray(sum0_arr, dtype=np.float32)
    sum1 = np.asarray(sum1_arr, dtype=np.float32)

    mean0 = np.full(len(joined), np.nan, dtype=np.float32)
    mean1 = np.full(len(joined), np.nan, dtype=np.float32)
    valid = counts > 0
    np.divide(sum0, counts, out=mean0, where=valid)
    np.divide(sum1, counts, out=mean1, where=valid)

    joined[output_names[0]] = mean0
    joined[output_names[1]] = mean1
    joined["hrrr_hours_found"] = counts

    tz_map = dict(zip(station_tz["station_id"].to_numpy(), station_tz["timezone"].to_numpy()))
    joined["timezone"] = joined["station_id"].map(tz_map).fillna("UTC")

    expected_lookup_rows = joined[["date_local", "timezone"]].drop_duplicates().reset_index(drop=True)
    expected_lookup_rows["hrrr_hours_expected"] = expected_lookup_rows.apply(
        lambda row: _expected_hours_for_local_day(row["date_local"], row["timezone"]),
        axis=1,
    )

    joined = joined.merge(expected_lookup_rows, on=["date_local", "timezone"], how="left")
    joined["hrrr_hours_expected"] = joined["hrrr_hours_expected"].fillna(24).astype(int)
    joined["hrrr_day_complete"] = joined["hrrr_hours_found"] == joined["hrrr_hours_expected"]

    return joined


def _station_ids_hash(stations: pd.DataFrame) -> str:
    text = "\n".join(stations["station_id"].astype(str).tolist())
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _meta_matches(existing: dict, expected: dict) -> bool:
    keys = [
        "run_start_date",
        "run_end_date",
        "pm_rows",
        "station_count",
        "total_hours",
        "station_ids_hash",
        "station_limit",
    ]
    return all(existing.get(k) == expected.get(k) for k in keys)


def _initialize_or_load_checkpoint(
    paths: CheckpointPaths,
    *,
    expected_meta: dict,
    row_count: int,
    total_hours: int,
    resume: bool,
) -> tuple[np.memmap, np.memmap, np.memmap, np.memmap]:
    paths.root.mkdir(parents=True, exist_ok=True)

    have_all = all(
        p.exists()
        for p in [paths.metadata_json, paths.sum0_bin, paths.sum1_bin, paths.count_bin, paths.processed_mask_bin]
    )

    should_reset = True
    if resume and have_all:
        existing_meta = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
        if _meta_matches(existing_meta, expected_meta):
            should_reset = False
            LOGGER.info("Resuming station checkpoint from %s", paths.root)
        else:
            LOGGER.warning("Checkpoint metadata mismatch; resetting checkpoint at %s", paths.root)

    if should_reset:
        for p in [paths.sum0_bin, paths.sum1_bin, paths.count_bin, paths.processed_mask_bin]:
            if p.exists():
                p.unlink()
        sum0 = np.memmap(paths.sum0_bin, mode="w+", dtype=np.float32, shape=(row_count,))
        sum1 = np.memmap(paths.sum1_bin, mode="w+", dtype=np.float32, shape=(row_count,))
        count = np.memmap(paths.count_bin, mode="w+", dtype=np.uint16, shape=(row_count,))
        processed_mask = np.memmap(paths.processed_mask_bin, mode="w+", dtype=np.uint8, shape=(total_hours,))
        sum0[:] = 0.0
        sum1[:] = 0.0
        count[:] = 0
        processed_mask[:] = 0
        sum0.flush()
        sum1.flush()
        count.flush()
        processed_mask.flush()
        paths.metadata_json.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
        LOGGER.info("Initialized station checkpoint at %s", paths.root)
    else:
        sum0 = np.memmap(paths.sum0_bin, mode="r+", dtype=np.float32, shape=(row_count,))
        sum1 = np.memmap(paths.sum1_bin, mode="r+", dtype=np.float32, shape=(row_count,))
        count = np.memmap(paths.count_bin, mode="r+", dtype=np.uint16, shape=(row_count,))
        processed_mask = np.memmap(paths.processed_mask_bin, mode="r+", dtype=np.uint8, shape=(total_hours,))

    return sum0, sum1, count, processed_mask


def _write_progress(
    progress_path: Path,
    *,
    total_hours: int,
    target_hours: int,
    processed_total: int,
    processed_target: int,
    last_hour_utc: datetime | None,
    output_path: Path,
    checkpoint_dir: Path,
) -> None:
    payload = {
        "total_hours": int(total_hours),
        "target_hours": int(target_hours),
        "processed_total": int(processed_total),
        "processed_target": int(processed_target),
        "processed_fraction_target": float(processed_target / target_hours) if target_hours > 0 else 0.0,
        "last_hour_utc": last_hour_utc.isoformat() if last_hour_utc is not None else None,
        "output_path": str(output_path),
        "checkpoint_dir": str(checkpoint_dir),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def materialize_checkpoint_snapshot(
    config: PipelineConfig,
    station_limit: int | None = None,
    max_hours: int | None = None,
    checkpoint_dir: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    pm_df, stations, _lookup = _prepare_inputs(config, station_limit=station_limit)
    station_tz = stations[["station_id", "timezone"]].copy()

    all_hours = _utc_hour_range(config.run.start_date, config.run.end_date)
    total_hours = len(all_hours)
    target_hours = min(int(max_hours), total_hours) if max_hours is not None else total_hours

    checkpoint_root = checkpoint_dir or _default_checkpoint_dir(config, station_limit=station_limit)
    paths = _checkpoint_paths(checkpoint_root)

    if not all(p.exists() for p in [paths.sum0_bin, paths.sum1_bin, paths.count_bin, paths.processed_mask_bin]):
        raise ValueError(f"Checkpoint files not found at {checkpoint_root}")

    row_count = len(pm_df)
    sum0 = np.memmap(paths.sum0_bin, mode="r", dtype=np.float32, shape=(row_count,))
    sum1 = np.memmap(paths.sum1_bin, mode="r", dtype=np.float32, shape=(row_count,))
    count = np.memmap(paths.count_bin, mode="r", dtype=np.uint16, shape=(row_count,))
    processed_mask = np.memmap(paths.processed_mask_bin, mode="r", dtype=np.uint8, shape=(total_hours,))

    variable_specs = list(config.hrrr.station_variables)
    output_names = [spec.output_column or spec.variable.lower() for spec in variable_specs]
    joined = _build_joined_from_arrays(pm_df, station_tz, sum0, sum1, count, output_names=output_names)

    final_output = output_path
    if final_output is None:
        final_output = config.paths.station_daily_output.with_name(
            config.paths.station_daily_output.stem + "_partial.parquet"
        )

    _atomic_write_parquet(joined, final_output)

    _write_progress(
        paths.progress_json,
        total_hours=total_hours,
        target_hours=target_hours,
        processed_total=int(np.sum(np.asarray(processed_mask, dtype=np.uint8))),
        processed_target=int(np.sum(np.asarray(processed_mask[:target_hours], dtype=np.uint8))),
        last_hour_utc=None,
        output_path=final_output,
        checkpoint_dir=checkpoint_root,
    )

    LOGGER.info("Wrote checkpoint snapshot to %s", final_output)
    return final_output


def run(
    config: PipelineConfig,
    max_hours: int | None = None,
    station_limit: int | None = None,
    workers: int | None = None,
    resume: bool = True,
    checkpoint_dir: Path | None = None,
    checkpoint_flush_hours: int = 1,
) -> StationDailyStats:
    pm_df, stations, station_day_lookup = _prepare_inputs(config, station_limit=station_limit)

    transformer = hrrr_transformer_to_xy()
    x_vals, y_vals = transformer.transform(stations["longitude"].to_numpy(), stations["latitude"].to_numpy())
    stations["x"] = x_vals
    stations["y"] = y_vals

    all_utc_hours = _utc_hour_range(config.run.start_date, config.run.end_date)
    total_hours = len(all_utc_hours)
    target_hours = min(int(max_hours), total_hours) if max_hours is not None else total_hours
    worker_count = _default_workers(workers)

    by_tz: dict[str, np.ndarray] = {}
    tz_arr = stations["timezone"].to_numpy()
    for tz_name in sorted(set(tz_arr)):
        by_tz[tz_name] = np.where(tz_arr == tz_name)[0]
    tz_cache = {tz_name: ZoneInfo(tz_name) for tz_name in by_tz}

    station_ids = stations["station_id"].to_numpy()
    x_points = stations["x"].to_numpy(dtype=np.float64)
    y_points = stations["y"].to_numpy(dtype=np.float64)

    variable_specs = list(config.hrrr.station_variables)
    if len(variable_specs) != 2:
        raise ValueError("Station daily pipeline expects exactly 2 HRRR station variables")

    output_names = [spec.output_column or spec.variable.lower() for spec in variable_specs]
    checkpoint_root = checkpoint_dir or _default_checkpoint_dir(config, station_limit=station_limit)
    paths = _checkpoint_paths(checkpoint_root)

    expected_meta = {
        "run_start_date": config.run.start_date.isoformat(),
        "run_end_date": config.run.end_date.isoformat(),
        "pm_rows": int(len(pm_df)),
        "station_count": int(len(stations)),
        "total_hours": int(total_hours),
        "station_ids_hash": _station_ids_hash(stations),
        "station_limit": int(station_limit) if station_limit is not None else None,
        "created_by": "gribcheck.station_daily",
    }

    sum0_arr, sum1_arr, count_arr, processed_mask = _initialize_or_load_checkpoint(
        paths,
        expected_meta=expected_meta,
        row_count=len(pm_df),
        total_hours=total_hours,
        resume=resume,
    )

    processed_total = int(np.sum(np.asarray(processed_mask, dtype=np.uint8)))
    processed_target = int(np.sum(np.asarray(processed_mask[:target_hours], dtype=np.uint8)))

    LOGGER.info(
        "Sampling %d/%d UTC hours for %d stations using %d worker(s); already processed target=%d",
        target_hours,
        total_hours,
        len(stations),
        worker_count,
        processed_target,
    )

    remaining: list[tuple[int, pd.Timestamp]] = [
        (idx, all_utc_hours[idx]) for idx in range(target_hours) if int(processed_mask[idx]) == 0
    ]

    flush_every = max(1, int(checkpoint_flush_hours))
    processed_since_flush = 0
    last_completed_hour: datetime | None = None

    if worker_count == 1:
        reader = HRRRAnalysisReader(config.hrrr)
        for hour_idx, utc_hour in remaining:
            sampled = _sample_two_fields(
                reader=reader,
                run_time_utc=utc_hour.to_pydatetime(),
                variable_specs=variable_specs,
                x_points=x_points,
                y_points=y_points,
            )
            if sampled is not None:
                _accumulate_hour_arrays(
                    sum0_arr=sum0_arr,
                    sum1_arr=sum1_arr,
                    count_arr=count_arr,
                    station_ids=station_ids,
                    station_day_lookup=station_day_lookup,
                    by_tz=by_tz,
                    tz_cache=tz_cache,
                    utc_hour=utc_hour.to_pydatetime(),
                    sampled_0=sampled[0],
                    sampled_1=sampled[1],
                )

            processed_mask[hour_idx] = 1
            processed_total += 1
            processed_target += 1
            processed_since_flush += 1
            last_completed_hour = utc_hour.to_pydatetime()
            LOGGER.info("Processed hour %d / %d (%s)", processed_target, target_hours, utc_hour)

            if processed_since_flush >= flush_every:
                sum0_arr.flush()
                sum1_arr.flush()
                count_arr.flush()
                processed_mask.flush()
                _write_progress(
                    paths.progress_json,
                    total_hours=total_hours,
                    target_hours=target_hours,
                    processed_total=processed_total,
                    processed_target=processed_target,
                    last_hour_utc=last_completed_hour,
                    output_path=config.paths.station_daily_output,
                    checkpoint_dir=checkpoint_root,
                )
                processed_since_flush = 0
    else:
        max_in_flight = max(worker_count * 4, worker_count)
        submitted = 0
        in_flight = {}

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            while submitted < len(remaining) and len(in_flight) < max_in_flight:
                hour_idx, utc_hour = remaining[submitted]
                fut = pool.submit(
                    _sample_two_fields_threaded,
                    config,
                    utc_hour.to_pydatetime(),
                    variable_specs,
                    x_points,
                    y_points,
                )
                in_flight[fut] = (hour_idx, utc_hour)
                submitted += 1

            while in_flight:
                done, _pending = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    hour_idx, utc_hour = in_flight.pop(fut)
                    sampled = fut.result()
                    if sampled is not None:
                        _accumulate_hour_arrays(
                            sum0_arr=sum0_arr,
                            sum1_arr=sum1_arr,
                            count_arr=count_arr,
                            station_ids=station_ids,
                            station_day_lookup=station_day_lookup,
                            by_tz=by_tz,
                            tz_cache=tz_cache,
                            utc_hour=utc_hour.to_pydatetime(),
                            sampled_0=sampled[0],
                            sampled_1=sampled[1],
                        )

                    processed_mask[hour_idx] = 1
                    processed_total += 1
                    processed_target += 1
                    processed_since_flush += 1
                    last_completed_hour = utc_hour.to_pydatetime()
                    LOGGER.info("Processed hour %d / %d (%s)", processed_target, target_hours, utc_hour)

                    if processed_since_flush >= flush_every:
                        sum0_arr.flush()
                        sum1_arr.flush()
                        count_arr.flush()
                        processed_mask.flush()
                        _write_progress(
                            paths.progress_json,
                            total_hours=total_hours,
                            target_hours=target_hours,
                            processed_total=processed_total,
                            processed_target=processed_target,
                            last_hour_utc=last_completed_hour,
                            output_path=config.paths.station_daily_output,
                            checkpoint_dir=checkpoint_root,
                        )
                        processed_since_flush = 0

                    if submitted < len(remaining):
                        next_hour_idx, next_hour = remaining[submitted]
                        next_fut = pool.submit(
                            _sample_two_fields_threaded,
                            config,
                            next_hour.to_pydatetime(),
                            variable_specs,
                            x_points,
                            y_points,
                        )
                        in_flight[next_fut] = (next_hour_idx, next_hour)
                        submitted += 1

    sum0_arr.flush()
    sum1_arr.flush()
    count_arr.flush()
    processed_mask.flush()

    station_tz = stations[["station_id", "timezone"]].copy()
    joined = _build_joined_from_arrays(pm_df, station_tz, sum0_arr, sum1_arr, count_arr, output_names)
    _atomic_write_parquet(joined, config.paths.station_daily_output)

    _write_progress(
        paths.progress_json,
        total_hours=total_hours,
        target_hours=target_hours,
        processed_total=processed_total,
        processed_target=processed_target,
        last_hour_utc=last_completed_hour,
        output_path=config.paths.station_daily_output,
        checkpoint_dir=checkpoint_root,
    )

    LOGGER.info(
        "Station daily join snapshot written: %d rows (processed target=%d/%d)",
        len(joined),
        processed_target,
        target_hours,
    )

    return StationDailyStats(
        station_count=len(stations),
        hour_count_attempted=target_hours,
        hour_count_processed_total=processed_target,
        joined_rows=len(joined),
        checkpoint_dir=str(checkpoint_root),
    )
