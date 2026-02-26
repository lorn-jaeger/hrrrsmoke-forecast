from __future__ import annotations

import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import local

import numcodecs
import numpy as np
import pandas as pd
import zarr

from gribcheck.config import PipelineConfig
from gribcheck.fire import load_filtered_fire_records
from gribcheck.geo_utils import hrrr_transformer_to_xy, split_for_date
from gribcheck.hrrr import HRRRAnalysisReader
from gribcheck.models import FireRecord, VariableSpec
from gribcheck.viirs import build_viirs_rasterizer, load_or_build_viirs_hourly_points

LOGGER = logging.getLogger(__name__)
_THREAD_LOCAL = local()


@dataclass(frozen=True)
class WildfireRasterStats:
    fire_count: int
    sample_count: int
    channels_used: int
    estimated_bytes: int


@dataclass
class ReductionPlan:
    include_upper_air: bool
    cadence_after_72h: int
    cap_samples_per_fire: int | None
    actions: list[str]


def _default_workers(workers: int | None) -> int:
    if workers is not None:
        return max(1, int(workers))
    cpu_count = os.cpu_count() or 2
    return max(1, min(8, cpu_count))


def _floor_to_hour(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _hours_in_fire_window(
    rec: FireRecord,
    cadence_after_72h: int,
    cap: int | None,
    sample_hours_utc: set[int] | None = None,
) -> int:
    start = _floor_to_hour(rec.start_time_utc)
    end = _floor_to_hour(rec.end_time_utc)
    if end < start:
        return 0

    total_hours = int((end - start).total_seconds() // 3600) + 1
    count = 0
    for i in range(total_hours):
        if i > 72 and cadence_after_72h == 2 and (i % 2 == 1):
            continue
        hour_utc = (start.hour + i) % 24
        if sample_hours_utc is not None and hour_utc not in sample_hours_utc:
            continue
        count += 1
        if cap is not None and count >= cap:
            break
    return count


def _estimate_total_samples(
    records: list[FireRecord],
    cadence_after_72h: int,
    cap: int | None,
    sample_hours_utc: set[int] | None = None,
) -> int:
    return int(
        sum(
            _hours_in_fire_window(
                r,
                cadence_after_72h,
                cap,
                sample_hours_utc=sample_hours_utc,
            )
            for r in records
        )
    )


def _estimate_dataset_bytes(sample_count: int, channel_count: int, label_count: int, patch_size: tuple[int, int]) -> int:
    h, w = patch_size
    values_per_sample = (channel_count + label_count) * h * w
    bytes_per_sample = values_per_sample * 2  # float16
    return int(sample_count * bytes_per_sample)


def _is_upper_air_or_dzdt(spec: VariableSpec) -> bool:
    level = spec.level.lower()
    if level.endswith("mb"):
        return True
    if spec.variable.upper() == "DZDT":
        return True
    return False


def _build_reduction_plan(
    config: PipelineConfig,
    records: list[FireRecord],
    label_count: int,
    sample_hours_utc: set[int] | None = None,
) -> tuple[ReductionPlan, int, int]:
    active_specs = list(config.wildfire.analysis_variables)

    include_upper_air = True
    cadence_after_72h = 1
    cap = None
    actions: list[str] = []

    def current_specs() -> list[VariableSpec]:
        if include_upper_air:
            return list(active_specs)
        return [spec for spec in active_specs if not _is_upper_air_or_dzdt(spec)]

    budget_bytes = int(config.storage.budget_gb * (1024**3))

    while True:
        sample_count = _estimate_total_samples(
            records,
            cadence_after_72h=cadence_after_72h,
            cap=cap,
            sample_hours_utc=sample_hours_utc,
        )
        # +1 channel for VIIRS FRP rasterized feature.
        ch_count = len(current_specs()) + 1
        est_bytes = _estimate_dataset_bytes(
            sample_count=sample_count,
            channel_count=ch_count,
            label_count=label_count,
            patch_size=config.wildfire.patch_size,
        )

        if est_bytes <= budget_bytes:
            plan = ReductionPlan(
                include_upper_air=include_upper_air,
                cadence_after_72h=cadence_after_72h,
                cap_samples_per_fire=cap,
                actions=actions,
            )
            return plan, sample_count, est_bytes

        if include_upper_air and config.storage.drop_upper_air_and_dzdt:
            include_upper_air = False
            actions.append("drop_upper_air_and_dzdt")
            continue

        if cadence_after_72h == 1 and config.storage.two_hour_cadence_after_72h:
            cadence_after_72h = 2
            actions.append("two_hour_cadence_after_72h")
            continue

        if cap is None:
            cap = config.storage.cap_samples_per_fire
            actions.append(f"cap_samples_per_fire={cap}")
            continue

        # No further reduction steps.
        plan = ReductionPlan(
            include_upper_air=include_upper_air,
            cadence_after_72h=cadence_after_72h,
            cap_samples_per_fire=cap,
            actions=actions,
        )
        return plan, sample_count, est_bytes


def _iter_fire_hours(
    rec: FireRecord,
    cadence_after_72h: int,
    cap: int | None,
    sample_hours_utc: set[int] | None = None,
    max_hours_per_fire: int | None = None,
):
    start = _floor_to_hour(rec.start_time_utc)
    end = _floor_to_hour(rec.end_time_utc)
    total_hours = int((end - start).total_seconds() // 3600) + 1

    emitted = 0
    for i in range(max(total_hours, 0)):
        if max_hours_per_fire is not None and i >= int(max_hours_per_fire):
            break
        if i > 72 and cadence_after_72h == 2 and (i % 2 == 1):
            continue
        ts = start + timedelta(hours=i)
        if sample_hours_utc is not None and ts.hour not in sample_hours_utc:
            continue
        yield ts
        emitted += 1
        if cap is not None and emitted >= cap:
            break


def _normalize_sample_hours(sample_hours_utc: tuple[int, ...] | None) -> set[int] | None:
    if sample_hours_utc is None:
        return None
    out: set[int] = set()
    for value in sample_hours_utc:
        hour = int(value)
        if hour < 0 or hour > 23:
            raise ValueError(f"sample hour must be in [0, 23], got {hour}")
        out.add(hour)
    if not out:
        raise ValueError("sample_hours_utc must not be empty when provided")
    return out


def _build_bounds_xy(rec: FireRecord, buffer_km: float):
    transformer = hrrr_transformer_to_xy()
    x0, y0 = transformer.transform(rec.min_lon, rec.min_lat)
    x1, y1 = transformer.transform(rec.max_lon, rec.max_lat)

    xmin = min(x0, x1)
    xmax = max(x0, x1)
    ymin = min(y0, y1)
    ymax = max(y0, y1)

    buffer_m = buffer_km * 1000.0
    return (xmin - buffer_m, ymin - buffer_m, xmax + buffer_m, ymax + buffer_m)


def _subset_to_bounds(field, bounds_xy):
    xmin, ymin, xmax, ymax = bounds_xy

    x_vals = field["x"].to_numpy()
    y_vals = field["y"].to_numpy()

    x_slice = slice(xmin, xmax) if x_vals[0] <= x_vals[-1] else slice(xmax, xmin)
    y_slice = slice(ymin, ymax) if y_vals[0] <= y_vals[-1] else slice(ymax, ymin)

    subset = field.sel(x=x_slice, y=y_slice)
    if subset.size == 0:
        return None
    return subset


def _resample_to_patch(field, patch_size: tuple[int, int]) -> np.ndarray | None:
    h, w = patch_size
    if field is None or field.size == 0:
        return None

    x_vals = field["x"].to_numpy()
    y_vals = field["y"].to_numpy()

    if len(x_vals) < 2 or len(y_vals) < 2:
        return None

    x_new = np.linspace(float(x_vals.min()), float(x_vals.max()), w)
    y_new = np.linspace(float(y_vals.min()), float(y_vals.max()), h)
    interp = field.interp(x=x_new, y=y_new, method="linear")
    arr = interp.to_numpy().astype(np.float32, copy=False)
    if arr.shape != (h, w):
        return None
    return arr


def _extract_patch(reader: HRRRAnalysisReader, run_time_utc: datetime, spec: VariableSpec, bounds_xy, patch_size):
    da = reader.load_field(run_time_utc, spec)
    if da is None:
        return None
    subset = _subset_to_bounds(da, bounds_xy)
    return _resample_to_patch(subset, patch_size=patch_size)


def _thread_reader(config: PipelineConfig) -> HRRRAnalysisReader:
    reader = getattr(_THREAD_LOCAL, "hrrr_reader", None)
    if reader is None:
        reader = HRRRAnalysisReader(config.hrrr)
        _THREAD_LOCAL.hrrr_reader = reader
    return reader


def _extract_patch_threaded(
    config: PipelineConfig,
    run_time_utc: datetime,
    spec: VariableSpec,
    bounds_xy,
    patch_size,
):
    reader = _thread_reader(config)
    return _extract_patch(reader, run_time_utc=run_time_utc, spec=spec, bounds_xy=bounds_xy, patch_size=patch_size)


def _create_or_open_zarr(path: Path, channels: list[str], leads: list[int], patch_size: tuple[int, int], dtype: str, compressor: str):
    if path.exists():
        # Start fresh each run for deterministic output.
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(path), mode="w")

    if compressor.lower() == "zstd":
        comp = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    else:
        comp = None

    h, w = patch_size

    inputs_group = group.create_group("inputs")
    for channel in channels:
        inputs_group.create_dataset(
            channel,
            shape=(0, h, w),
            chunks=(1, h, w),
            dtype=dtype,
            compressor=comp,
        )

    labels_group = group.create_group("labels")
    for lead in leads:
        labels_group.create_dataset(
            f"t_plus_{lead}h",
            shape=(0, h, w),
            chunks=(1, h, w),
            dtype=dtype,
            compressor=comp,
        )

    group.attrs["channels"] = channels
    group.attrs["label_leads_hours"] = leads
    group.attrs["patch_size"] = [h, w]

    return group


def _append_2d(arr: zarr.Array, value: np.ndarray) -> int:
    n = arr.shape[0]
    arr.resize((n + 1, arr.shape[1], arr.shape[2]))
    arr[n, :, :] = value
    return n


def _shift_patch(arr: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    shifted = np.roll(arr, shift=(shift_y, shift_x), axis=(0, 1))
    if shift_y > 0:
        shifted[:shift_y, :] = 0.0
    elif shift_y < 0:
        shifted[shift_y:, :] = 0.0
    if shift_x > 0:
        shifted[:, :shift_x] = 0.0
    elif shift_x < 0:
        shifted[:, shift_x:] = 0.0
    return shifted


def _advection_baseline(
    massden_patch: np.ndarray,
    ugrd_patch: np.ndarray,
    vgrd_patch: np.ndarray,
    lead_hours: int,
    grid_spacing_m: float = 3000.0,
) -> np.ndarray:
    # Translate near-surface smoke by mean 10m wind as a simple baseline.
    u_mean = float(np.nanmean(ugrd_patch))
    v_mean = float(np.nanmean(vgrd_patch))
    seconds = lead_hours * 3600.0

    dx_pixels = int(np.rint((u_mean * seconds) / grid_spacing_m))
    dy_pixels = int(np.rint((v_mean * seconds) / grid_spacing_m))
    return _shift_patch(massden_patch, shift_y=dy_pixels, shift_x=dx_pixels)


def _maybe_write_qa_tiff(path: Path, sample_id: int, channel_name: str, arr: np.ndarray) -> None:
    try:
        import tifffile
    except Exception:
        return
    tifffile.imwrite(path / f"sample_{sample_id:06d}_{channel_name}.tiff", arr.astype(np.float32), compression="deflate")


def run(
    config: PipelineConfig,
    max_fires: int | None = None,
    max_samples_total: int | None = None,
    max_hours_per_fire: int | None = None,
    workers: int | None = None,
    sample_hours_utc: tuple[int, ...] | None = None,
    next_day_only: bool = False,
) -> WildfireRasterStats:
    records = load_filtered_fire_records(config)
    if not records:
        raise ValueError("No wildfire records available after filtering")
    if max_fires is not None:
        records = records[: int(max_fires)]

    sample_hours = _normalize_sample_hours(sample_hours_utc)
    leads = [24] if next_day_only else sorted(config.wildfire.label_lead_hours)
    reduction_plan, estimated_samples, estimated_bytes = _build_reduction_plan(
        config,
        records,
        label_count=len(leads),
        sample_hours_utc=sample_hours,
    )

    active_specs = list(config.wildfire.analysis_variables)
    if not reduction_plan.include_upper_air:
        active_specs = [spec for spec in active_specs if not _is_upper_air_or_dzdt(spec)]

    hrrr_channels = [spec.channel_name or f"{spec.variable}_{spec.level}" for spec in active_specs]
    channels = [config.wildfire.frp_channel_name, *hrrr_channels]
    patch_size = tuple(config.wildfire.patch_size)
    worker_count = _default_workers(workers)

    reader = HRRRAnalysisReader(config.hrrr)
    viirs_df = load_or_build_viirs_hourly_points(config)
    viirs_rasterizer = build_viirs_rasterizer(viirs_df)

    zarr_group = _create_or_open_zarr(
        path=config.paths.wildfire_zarr_output,
        channels=channels,
        leads=leads,
        patch_size=patch_size,
        dtype=config.storage.dtype,
        compressor=config.storage.compressor,
    )

    index_rows: list[dict[str, object]] = []

    input_arrays = {channel: zarr_group[f"inputs/{channel}"] for channel in channels}
    label_arrays = {lead: zarr_group[f"labels/t_plus_{lead}h"] for lead in leads}

    sample_count = 0
    LOGGER.info(
        "Building wildfire raster dataset with %d worker(s) for HRRR extraction; sample_hours_utc=%s; next_day_only=%s",
        worker_count,
        sorted(sample_hours) if sample_hours is not None else "all",
        next_day_only,
    )

    with ThreadPoolExecutor(max_workers=worker_count) if worker_count > 1 else nullcontext() as pool:
        for fire_idx, rec in enumerate(records):
            if fire_idx % 200 == 0:
                LOGGER.info("Processing fire %d / %d", fire_idx + 1, len(records))

            bounds_xy = _build_bounds_xy(rec, buffer_km=config.wildfire.buffer_km)

            for run_time in _iter_fire_hours(
                rec,
                cadence_after_72h=reduction_plan.cadence_after_72h,
                cap=reduction_plan.cap_samples_per_fire,
                sample_hours_utc=sample_hours,
                max_hours_per_fire=max_hours_per_fire,
            ):
                if max_samples_total is not None and sample_count >= int(max_samples_total):
                    break
                input_patches: dict[str, np.ndarray] = {}
                label_patches: dict[int, np.ndarray] = {}

                missing = False
                frp_patch = viirs_rasterizer.patch_for_hour(
                    run_time_utc=run_time,
                    bounds_xy=bounds_xy,
                    patch_size=patch_size,
                )
                input_patches[config.wildfire.frp_channel_name] = frp_patch

                if worker_count == 1:
                    for spec, channel_name in zip(active_specs, hrrr_channels):
                        patch = _extract_patch(
                            reader,
                            run_time_utc=run_time,
                            spec=spec,
                            bounds_xy=bounds_xy,
                            patch_size=patch_size,
                        )
                        if patch is None:
                            missing = True
                            break
                        input_patches[channel_name] = patch

                    if missing:
                        continue

                    for lead in leads:
                        label_time = run_time + timedelta(hours=lead)
                        patch = _extract_patch(
                            reader,
                            run_time_utc=label_time,
                            spec=config.wildfire.label_variable,
                            bounds_xy=bounds_xy,
                            patch_size=patch_size,
                        )
                        if patch is None:
                            missing = True
                            break
                        label_patches[lead] = patch
                else:
                    futures = {}
                    for spec, channel_name in zip(active_specs, hrrr_channels):
                        fut = pool.submit(
                            _extract_patch_threaded,
                            config,
                            run_time,
                            spec,
                            bounds_xy,
                            patch_size,
                        )
                        futures[fut] = ("input", channel_name)

                    for lead in leads:
                        label_time = run_time + timedelta(hours=lead)
                        fut = pool.submit(
                            _extract_patch_threaded,
                            config,
                            label_time,
                            config.wildfire.label_variable,
                            bounds_xy,
                            patch_size,
                        )
                        futures[fut] = ("label", lead)

                    for fut in as_completed(futures):
                        kind, key = futures[fut]
                        patch = fut.result()
                        if patch is None:
                            missing = True
                            break
                        if kind == "input":
                            input_patches[str(key)] = patch
                        else:
                            label_patches[int(key)] = patch

                if missing:
                    continue

                write_idx: int | None = None
                for channel in channels:
                    idx = _append_2d(input_arrays[channel], input_patches[channel].astype(np.float16, copy=False))
                    if write_idx is None:
                        write_idx = idx

                assert write_idx is not None

                for lead in leads:
                    idx2 = _append_2d(label_arrays[lead], label_patches[lead].astype(np.float16, copy=False))
                    if idx2 != write_idx:
                        raise RuntimeError("Label and input array index misalignment")

                if sample_count < 16:
                    config.paths.qa_tiff_dir.mkdir(parents=True, exist_ok=True)
                    _maybe_write_qa_tiff(
                        config.paths.qa_tiff_dir,
                        write_idx,
                        config.wildfire.frp_channel_name,
                        input_patches[config.wildfire.frp_channel_name],
                    )

                split = split_for_date(run_time.date(), config.split)
                massden_patch = input_patches.get("MASSDEN_8m")
                ugrd10_patch = input_patches.get("UGRD_10m")
                vgrd10_patch = input_patches.get("VGRD_10m")
                persistence_mse_12 = np.nan
                persistence_mse_24 = np.nan
                advection_mse_12 = np.nan
                advection_mse_24 = np.nan
                if massden_patch is not None:
                    if 12 in leads:
                        persistence_mse_12 = float(np.nanmean((massden_patch - label_patches[12]) ** 2))
                    if 24 in leads:
                        persistence_mse_24 = float(np.nanmean((massden_patch - label_patches[24]) ** 2))
                if massden_patch is not None and ugrd10_patch is not None and vgrd10_patch is not None:
                    if 12 in leads:
                        adv12 = _advection_baseline(massden_patch, ugrd10_patch, vgrd10_patch, lead_hours=12)
                        advection_mse_12 = float(np.nanmean((adv12 - label_patches[12]) ** 2))
                    if 24 in leads:
                        adv24 = _advection_baseline(massden_patch, ugrd10_patch, vgrd10_patch, lead_hours=24)
                        advection_mse_24 = float(np.nanmean((adv24 - label_patches[24]) ** 2))

                index_rows.append(
                    {
                        "sample_id": int(write_idx),
                        "fire_id": rec.unique_fire_id,
                        "incident_name": rec.incident_name,
                        "state": rec.state,
                        "run_time_utc": run_time.isoformat(),
                        "run_date": run_time.date().isoformat(),
                        "bbox_min_lon": rec.min_lon,
                        "bbox_min_lat": rec.min_lat,
                        "bbox_max_lon": rec.max_lon,
                        "bbox_max_lat": rec.max_lat,
                        "size_acres": rec.size_acres,
                        "split": split,
                        "label_t_plus_12h_available": 12 in leads,
                        "label_t_plus_24h_available": 24 in leads,
                        "persistence_mse_t_plus_12h": persistence_mse_12,
                        "persistence_mse_t_plus_24h": persistence_mse_24,
                        "advection_mse_t_plus_12h": advection_mse_12,
                        "advection_mse_t_plus_24h": advection_mse_24,
                    }
                )

                sample_count += 1

                if sample_count % config.storage.projection_check_interval == 0:
                    LOGGER.info("Raster samples written: %d", sample_count)

            if max_samples_total is not None and sample_count >= int(max_samples_total):
                break

    index_df = pd.DataFrame(index_rows)
    config.paths.wildfire_index_output.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_parquet(config.paths.wildfire_index_output, index=False)
    baseline_summary = (
        index_df[
            [
                "sample_id",
                "split",
                "persistence_mse_t_plus_12h",
                "persistence_mse_t_plus_24h",
                "advection_mse_t_plus_12h",
                "advection_mse_t_plus_24h",
            ]
        ]
        .copy()
    )
    baseline_summary_path = config.paths.processed_dir / "forecast_baselines.parquet"
    baseline_summary.to_parquet(baseline_summary_path, index=False)

    build_log = {
        "fire_count": len(records),
        "sample_count_written": int(sample_count),
        "frp_source": "VIIRS",
        "channels_used": channels,
        "label_leads": leads,
        "estimated_sample_count": int(estimated_samples),
        "estimated_bytes": int(estimated_bytes),
        "budget_bytes": int(config.storage.budget_gb * (1024**3)),
        "sample_hours_utc": sorted(sample_hours) if sample_hours is not None else "all",
        "next_day_only": bool(next_day_only),
        "reduction_actions": reduction_plan.actions,
        "reduction_plan": {
            "include_upper_air": reduction_plan.include_upper_air,
            "cadence_after_72h": reduction_plan.cadence_after_72h,
            "cap_samples_per_fire": reduction_plan.cap_samples_per_fire,
        },
        "baseline_summary_output": str(baseline_summary_path),
    }

    config.paths.dataset_build_log_output.parent.mkdir(parents=True, exist_ok=True)
    config.paths.dataset_build_log_output.write_text(json.dumps(build_log, indent=2), encoding="utf-8")

    LOGGER.info("Wildfire raster build complete: %d samples", sample_count)

    return WildfireRasterStats(
        fire_count=len(records),
        sample_count=sample_count,
        channels_used=len(channels),
        estimated_bytes=estimated_bytes,
    )
