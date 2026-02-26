from __future__ import annotations

import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

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


def _iter_fire_days(
    rec: FireRecord,
    cap: int | None,
    run_start: date,
    run_end: date,
    require_next_day: bool,
    max_days_per_fire: int | None = None,
):
    start_day = max(rec.start_date, run_start)
    end_day = min(rec.end_date, run_end)
    if require_next_day:
        end_day = min(end_day, run_end - timedelta(days=1))
    if end_day < start_day:
        return

    emitted = 0
    day = start_day
    while day <= end_day:
        if max_days_per_fire is not None and emitted >= int(max_days_per_fire):
            break
        yield day
        emitted += 1
        if cap is not None and emitted >= cap:
            break
        day += timedelta(days=1)


def _days_in_fire_window(
    rec: FireRecord,
    cap: int | None,
    run_start: date,
    run_end: date,
    require_next_day: bool,
) -> int:
    return sum(
        1
        for _ in _iter_fire_days(
            rec,
            cap=cap,
            run_start=run_start,
            run_end=run_end,
            require_next_day=require_next_day,
            max_days_per_fire=None,
        )
    )


def _estimate_total_daily_samples(
    records: list[FireRecord],
    cap: int | None,
    run_start: date,
    run_end: date,
    require_next_day: bool,
) -> int:
    return int(
        sum(
            _days_in_fire_window(
                r,
                cap=cap,
                run_start=run_start,
                run_end=run_end,
                require_next_day=require_next_day,
            )
            for r in records
        )
    )


def _build_daily_reduction_plan(
    config: PipelineConfig,
    records: list[FireRecord],
    label_count: int,
    require_next_day: bool,
) -> tuple[ReductionPlan, int, int]:
    active_specs = list(config.wildfire.analysis_variables)

    include_upper_air = True
    cap = None
    actions: list[str] = []

    def current_specs() -> list[VariableSpec]:
        if include_upper_air:
            return list(active_specs)
        return [spec for spec in active_specs if not _is_upper_air_or_dzdt(spec)]

    budget_bytes = int(config.storage.budget_gb * (1024**3))

    while True:
        sample_count = _estimate_total_daily_samples(
            records,
            cap=cap,
            run_start=config.run.start_date,
            run_end=config.run.end_date,
            require_next_day=require_next_day,
        )
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
                cadence_after_72h=1,
                cap_samples_per_fire=cap,
                actions=actions,
            )
            return plan, sample_count, est_bytes

        if include_upper_air and config.storage.drop_upper_air_and_dzdt:
            include_upper_air = False
            actions.append("drop_upper_air_and_dzdt")
            continue

        if cap is None:
            cap = config.storage.cap_samples_per_fire
            actions.append(f"cap_samples_per_fire={cap}")
            continue

        plan = ReductionPlan(
            include_upper_air=include_upper_air,
            cadence_after_72h=1,
            cap_samples_per_fire=cap,
            actions=actions,
        )
        return plan, sample_count, est_bytes


def _utc_datetime(day_utc: date, hour_utc: int) -> datetime:
    return datetime.combine(day_utc, time(hour_utc, 0), tzinfo=timezone.utc)


def _mean_patches(patches: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(patches, axis=0).astype(np.float32, copy=False)
    return np.mean(stack, axis=0, dtype=np.float32)


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


def _extract_patch_threaded(
    reader: HRRRAnalysisReader,
    run_time_utc: datetime,
    spec: VariableSpec,
    bounds_xy,
    patch_size,
):
    return _extract_patch(reader, run_time_utc=run_time_utc, spec=spec, bounds_xy=bounds_xy, patch_size=patch_size)


def _index_checkpoint_path(index_path: Path) -> Path:
    return index_path.with_suffix(".checkpoint.jsonl")


def _sample_key(
    fire_id: str,
    sample_day: date,
    run_time: datetime,
    daily_aggregate: bool,
    sample_hours: set[int] | None,
) -> str:
    if daily_aggregate:
        hours = ",".join(str(h) for h in sorted(sample_hours or []))
        return f"{fire_id}|daily|{sample_day.isoformat()}|{hours}"
    run_time_utc = run_time.astimezone(timezone.utc).replace(microsecond=0)
    return f"{fire_id}|hourly|{run_time_utc.isoformat()}"


def _sample_key_from_index_row(row: dict[str, object]) -> str:
    sample_key = row.get("sample_key")
    if isinstance(sample_key, str) and sample_key.strip():
        return sample_key

    fire_id = str(row.get("fire_id", ""))
    mode = str(row.get("aggregation_mode", "hourly"))
    if mode.startswith("daily"):
        run_date = str(row.get("run_date", ""))
        source_hours = str(row.get("source_hours_utc", ""))
        return f"{fire_id}|daily|{run_date}|{source_hours}"

    run_time_utc = str(row.get("run_time_utc", ""))
    return f"{fire_id}|hourly|{run_time_utc}"


def _read_index_checkpoint_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rec = json.loads(text)
            except json.JSONDecodeError:
                LOGGER.warning("Skipping malformed index checkpoint line %d in %s", line_no, path)
                continue
            if isinstance(rec, dict):
                rows.append(rec)
    return rows


def _append_index_checkpoint_rows(path: Path, rows: list[dict[str, object]], mode: str = "a") -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":"), allow_nan=True))
            f.write("\n")


def _load_resume_index_rows(index_path: Path, checkpoint_path: Path) -> tuple[list[dict[str, object]], str]:
    if checkpoint_path.exists():
        rows = _read_index_checkpoint_rows(checkpoint_path)
        if rows:
            return rows, "checkpoint"

    if index_path.exists():
        df = pd.read_parquet(index_path)
        if df.empty:
            return [], "parquet"
        rows = df.to_dict(orient="records")
        return rows, "parquet"

    return [], "none"


def _validate_resume_index_mode(
    rows: list[dict[str, object]],
    daily_aggregate: bool,
    sample_hours: set[int] | None,
) -> None:
    if not rows:
        return
    expected_mode = "daily_4h_mean" if daily_aggregate else "hourly"
    expected_hours = ",".join(str(h) for h in sorted(sample_hours or [])) if daily_aggregate else None

    mode_values: set[str] = set()
    hour_values: set[str] = set()
    for row in rows:
        mode_val = row.get("aggregation_mode")
        if mode_val is not None and str(mode_val) != "":
            mode_values.add(str(mode_val))
        hour_val = row.get("source_hours_utc")
        if hour_val is not None and str(hour_val) != "":
            hour_values.add(str(hour_val))

    if mode_values and mode_values != {expected_mode}:
        raise ValueError(
            f"Resume index has aggregation_mode={sorted(mode_values)}, expected {expected_mode}; "
            "use --no-resume to restart output from scratch."
        )

    if daily_aggregate and expected_hours is not None and hour_values and hour_values != {expected_hours}:
        raise ValueError(
            f"Resume index has source_hours_utc={sorted(hour_values)}, expected {expected_hours}; "
            "use --no-resume to restart output from scratch."
        )


def _create_or_open_zarr(
    path: Path,
    channels: list[str],
    leads: list[int],
    patch_size: tuple[int, int],
    dtype: str,
    compressor: str,
    resume: bool,
):
    if resume and path.exists():
        group = zarr.open_group(str(path), mode="a")
        existing_channels = [str(v) for v in group.attrs.get("channels", [])]
        existing_leads = [int(v) for v in group.attrs.get("label_leads_hours", [])]
        existing_patch = tuple(int(v) for v in group.attrs.get("patch_size", []))
        if existing_channels != channels or existing_leads != leads or existing_patch != patch_size:
            raise ValueError(
                "Existing wildfire zarr metadata does not match current run settings; "
                "use --no-resume to rebuild."
            )
        for channel in channels:
            arr = group[f"inputs/{channel}"]
            if tuple(arr.shape[1:]) != tuple(patch_size):
                raise ValueError(f"Existing input dataset shape mismatch for channel {channel}")
            if np.dtype(arr.dtype) != np.dtype(dtype):
                raise ValueError(f"Existing input dataset dtype mismatch for channel {channel}")
        for lead in leads:
            arr = group[f"labels/t_plus_{lead}h"]
            if tuple(arr.shape[1:]) != tuple(patch_size):
                raise ValueError(f"Existing label dataset shape mismatch for lead {lead}")
            if np.dtype(arr.dtype) != np.dtype(dtype):
                raise ValueError(f"Existing label dataset dtype mismatch for lead {lead}")
        return group

    if path.exists():
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


def _align_zarr_sample_count(
    input_arrays: dict[str, zarr.Array],
    label_arrays: dict[int, zarr.Array],
    expected_count: int,
) -> int:
    truncated = 0
    for arr in list(input_arrays.values()) + list(label_arrays.values()):
        n = int(arr.shape[0])
        if n < expected_count:
            raise ValueError(
                f"Existing zarr length ({n}) is smaller than resume index rows ({expected_count}); "
                "use --no-resume to rebuild."
            )
        if n > expected_count:
            arr.resize((expected_count, arr.shape[1], arr.shape[2]))
            truncated += n - expected_count
    return truncated


def _materialize_index_parquet_from_checkpoint(
    checkpoint_path: Path,
    index_path: Path,
    fallback_rows: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    if checkpoint_path.exists():
        rows = _read_index_checkpoint_rows(checkpoint_path)
    else:
        rows = list(fallback_rows or [])

    if not rows:
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(rows)
        if "sample_id" in df.columns:
            df = df.sort_values("sample_id").drop_duplicates(subset=["sample_id"], keep="last").reset_index(drop=True)
        if "sample_key" in df.columns:
            df = df.drop_duplicates(subset=["sample_key"], keep="last").reset_index(drop=True)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(index_path, index=False)
    return df


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
    daily_aggregate: bool = False,
    resume: bool = True,
    checkpoint_flush_samples: int = 1,
) -> WildfireRasterStats:
    records = load_filtered_fire_records(config)
    if not records:
        raise ValueError("No wildfire records available after filtering")
    records = sorted(records, key=lambda r: (r.start_time_utc, r.end_time_utc, r.unique_fire_id))
    if max_fires is not None:
        records = records[: int(max_fires)]

    if daily_aggregate and not next_day_only:
        raise ValueError("daily_aggregate mode requires --next-day-only for an unambiguous daily target")

    normalized_hours = _normalize_sample_hours(sample_hours_utc)
    if daily_aggregate and normalized_hours is None:
        sample_hours = {0, 6, 12, 18}
    else:
        sample_hours = normalized_hours

    leads = [24] if next_day_only else sorted(config.wildfire.label_lead_hours)

    if daily_aggregate:
        reduction_plan, estimated_samples, estimated_bytes = _build_daily_reduction_plan(
            config,
            records,
            label_count=len(leads),
            require_next_day=bool(next_day_only),
        )
    else:
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
    max_samples_cap = int(max_samples_total) if max_samples_total is not None else None
    flush_every = max(1, int(checkpoint_flush_samples))
    label_channel_name: str | None = None
    for spec, channel_name in zip(active_specs, hrrr_channels):
        if (
            spec.variable.upper() == config.wildfire.label_variable.variable.upper()
            and str(spec.level) == str(config.wildfire.label_variable.level)
        ):
            label_channel_name = channel_name
            break

    index_checkpoint = _index_checkpoint_path(config.paths.wildfire_index_output)
    if not resume:
        if config.paths.wildfire_index_output.exists():
            config.paths.wildfire_index_output.unlink()
        if index_checkpoint.exists():
            index_checkpoint.unlink()

    existing_rows, resume_source = _load_resume_index_rows(config.paths.wildfire_index_output, index_checkpoint) if resume else ([], "disabled")
    _validate_resume_index_mode(existing_rows, daily_aggregate=daily_aggregate, sample_hours=sample_hours)
    for row in existing_rows:
        row["sample_key"] = _sample_key_from_index_row(row)

    if existing_rows:
        dedupe_df = pd.DataFrame(existing_rows)
        if "sample_id" in dedupe_df.columns:
            dedupe_df = (
                dedupe_df.sort_values("sample_id")
                .drop_duplicates(subset=["sample_id"], keep="last")
                .reset_index(drop=True)
            )
        if "sample_key" in dedupe_df.columns:
            dedupe_df = dedupe_df.drop_duplicates(subset=["sample_key"], keep="last").reset_index(drop=True)
        existing_rows = dedupe_df.to_dict(orient="records")

    if resume and existing_rows and not config.paths.wildfire_zarr_output.exists():
        raise ValueError(
            f"Resume index exists ({len(existing_rows)} rows) but zarr output is missing at "
            f"{config.paths.wildfire_zarr_output}; use --no-resume to rebuild."
        )

    if resume and existing_rows and resume_source == "parquet" and not index_checkpoint.exists():
        _append_index_checkpoint_rows(index_checkpoint, existing_rows, mode="w")

    completed_sample_keys: set[str] = {str(row["sample_key"]) for row in existing_rows}
    resumed_samples = len(existing_rows)

    # Keep an in-process LRU cache of full HRRR fields so repeated fire AOIs
    # at the same (run_time, variable, level) do not trigger repeated S3 reads.
    field_cache_entries = max(32, min(256, worker_count * 16))
    reader = HRRRAnalysisReader(config.hrrr, max_cache_entries=field_cache_entries)
    viirs_df = load_or_build_viirs_hourly_points(config)
    viirs_rasterizer = build_viirs_rasterizer(viirs_df)

    zarr_group = _create_or_open_zarr(
        path=config.paths.wildfire_zarr_output,
        channels=channels,
        leads=leads,
        patch_size=patch_size,
        dtype=config.storage.dtype,
        compressor=config.storage.compressor,
        resume=bool(resume),
    )

    index_rows_buffer: list[dict[str, object]] = []

    input_arrays = {channel: zarr_group[f"inputs/{channel}"] for channel in channels}
    label_arrays = {lead: zarr_group[f"labels/t_plus_{lead}h"] for lead in leads}
    truncated_samples = _align_zarr_sample_count(input_arrays, label_arrays, expected_count=resumed_samples)
    if truncated_samples > 0:
        LOGGER.warning(
            "Truncated %d unindexed samples from existing zarr to align with resume index rows",
            truncated_samples,
        )

    sample_count = resumed_samples
    new_samples_written = 0
    reused_next_day_massden_inputs = 0

    def _flush_index_rows() -> None:
        nonlocal index_rows_buffer
        if not index_rows_buffer:
            return
        _append_index_checkpoint_rows(index_checkpoint, index_rows_buffer, mode="a")
        index_rows_buffer = []

    LOGGER.info(
        "Building wildfire raster dataset with %d worker(s) for HRRR extraction; field_cache_entries=%d; daily_aggregate=%s; sample_hours_utc=%s; next_day_only=%s; resume=%s; resumed_samples=%d; flush_every=%d",
        worker_count,
        field_cache_entries,
        daily_aggregate,
        sorted(sample_hours) if sample_hours is not None else "all",
        next_day_only,
        bool(resume),
        resumed_samples,
        flush_every,
    )

    try:
        with ThreadPoolExecutor(max_workers=worker_count) if worker_count > 1 else nullcontext() as pool:
            for fire_idx, rec in enumerate(records):
                if fire_idx % 200 == 0:
                    LOGGER.info("Processing fire %d / %d", fire_idx + 1, len(records))

                bounds_xy = _build_bounds_xy(rec, buffer_km=config.wildfire.buffer_km)
                next_day_label_cache: dict[date, np.ndarray] = {}
                if daily_aggregate:
                    if sample_hours is None:
                        raise ValueError("daily_aggregate mode requires explicit sample hours")
                    sample_hours_sorted = sorted(sample_hours)
                    max_days_per_fire = int(max_hours_per_fire) if max_hours_per_fire is not None else None
                    sample_iterator = (
                        (sample_day, _utc_datetime(sample_day, sample_hours_sorted[0]))
                        for sample_day in _iter_fire_days(
                            rec,
                            cap=reduction_plan.cap_samples_per_fire,
                            run_start=config.run.start_date,
                            run_end=config.run.end_date,
                            require_next_day=bool(next_day_only),
                            max_days_per_fire=max_days_per_fire,
                        )
                    )
                else:
                    sample_iterator = (
                        (run_time.date(), run_time)
                        for run_time in _iter_fire_hours(
                            rec,
                            cadence_after_72h=reduction_plan.cadence_after_72h,
                            cap=reduction_plan.cap_samples_per_fire,
                            sample_hours_utc=sample_hours,
                            max_hours_per_fire=max_hours_per_fire,
                        )
                    )

                for sample_day, run_time in sample_iterator:
                    if max_samples_cap is not None and sample_count >= max_samples_cap:
                        break

                    sample_key = _sample_key(
                        fire_id=rec.unique_fire_id,
                        sample_day=sample_day,
                        run_time=run_time,
                        daily_aggregate=daily_aggregate,
                        sample_hours=sample_hours,
                    )
                    if sample_key in completed_sample_keys:
                        continue

                    if daily_aggregate:
                        source_times = [_utc_datetime(sample_day, hour) for hour in sorted(sample_hours)]
                        label_times_by_lead = {
                            24: [_utc_datetime(sample_day + timedelta(days=1), hour) for hour in sorted(sample_hours)]
                        }
                    else:
                        source_times = [run_time]
                        label_times_by_lead = {
                            lead: [run_time + timedelta(hours=lead)]
                            for lead in leads
                        }

                    input_hourly: dict[str, list[np.ndarray]] = {config.wildfire.frp_channel_name: []}
                    for channel_name in hrrr_channels:
                        input_hourly[channel_name] = []
                    label_hourly: dict[int, list[np.ndarray]] = {lead: [] for lead in leads}

                    missing = False

                    for ts in source_times:
                        frp_patch = viirs_rasterizer.patch_for_hour(
                            run_time_utc=ts,
                            bounds_xy=bounds_xy,
                            patch_size=patch_size,
                        )
                        input_hourly[config.wildfire.frp_channel_name].append(frp_patch)

                    if worker_count == 1:
                        for spec, channel_name in zip(active_specs, hrrr_channels):
                            if (
                                daily_aggregate
                                and label_channel_name is not None
                                and channel_name == label_channel_name
                                and sample_day in next_day_label_cache
                            ):
                                cached_patch = next_day_label_cache.pop(sample_day)
                                input_hourly[channel_name] = [cached_patch] * len(source_times)
                                reused_next_day_massden_inputs += 1
                                continue
                            for ts in source_times:
                                patch = _extract_patch(
                                    reader,
                                    run_time_utc=ts,
                                    spec=spec,
                                    bounds_xy=bounds_xy,
                                    patch_size=patch_size,
                                )
                                if patch is None:
                                    missing = True
                                    break
                                input_hourly[channel_name].append(patch)
                            if missing:
                                break

                        if missing:
                            continue

                        for lead, label_times in label_times_by_lead.items():
                            for ts in label_times:
                                patch = _extract_patch(
                                    reader,
                                    run_time_utc=ts,
                                    spec=config.wildfire.label_variable,
                                    bounds_xy=bounds_xy,
                                    patch_size=patch_size,
                                )
                                if patch is None:
                                    missing = True
                                    break
                                label_hourly[lead].append(patch)
                            if missing:
                                break
                    else:
                        futures = {}
                        if daily_aggregate and label_channel_name is not None and sample_day in next_day_label_cache:
                            cached_patch = next_day_label_cache.pop(sample_day)
                            input_hourly[label_channel_name] = [cached_patch] * len(source_times)
                            reused_next_day_massden_inputs += 1
                        for spec, channel_name in zip(active_specs, hrrr_channels):
                            if (
                                daily_aggregate
                                and label_channel_name is not None
                                and channel_name == label_channel_name
                                and len(input_hourly[channel_name]) == len(source_times)
                            ):
                                continue
                            for ts in source_times:
                                fut = pool.submit(
                                    _extract_patch_threaded,
                                    reader,
                                    ts,
                                    spec,
                                    bounds_xy,
                                    patch_size,
                                )
                                futures[fut] = ("input", channel_name)

                        for lead, label_times in label_times_by_lead.items():
                            for ts in label_times:
                                fut = pool.submit(
                                    _extract_patch_threaded,
                                    reader,
                                    ts,
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
                                input_hourly[str(key)].append(patch)
                            else:
                                label_hourly[int(key)].append(patch)

                    if missing:
                        continue

                    expected_input_count = len(source_times)
                    if any(len(v) != expected_input_count for v in input_hourly.values()):
                        continue
                    if any(len(label_hourly[lead]) != len(label_times_by_lead[lead]) for lead in leads):
                        continue

                    input_patches = {channel: _mean_patches(patches) for channel, patches in input_hourly.items()}
                    label_patches = {lead: _mean_patches(patches) for lead, patches in label_hourly.items()}
                    if daily_aggregate and label_channel_name is not None and 24 in label_patches:
                        next_day_label_cache[sample_day + timedelta(days=1)] = label_patches[24]

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

                    if write_idx < 16:
                        config.paths.qa_tiff_dir.mkdir(parents=True, exist_ok=True)
                        _maybe_write_qa_tiff(
                            config.paths.qa_tiff_dir,
                            write_idx,
                            config.wildfire.frp_channel_name,
                            input_patches[config.wildfire.frp_channel_name],
                        )

                    split = split_for_date(sample_day, config.split)
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

                    row = {
                        "sample_id": int(write_idx),
                        "sample_key": sample_key,
                        "fire_id": rec.unique_fire_id,
                        "incident_name": rec.incident_name,
                        "state": rec.state,
                        "run_time_utc": run_time.isoformat(),
                        "run_date": sample_day.isoformat(),
                        "aggregation_mode": "daily_4h_mean" if daily_aggregate else "hourly",
                        "source_hours_utc": ",".join(str(h) for h in sorted(sample_hours)) if daily_aggregate else str(run_time.hour),
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
                    index_rows_buffer.append(row)
                    completed_sample_keys.add(sample_key)
                    sample_count += 1
                    new_samples_written += 1

                    if len(index_rows_buffer) >= flush_every:
                        _flush_index_rows()

                    if sample_count % config.storage.projection_check_interval == 0:
                        LOGGER.info("Raster samples written: %d", sample_count)

                if max_samples_cap is not None and sample_count >= max_samples_cap:
                    break
    finally:
        _flush_index_rows()

    index_df = _materialize_index_parquet_from_checkpoint(
        checkpoint_path=index_checkpoint,
        index_path=config.paths.wildfire_index_output,
        fallback_rows=existing_rows,
    )
    baseline_columns = [
        "sample_id",
        "split",
        "persistence_mse_t_plus_12h",
        "persistence_mse_t_plus_24h",
        "advection_mse_t_plus_12h",
        "advection_mse_t_plus_24h",
    ]
    baseline_summary = index_df.reindex(columns=baseline_columns).copy()
    baseline_summary_path = config.paths.processed_dir / "forecast_baselines.parquet"
    baseline_summary.to_parquet(baseline_summary_path, index=False)

    build_log = {
        "fire_count": len(records),
        "sample_count_written": int(sample_count),
        "new_samples_written": int(new_samples_written),
        "resumed_samples": int(resumed_samples),
        "resume_enabled": bool(resume),
        "resume_index_source": resume_source,
        "index_checkpoint_path": str(index_checkpoint),
        "checkpoint_flush_samples": int(flush_every),
        "truncated_unindexed_samples": int(truncated_samples),
        "frp_source": "VIIRS",
        "channels_used": channels,
        "label_leads": leads,
        "estimated_sample_count": int(estimated_samples),
        "estimated_bytes": int(estimated_bytes),
        "budget_bytes": int(config.storage.budget_gb * (1024**3)),
        "field_cache_entries": int(field_cache_entries),
        "field_cache_hits": int(reader.cache_hits),
        "field_cache_misses": int(reader.cache_misses),
        "sample_hours_utc": sorted(sample_hours) if sample_hours is not None else "all",
        "next_day_only": bool(next_day_only),
        "daily_aggregate": bool(daily_aggregate),
        "reused_next_day_massden_inputs": int(reused_next_day_massden_inputs),
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

    LOGGER.info(
        "Wildfire raster build complete: total_samples=%d new_samples=%d resumed_samples=%d (field cache hits=%d, misses=%d, next-day-massden-reuse=%d)",
        sample_count,
        new_samples_written,
        resumed_samples,
        reader.cache_hits,
        reader.cache_misses,
        reused_next_day_massden_inputs,
    )

    return WildfireRasterStats(
        fire_count=len(records),
        sample_count=sample_count,
        channels_used=len(channels),
        estimated_bytes=estimated_bytes,
    )
