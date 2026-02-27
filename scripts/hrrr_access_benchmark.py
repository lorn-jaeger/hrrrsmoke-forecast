#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import s3fs
import xarray as xr

MIB = 1024 * 1024

DEFAULT_MULTI_VAR_SPECS: tuple[tuple[str, str], ...] = (
    ("MASSDEN", "8m_above_ground"),
    ("COLMD", "entire_atmosphere_single_layer"),
    ("UGRD", "10m_above_ground"),
    ("VGRD", "10m_above_ground"),
    ("UGRD", "850mb"),
    ("VGRD", "850mb"),
    ("DZDT", "700mb"),
    ("HPBL", "surface"),
    ("PRATE", "surface"),
    ("TMP", "2m_above_ground"),
    ("RH", "2m_above_ground"),
)


@dataclass
class BenchResult:
    pattern_key: str
    pattern_name: str
    source: str
    repeats: int
    bytes_read: int
    elapsed_seconds: float
    throughput_mib_s: float
    success: bool
    error: str | None = None


@dataclass
class PatternSpec:
    key: str
    name: str
    source: str
    runner: Callable[[], int]


def parse_iso_datetime(text: str) -> datetime:
    value = text.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def zarr_field_store_path(
    run_time_utc: datetime,
    variable: str,
    level: str,
    bucket: str,
    product: str,
) -> str:
    ymd = run_time_utc.strftime("%Y%m%d")
    hh = run_time_utc.strftime("%H")
    return f"s3://{bucket}/{product}/{ymd}/{ymd}_{hh}z_anl.zarr/{level}/{variable}/{level}"


def open_zarr_field(
    run_time_utc: datetime,
    variable: str,
    level: str,
    bucket: str,
    product: str,
    anonymous: bool,
) -> xr.DataArray:
    fs = s3fs.S3FileSystem(anon=anonymous)
    store_path = zarr_field_store_path(
        run_time_utc=run_time_utc,
        variable=variable,
        level=level,
        bucket=bucket,
        product=product,
    )
    mapper = s3fs.S3Map(store_path, s3=fs, check=False)
    ds = xr.open_zarr(mapper, consolidated=False, decode_timedelta=False)

    if variable in ds.data_vars:
        da = ds[variable]
    elif len(ds.data_vars) == 1:
        da = next(iter(ds.data_vars.values()))
    else:
        raise KeyError(f"{variable} not found in {store_path}; vars={list(ds.data_vars)}")

    for dim in ("time", "reference_time", "step"):
        if dim in da.dims:
            da = da.isel({dim: 0})
    return da


def center_subset(da: xr.DataArray, height: int, width: int) -> xr.DataArray:
    if da.ndim < 2:
        raise ValueError(f"Expected at least 2 dimensions, got {da.ndim}")

    y_dim = da.dims[-2]
    x_dim = da.dims[-1]
    y_size = int(da.sizes[y_dim])
    x_size = int(da.sizes[x_dim])

    h = max(1, min(height, y_size))
    w = max(1, min(width, x_size))
    y0 = max(0, (y_size - h) // 2)
    x0 = max(0, (x_size - w) // 2)
    return da.isel({y_dim: slice(y0, y0 + h), x_dim: slice(x0, x0 + w)})


def load_da_bytes(da: xr.DataArray) -> int:
    arr = da.to_numpy()
    return int(arr.nbytes)


def download_http(
    url: str,
    chunk_bytes: int,
    timeout_seconds: float,
    byte_limit: int,
    use_range_header: bool,
) -> int:
    headers: dict[str, str] = {
        "User-Agent": "gribcheck-benchmark/0.1 (+https://github.com/)",
        "Accept": "*/*",
        "Connection": "keep-alive",
    }
    if use_range_header and byte_limit > 0:
        headers["Range"] = f"bytes=0-{byte_limit - 1}"

    req = urllib.request.Request(url=url, headers=headers)
    read_total = 0
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        while True:
            block = resp.read(chunk_bytes)
            if not block:
                break
            read_total += len(block)
            if byte_limit > 0 and not use_range_header and read_total >= byte_limit:
                break
    return int(read_total)


def grib_url(source: str, run_time_utc: datetime) -> str:
    ymd = run_time_utc.strftime("%Y%m%d")
    hh = run_time_utc.strftime("%H")
    if source == "aws":
        return f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{ymd}/conus/hrrr.t{hh}z.wrfsfcf00.grib2"
    if source == "pando":
        return f"https://pando-rgw01.chpc.utah.edu/hrrr/sfc/{ymd}/hrrr.t{hh}z.wrfsfcf00.grib2"
    if source == "gcs":
        return f"https://storage.googleapis.com/high-resolution-rapid-refresh/hrrr.{ymd}/conus/hrrr.t{hh}z.wrfsfcf00.grib2"
    if source == "nomads":
        return f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.{ymd}/conus/hrrr.t{hh}z.wrfsfcf00.grib2"
    raise ValueError(f"Unsupported GRIB source: {source}")


def run_pattern(spec: PatternSpec, repeats: int) -> BenchResult:
    total_bytes = 0
    total_elapsed = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        nbytes = spec.runner()
        elapsed = time.perf_counter() - t0
        total_bytes += int(nbytes)
        total_elapsed += float(elapsed)

    throughput = (total_bytes / MIB) / max(total_elapsed, 1e-9)
    return BenchResult(
        pattern_key=spec.key,
        pattern_name=spec.name,
        source=spec.source,
        repeats=repeats,
        bytes_read=int(total_bytes),
        elapsed_seconds=float(total_elapsed),
        throughput_mib_s=float(throughput),
        success=True,
    )


def run_pattern_safe(spec: PatternSpec, repeats: int) -> BenchResult:
    try:
        return run_pattern(spec, repeats=repeats)
    except Exception as exc:  # pragma: no cover - network failures are expected in benchmarks
        return BenchResult(
            pattern_key=spec.key,
            pattern_name=spec.name,
            source=spec.source,
            repeats=repeats,
            bytes_read=0,
            elapsed_seconds=0.0,
            throughput_mib_s=0.0,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def parse_hours_csv(value: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if token == "":
            continue
        h = int(token)
        if h < 0 or h > 23:
            raise ValueError(f"Hour must be 0..23, got {h}")
        out.append(h)
    if not out:
        raise ValueError("At least one sample hour is required")
    return tuple(sorted(set(out)))


def parse_multi_var_specs(value: str) -> tuple[tuple[str, str], ...]:
    if value.strip() == "":
        return DEFAULT_MULTI_VAR_SPECS

    specs: list[tuple[str, str]] = []
    for item in value.split(","):
        token = item.strip()
        if token == "":
            continue
        if "/" not in token:
            raise ValueError(f"Expected VARIABLE/LEVEL token, got: {token}")
        variable, level = token.split("/", 1)
        specs.append((variable.strip(), level.strip()))

    if not specs:
        raise ValueError("No multi-variable specs were parsed")
    return tuple(specs)


def parse_sources_csv(value: str) -> tuple[str, ...]:
    valid = {"aws", "pando", "gcs", "nomads"}
    sources: list[str] = []
    for item in value.split(","):
        token = item.strip().lower()
        if token == "":
            continue
        if token not in valid:
            raise ValueError(f"Unsupported source '{token}'. Choose from: {sorted(valid)}")
        sources.append(token)
    if not sources:
        raise ValueError("At least one source is required")
    return tuple(dict.fromkeys(sources))


def format_bytes_mib(value: int) -> str:
    return f"{value / MIB:.1f}"


def make_patterns(args: argparse.Namespace) -> list[PatternSpec]:
    run_time = parse_iso_datetime(args.run_time)
    sample_hours = parse_hours_csv(args.sample_hours_utc)
    multi_specs = parse_multi_var_specs(args.multi_var_specs)
    grib_sources = parse_sources_csv(args.grib_sources)

    selected = {token.strip() for token in args.patterns.split(",") if token.strip()}
    include_all = "all" in selected

    def want(key: str) -> bool:
        return include_all or key in selected

    patterns: list[PatternSpec] = []

    if want("zarr_full"):
        def _zarr_full() -> int:
            da = open_zarr_field(
                run_time_utc=run_time,
                variable=args.zarr_variable,
                level=args.zarr_level,
                bucket=args.zarr_bucket,
                product=args.zarr_product,
                anonymous=bool(args.zarr_anonymous),
            )
            return load_da_bytes(da)

        patterns.append(
            PatternSpec(
                key="zarr_full",
                name="zarr_full_conus_single_field",
                source="hrrrzarr",
                runner=_zarr_full,
            )
        )

    if want("zarr_subset"):
        def _zarr_subset() -> int:
            da = open_zarr_field(
                run_time_utc=run_time,
                variable=args.zarr_variable,
                level=args.zarr_level,
                bucket=args.zarr_bucket,
                product=args.zarr_product,
                anonymous=bool(args.zarr_anonymous),
            )
            sub = center_subset(da, args.subset_size, args.subset_size)
            return load_da_bytes(sub)

        patterns.append(
            PatternSpec(
                key="zarr_subset",
                name=f"zarr_center_subset_{args.subset_size}x{args.subset_size}",
                source="hrrrzarr",
                runner=_zarr_subset,
            )
        )

    if want("zarr_multi_var_seq"):
        def _zarr_multi_var_seq() -> int:
            total = 0
            for variable, level in multi_specs:
                da = open_zarr_field(
                    run_time_utc=run_time,
                    variable=variable,
                    level=level,
                    bucket=args.zarr_bucket,
                    product=args.zarr_product,
                    anonymous=bool(args.zarr_anonymous),
                )
                total += load_da_bytes(da)
            return total

        patterns.append(
            PatternSpec(
                key="zarr_multi_var_seq",
                name=f"zarr_multi_var_seq_{len(multi_specs)}vars",
                source="hrrrzarr",
                runner=_zarr_multi_var_seq,
            )
        )

    if want("zarr_multi_var_threaded"):
        def _zarr_multi_var_threaded() -> int:
            total = 0

            def _load_one(variable: str, level: str) -> int:
                da = open_zarr_field(
                    run_time_utc=run_time,
                    variable=variable,
                    level=level,
                    bucket=args.zarr_bucket,
                    product=args.zarr_product,
                    anonymous=bool(args.zarr_anonymous),
                )
                return load_da_bytes(da)

            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                futures = [pool.submit(_load_one, variable, level) for variable, level in multi_specs]
                for fut in as_completed(futures):
                    total += int(fut.result())
            return total

        patterns.append(
            PatternSpec(
                key="zarr_multi_var_threaded",
                name=f"zarr_multi_var_threaded_{len(multi_specs)}vars",
                source="hrrrzarr",
                runner=_zarr_multi_var_threaded,
            )
        )

    if want("zarr_multi_time_seq"):
        def _zarr_multi_time_seq() -> int:
            total = 0
            for hour in sample_hours:
                ts = datetime.combine(run_time.date(), datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=hour)
                da = open_zarr_field(
                    run_time_utc=ts,
                    variable=args.zarr_variable,
                    level=args.zarr_level,
                    bucket=args.zarr_bucket,
                    product=args.zarr_product,
                    anonymous=bool(args.zarr_anonymous),
                )
                total += load_da_bytes(da)
            return total

        patterns.append(
            PatternSpec(
                key="zarr_multi_time_seq",
                name=f"zarr_multi_time_seq_{len(sample_hours)}hours",
                source="hrrrzarr",
                runner=_zarr_multi_time_seq,
            )
        )

    if want("zarr_multi_time_threaded"):
        def _zarr_multi_time_threaded() -> int:
            total = 0

            def _load_at_hour(hour: int) -> int:
                ts = datetime.combine(run_time.date(), datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=hour)
                da = open_zarr_field(
                    run_time_utc=ts,
                    variable=args.zarr_variable,
                    level=args.zarr_level,
                    bucket=args.zarr_bucket,
                    product=args.zarr_product,
                    anonymous=bool(args.zarr_anonymous),
                )
                return load_da_bytes(da)

            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                futures = [pool.submit(_load_at_hour, hour) for hour in sample_hours]
                for fut in as_completed(futures):
                    total += int(fut.result())
            return total

        patterns.append(
            PatternSpec(
                key="zarr_multi_time_threaded",
                name=f"zarr_multi_time_threaded_{len(sample_hours)}hours",
                source="hrrrzarr",
                runner=_zarr_multi_time_threaded,
            )
        )

    if want("grib_range"):
        range_bytes = max(1, int(args.grib_range_mib)) * MIB

        for source in grib_sources:
            url = grib_url(source, run_time)

            def _make_runner(download_url: str) -> Callable[[], int]:
                def _runner() -> int:
                    return download_http(
                        url=download_url,
                        chunk_bytes=args.http_chunk_kib * 1024,
                        timeout_seconds=float(args.http_timeout_seconds),
                        byte_limit=range_bytes,
                        use_range_header=True,
                    )

                return _runner

            patterns.append(
                PatternSpec(
                    key="grib_range",
                    name=f"grib_http_range_{args.grib_range_mib}MiB",
                    source=source,
                    runner=_make_runner(url),
                )
            )

    if want("grib_full"):
        full_limit_bytes = max(0, int(args.grib_full_limit_mib)) * MIB

        for source in grib_sources:
            url = grib_url(source, run_time)

            def _make_runner(download_url: str) -> Callable[[], int]:
                def _runner() -> int:
                    return download_http(
                        url=download_url,
                        chunk_bytes=args.http_chunk_kib * 1024,
                        timeout_seconds=float(args.http_timeout_seconds),
                        byte_limit=full_limit_bytes,
                        use_range_header=False,
                    )

                return _runner

            suffix = "full_file" if full_limit_bytes == 0 else f"capped_{args.grib_full_limit_mib}MiB"
            patterns.append(
                PatternSpec(
                    key="grib_full",
                    name=f"grib_http_{suffix}",
                    source=source,
                    runner=_make_runner(url),
                )
            )

    if want("grib_range_parallel"):
        range_bytes = max(1, int(args.grib_range_mib)) * MIB
        urls = [grib_url(source, run_time) for source in grib_sources]

        def _grib_range_parallel() -> int:
            total = 0

            def _dl(download_url: str) -> int:
                return download_http(
                    url=download_url,
                    chunk_bytes=args.http_chunk_kib * 1024,
                    timeout_seconds=float(args.http_timeout_seconds),
                    byte_limit=range_bytes,
                    use_range_header=True,
                )

            with ThreadPoolExecutor(max_workers=len(urls)) as pool:
                futures = [pool.submit(_dl, u) for u in urls]
                for fut in as_completed(futures):
                    total += int(fut.result())
            return total

        patterns.append(
            PatternSpec(
                key="grib_range_parallel",
                name=f"grib_http_range_parallel_{args.grib_range_mib}MiB_each",
                source="+".join(grib_sources),
                runner=_grib_range_parallel,
            )
        )

    if want("local_grib") and args.local_grib_path:
        path = Path(args.local_grib_path)

        def _local_grib() -> int:
            total = 0
            limit = max(0, int(args.local_grib_limit_mib)) * MIB
            with path.open("rb") as f:
                while True:
                    block = f.read(args.local_chunk_kib * 1024)
                    if not block:
                        break
                    total += len(block)
                    if limit > 0 and total >= limit:
                        break
            return int(total)

        suffix = "full_file" if int(args.local_grib_limit_mib) == 0 else f"capped_{args.local_grib_limit_mib}MiB"
        patterns.append(
            PatternSpec(
                key="local_grib",
                name=f"local_grib_read_{suffix}",
                source=str(path),
                runner=_local_grib,
            )
        )

    return patterns


def print_ranked_results(results: list[BenchResult]) -> None:
    print("\nResults (sorted by throughput MiB/s):")
    print(
        "rank pattern source repeats bytes_mib elapsed_s throughput_mib_s status"
    )
    sorted_rows = sorted(
        results,
        key=lambda r: (r.success, r.throughput_mib_s),
        reverse=True,
    )
    for idx, row in enumerate(sorted_rows, start=1):
        status = "ok" if row.success else "error"
        print(
            f"{idx:>4} {row.pattern_name:<36} {row.source:<18} {row.repeats:>3} "
            f"{format_bytes_mib(row.bytes_read):>9} {row.elapsed_seconds:>9.3f} {row.throughput_mib_s:>16.2f} {status}"
        )
        if row.error:
            print(f"     error: {row.error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark HRRR access patterns for throughput")

    parser.add_argument("--run-time", default="2024-07-15T00:00:00Z", help="UTC analysis run time to benchmark")
    parser.add_argument("--repeats", type=int, default=1, help="Repeats per pattern")
    parser.add_argument(
        "--patterns",
        type=str,
        default="all",
        help=(
            "Comma-separated keys. Available: all,zarr_full,zarr_subset,zarr_multi_var_seq,"
            "zarr_multi_var_threaded,zarr_multi_time_seq,zarr_multi_time_threaded,"
            "grib_range,grib_full,grib_range_parallel,local_grib"
        ),
    )
    parser.add_argument("--workers", type=int, default=8, help="Thread workers for threaded patterns")

    parser.add_argument("--zarr-bucket", type=str, default="hrrrzarr")
    parser.add_argument("--zarr-product", type=str, default="sfc")
    parser.add_argument("--zarr-anonymous", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zarr-variable", type=str, default="MASSDEN")
    parser.add_argument("--zarr-level", type=str, default="8m_above_ground")
    parser.add_argument("--subset-size", type=int, default=512, help="Square subset size for zarr_subset")
    parser.add_argument(
        "--multi-var-specs",
        type=str,
        default=",".join(f"{v}/{l}" for v, l in DEFAULT_MULTI_VAR_SPECS),
        help="Comma-separated VARIABLE/LEVEL list for multi-var patterns",
    )
    parser.add_argument("--sample-hours-utc", type=str, default="0,6,12,18")

    parser.add_argument(
        "--grib-sources",
        type=str,
        default="aws,pando,gcs",
        help="Comma-separated: aws,pando,gcs,nomads",
    )
    parser.add_argument("--grib-range-mib", type=int, default=64, help="Bytes per source for range test")
    parser.add_argument(
        "--grib-full-limit-mib",
        type=int,
        default=256,
        help="Full-download cap in MiB (0 means full file)",
    )

    parser.add_argument("--http-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--http-chunk-kib", type=int, default=1024)

    parser.add_argument(
        "--local-grib-path",
        type=str,
        default="",
        help="Optional local GRIB2 path for disk-throughput comparison",
    )
    parser.add_argument("--local-grib-limit-mib", type=int, default=0, help="Optional cap for local GRIB read")
    parser.add_argument("--local-chunk-kib", type=int, default=1024)

    parser.add_argument("--output-json", type=str, default="", help="Optional output JSON path")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    patterns = make_patterns(args)
    if not patterns:
        raise ValueError("No patterns selected")

    print(f"Running {len(patterns)} pattern(s), repeats={args.repeats}")
    print(f"run_time={args.run_time}")

    results: list[BenchResult] = []
    for idx, spec in enumerate(patterns, start=1):
        print(f"[{idx}/{len(patterns)}] {spec.name} ({spec.source}) ...", flush=True)
        result = run_pattern_safe(spec, repeats=args.repeats)
        results.append(result)

    print_ranked_results(results)

    payload = {
        "run_time": args.run_time,
        "repeats": args.repeats,
        "results": [asdict(r) for r in results],
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {output_path}")


if __name__ == "__main__":
    main()
