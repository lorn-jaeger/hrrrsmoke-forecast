from __future__ import annotations

import argparse
import logging
from pathlib import Path

from gribcheck.config import ensure_output_dirs, load_config
from gribcheck.pipelines import accuracy, pm_ingest, station_daily, wildfire_raster


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gribcheck", description="HRRR + PM2.5 wildfire data pipelines")
    parser.add_argument(
        "--config",
        default="/Users/lorn/Code/gribcheck/config/pipeline_config.yaml",
        type=str,
        help="Path to YAML config",
    )
    parser.add_argument("--log-level", default="INFO", type=str)

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest-pm", help="Ingest and deduplicate PM2.5 files")

    station_parser = subparsers.add_parser("build-station-hrrr-daily", help="Build station-day HRRR + PM join")
    station_parser.add_argument("--max-hours", type=int, default=None, help="Optional cap on UTC hours processed")
    station_parser.add_argument("--station-limit", type=int, default=None, help="Optional cap on number of stations")
    station_parser.add_argument("--workers", type=int, default=None, help="Thread worker count for hourly HRRR sampling")

    accuracy_parser = subparsers.add_parser("evaluate-accuracy", help="Evaluate HRRR-vs-sensor accuracy")
    accuracy_parser.add_argument("--workers", type=int, default=None, help="Thread worker count for grouping/proximity work")

    wildfire_parser = subparsers.add_parser("build-wildfire-raster-dataset", help="Build wildfire raster forecast dataset")
    wildfire_parser.add_argument("--max-fires", type=int, default=None, help="Optional cap on wildfire records")
    wildfire_parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on total raster samples")
    wildfire_parser.add_argument("--max-hours-per-fire", type=int, default=None, help="Optional cap on scanned run hours per fire")
    wildfire_parser.add_argument("--workers", type=int, default=None, help="Thread worker count for HRRR patch extraction")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    config = load_config(Path(args.config))
    ensure_output_dirs(config)

    if args.command == "ingest-pm":
        stats = pm_ingest.run(config)
        print(stats)
        return

    if args.command == "build-station-hrrr-daily":
        stats = station_daily.run(
            config,
            max_hours=args.max_hours,
            station_limit=args.station_limit,
            workers=args.workers,
        )
        print(stats)
        return

    if args.command == "evaluate-accuracy":
        stats = accuracy.run(config, workers=args.workers)
        print(stats)
        return

    if args.command == "build-wildfire-raster-dataset":
        stats = wildfire_raster.run(
            config,
            max_fires=args.max_fires,
            max_samples_total=args.max_samples,
            max_hours_per_fire=args.max_hours_per_fire,
            workers=args.workers,
        )
        print(stats)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
