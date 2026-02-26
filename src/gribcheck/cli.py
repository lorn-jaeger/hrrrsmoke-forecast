from __future__ import annotations

import argparse
import logging
from pathlib import Path

from gribcheck.config import ensure_output_dirs, load_config
from gribcheck.pipelines import accuracy, pm_ingest, station_daily, wildfire_raster


def _parse_hours_csv(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return tuple(int(part.strip()) for part in text.split(",") if part.strip() != "")


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
    station_parser.add_argument("--workers", type=int, default=8, help="Thread worker count for hourly HRRR sampling")
    station_parser.add_argument("--checkpoint-dir", type=str, default=None, help="Optional checkpoint directory for resumable station run")
    station_parser.add_argument("--checkpoint-flush-hours", type=int, default=1, help="Flush checkpoint state every N newly processed hours")
    station_parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing checkpoint (use --no-resume to restart from scratch)",
    )

    snapshot_parser = subparsers.add_parser(
        "materialize-station-snapshot",
        help="Write a partial station daily parquet from checkpoint state",
    )
    snapshot_parser.add_argument("--max-hours", type=int, default=None, help="Optional target-hour view for progress metadata")
    snapshot_parser.add_argument("--station-limit", type=int, default=None, help="Optional station limit used by checkpoint")
    snapshot_parser.add_argument("--checkpoint-dir", type=str, default=None, help="Optional checkpoint directory")
    snapshot_parser.add_argument("--output-path", type=str, default=None, help="Optional output parquet path")

    accuracy_parser = subparsers.add_parser("evaluate-accuracy", help="Evaluate HRRR-vs-sensor accuracy")
    accuracy_parser.add_argument("--workers", type=int, default=8, help="Thread worker count for grouping/proximity work")
    accuracy_parser.add_argument(
        "--from-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Materialize a partial station snapshot from checkpoint before evaluation",
    )
    accuracy_parser.add_argument("--checkpoint-dir", type=str, default=None, help="Optional station checkpoint directory")

    wildfire_parser = subparsers.add_parser("build-wildfire-raster-dataset", help="Build wildfire raster forecast dataset")
    wildfire_parser.add_argument("--max-fires", type=int, default=None, help="Optional cap on wildfire records")
    wildfire_parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on total raster samples")
    wildfire_parser.add_argument("--max-hours-per-fire", type=int, default=None, help="Optional cap on scanned run hours per fire")
    wildfire_parser.add_argument("--workers", type=int, default=8, help="Thread worker count for HRRR patch extraction")
    wildfire_parser.add_argument(
        "--sample-hours-utc",
        type=str,
        default=None,
        help="Comma-separated UTC hours to sample each fire day (example: 0,6,12,18). Defaults to all hours.",
    )
    wildfire_parser.add_argument(
        "--next-day-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use only +24h next-day actual analysis label (no +12h label).",
    )
    wildfire_parser.add_argument(
        "--daily-aggregate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Aggregate selected source hours into one daily sample (labels follow selected target mode).",
    )

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
            resume=bool(args.resume),
            checkpoint_dir=(Path(args.checkpoint_dir) if args.checkpoint_dir else None),
            checkpoint_flush_hours=args.checkpoint_flush_hours,
        )
        print(stats)
        return

    if args.command == "materialize-station-snapshot":
        out = station_daily.materialize_checkpoint_snapshot(
            config,
            max_hours=args.max_hours,
            station_limit=args.station_limit,
            checkpoint_dir=(Path(args.checkpoint_dir) if args.checkpoint_dir else None),
            output_path=(Path(args.output_path) if args.output_path else None),
        )
        print(out)
        return

    if args.command == "evaluate-accuracy":
        stats = accuracy.run(
            config,
            workers=args.workers,
            from_checkpoint=bool(args.from_checkpoint),
            checkpoint_dir=(Path(args.checkpoint_dir) if args.checkpoint_dir else None),
        )
        print(stats)
        return

    if args.command == "build-wildfire-raster-dataset":
        stats = wildfire_raster.run(
            config,
            max_fires=args.max_fires,
            max_samples_total=args.max_samples,
            max_hours_per_fire=args.max_hours_per_fire,
            workers=args.workers,
            sample_hours_utc=_parse_hours_csv(args.sample_hours_utc),
            next_day_only=bool(args.next_day_only),
            daily_aggregate=bool(args.daily_aggregate),
        )
        print(stats)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
