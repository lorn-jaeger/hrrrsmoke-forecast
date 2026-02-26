#!/bin/zsh
set -euo pipefail
source /Users/lorn/Code/gribcheck/.venv/bin/activate
CFG=/Users/lorn/Code/gribcheck/config/pipeline_config.yaml
WORKERS=${WORKERS:-8}

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START ingest-pm"
gribcheck --config "$CFG" ingest-pm

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START build-station-hrrr-daily"
gribcheck --config "$CFG" build-station-hrrr-daily --workers "$WORKERS" --resume --checkpoint-flush-hours 1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START evaluate-accuracy"
gribcheck --config "$CFG" evaluate-accuracy --workers "$WORKERS"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START build-wildfire-raster-dataset"
gribcheck --config "$CFG" build-wildfire-raster-dataset --workers "$WORKERS" --sample-hours-utc 0,6,12,18 --next-day-only

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] PIPELINE_COMPLETE"
