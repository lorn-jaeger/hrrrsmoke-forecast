# gribcheck

Pipelines for:
- PM2.5 ingestion and deduplication (`88101` + `88502`)
- HRRR analysis join (`MASSDEN`, `COLMD`) to station-day PM2.5
- Accuracy evaluation by multiple strata
- Wildfire-focused raster dataset generation for `MASSDEN` `+12h/+24h`
- FRP channel from VIIRS (`DL_FIRE_J1*`) rasterized to each HRRR sample AOI/hour

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

## Commands

```bash
gribcheck ingest-pm --config /Users/lorn/Code/gribcheck/config/pipeline_config.yaml
gribcheck build-station-hrrr-daily --config /Users/lorn/Code/gribcheck/config/pipeline_config.yaml
gribcheck evaluate-accuracy --config /Users/lorn/Code/gribcheck/config/pipeline_config.yaml
gribcheck build-wildfire-raster-dataset --config /Users/lorn/Code/gribcheck/config/pipeline_config.yaml
```

For smoke tests on a smaller slice:

```bash
gribcheck build-station-hrrr-daily --config /Users/lorn/Code/gribcheck/config/pipeline_config.yaml --station-limit 25 --max-hours 6
gribcheck build-wildfire-raster-dataset --config /Users/lorn/Code/gribcheck/config/pipeline_config.yaml --max-fires 1 --max-samples 2 --max-hours-per-fire 2
```
