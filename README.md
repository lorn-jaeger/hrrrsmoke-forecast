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

## Notebook Git-Friendly Staging

This repo includes a notebook clean filter so `.ipynb` files are normalized for git:
- strips code cell outputs
- resets `execution_count` to `null`

The filter runs when files are staged, so your local working notebook can still keep outputs.

One-time setup per clone:

```bash
/Users/lorn/Code/gribcheck/scripts/setup_notebook_git_filter.sh
```

Optional manual cleanup:

```bash
python3 /Users/lorn/Code/gribcheck/scripts/ipynb_clean_filter.py --in-place /path/to/notebook.ipynb
```

For smoke tests on a smaller slice:

```bash
gribcheck build-station-hrrr-daily --config /Users/lorn/Code/gribcheck/config/pipeline_config.yaml --station-limit 25 --max-hours 6
gribcheck build-wildfire-raster-dataset --config /Users/lorn/Code/gribcheck/config/pipeline_config.yaml --max-fires 1 --max-samples 2 --max-hours-per-fire 2
```

## Access Throughput Benchmark

Use the benchmark script to compare HRRR access patterns (Zarr full/subset/multi-var/multi-time and GRIB multi-source HTTP range/full downloads):

```bash
python /Users/lorn/Code/gribcheck/scripts/hrrr_access_benchmark.py \
  --run-time 2024-07-15T00:00:00Z \
  --repeats 2 \
  --patterns all \
  --workers 8 \
  --output-json /Users/lorn/Code/gribcheck/reports/hrrr_access_benchmark.json
```

Quick network-only benchmark:

```bash
python /Users/lorn/Code/gribcheck/scripts/hrrr_access_benchmark.py \
  --patterns grib_range,grib_range_parallel \
  --grib-range-mib 64
```
