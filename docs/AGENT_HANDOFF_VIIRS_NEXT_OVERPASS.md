# Agent Handoff: Convert `gribcheck` to VIIRS Next-Overpass Fire Forecast Dataset Builder

## 1) Goal
Build a new dataset pipeline focused on **fire behavior forecasting at the next VIIRS overpass**.

Use this repo as the base, but shift targets from HRRR smoke labels to VIIRS-driven labels.

Primary requirements from project owner:
- Keep HRRR inputs, but use fewer covariates than the current smoke pipeline.
- Include as much VIIRS signal as possible from the sensor product.
- Add static/contextual inputs: landcover, soil moisture, soil temperature.
- Add engineered features:
  - cumulative FRP per pixel over fire lifetime up to sample time,
  - historical burn overlap count per pixel.

## 2) Current Repo Map (What Exists)
Core files to reuse:
- `/Users/lorn/Code/gribcheck/src/gribcheck/pipelines/wildfire_raster.py`
  - Resumable Zarr + checkpointed Parquet index writer.
  - Time-major extraction mode (faster) and progress logging.
- `/Users/lorn/Code/gribcheck/src/gribcheck/hrrr.py`
  - HRRR Zarr field reader with caching and missing-run/field short-circuit logic.
- `/Users/lorn/Code/gribcheck/src/gribcheck/viirs.py`
  - VIIRS zip ingestion (`DL_FIRE_J1*`) and rasterization to HRRR projection.
- `/Users/lorn/Code/gribcheck/src/gribcheck/fire.py`
  - Wildfire perimeter streaming and filtering rules.
- `/Users/lorn/Code/gribcheck/src/gribcheck/cli.py`
  - Existing CLI pattern.
- `/Users/lorn/Code/gribcheck/config/pipeline_config.yaml`
  - Current config structure.

Use these patterns instead of rewriting from scratch.

## 3) Important Conceptual Change
Current pipeline target = HRRR `MASSDEN` future fields.

New pipeline target = **VIIRS next-overpass outcome** (gridded target patch).

### Critical caveat
If using only `DL_FIRE_J1*` active-fire detections, you have a **positive-only detection stream** (hotspots only), not explicit “no-fire overpass” coverage. That means:
- You can forecast “next detection overpass behavior” robustly.
- True overpass-with-zero labels requires external swath coverage/non-detection source.

Implement pipeline to support both modes via config:
- `label_mode: next_detection` (default, with current data)
- `label_mode: next_overpass_with_coverage` (future extension when swath coverage source is added)

## 4) New Pipeline to Add
Create a new pipeline module instead of mutating smoke builder heavily:
- Add `/Users/lorn/Code/gribcheck/src/gribcheck/pipelines/viirs_next_overpass.py`
- Add CLI command: `build-viirs-next-overpass-dataset`

Keep the same resumable mechanics used in `wildfire_raster.py`:
- checkpoint jsonl,
- append-only Zarr with alignment checks,
- resume skip by stable `sample_key`,
- periodic flush.

## 5) Data Model (Recommended)

### Sample unit
One sample = one fire-day patch (daily aggregated) or fire-time patch (hourly). Start with daily aggregated for speed.

### Inputs (channels)
1. VIIRS-now channels at sample time window (e.g., 4-hour aggregate from selected UTC hours):
- `viirs_frp_sum_t`
- `viirs_count_t`
- optional VIIRS radiative/quality channels if present in CSV fields

2. VIIRS history channels:
- `viirs_frp_cumulative_from_fire_start`
- `viirs_detect_count_cumulative_from_fire_start`
- `hours_since_last_detection`

3. HRRR channels (reduced list):
- `UGRD_10m`, `VGRD_10m`
- `TMP_2m`, `RH_2m`
- `HPBL_surface`, `PRATE_surface`
- soil fields (verify actual level names in HRRR zarr):
  - `SOILW_*` levels (moisture)
  - `TSOIL_*` levels (soil temp)

4. Static channels:
- landcover class map (integer raster or one-hot planes)
- burn-history overlap count map
- optional topo (if available later): slope/aspect/elevation

### Targets
Default target planes for next VIIRS event:
- `target_next_frp_sum`
- `target_next_detect_count`
- `target_next_any_detect` (binary)

Also save metadata in index:
- `target_time_utc`
- `lead_hours`
- `label_mode`
- `label_coverage_ok` boolean

### Storage
- Primary: Zarr (float16 where appropriate; uint16 for counts if desired)
- Index: Parquet
- Keep 256x256 unless budget requires 192x192

## 6) Config Additions
Extend config dataclasses and yaml sections with a new block, e.g. `viirs_forecast`:
- `enabled: true`
- `sample_hours_utc: [0,6,12,18]`
- `daily_aggregate: true`
- `label_mode: next_detection`
- `max_label_lead_hours: 36`
- `input_variables` (reduced HRRR list)
- `soil_variables` list
- `static_layers` paths:
  - landcover raster path
  - burn history raster path (or burn perimeter source for preprocessing)
- `output_zarr`, `output_index`, `build_log`

Do not break existing smoke config/commands.

## 7) Implementation Steps (Execution Order)

### Step A: Add static layer preprocess utility
Create module:
- `/Users/lorn/Code/gribcheck/src/gribcheck/static_layers.py`

Responsibilities:
- Load landcover source raster.
- Reproject once to HRRR CRS/grid (or lazily sample by AOI).
- Build burn-history overlap raster once:
  - From historical burn perimeters, rasterize cumulative overlap count onto HRRR grid.
- Cache static rasters under `data/intermediate` as Zarr or compressed npy.

### Step B: Upgrade VIIRS ingestion for label building
In `/Users/lorn/Code/gribcheck/src/gribcheck/viirs.py`:
- Preserve existing FRP rasterizer behavior.
- Add helper to build per-fire/per-day VIIRS event timeline:
  - `next_viirs_event_time(sample_time, fire_id, bounds_xy, max_lead_hours)`.
- Add optional cumulative FRP accumulator map builder keyed by `(fire_id, sample_time)`.

### Step C: New pipeline `viirs_next_overpass.py`
Implement with structure mirroring `wildfire_raster.py`:
1. Load filtered fire records.
2. Build day tasks (time-major ordering by number of fires/day).
3. Pre-check required HRRR runs (reuse `HRRRAnalysisReader.run_available`).
4. Preload HRRR fields for the day.
5. Build VIIRS input/history/cumulative maps per task.
6. Find next VIIRS event target time and construct target maps.
7. Extract static layer patches.
8. Write inputs/targets to Zarr and row to checkpoint.
9. Materialize Parquet index from checkpoint at end.

### Step D: CLI wiring
Update `/Users/lorn/Code/gribcheck/src/gribcheck/cli.py`:
- New command `build-viirs-next-overpass-dataset`
- Include same ergonomics:
  - `--workers`
  - `--resume/--no-resume`
  - `--checkpoint-flush-samples`
  - `--time-major`
  - `--verbose-progress`

### Step E: Tests
Add tests similar to wildfire resume/daily tests:
- next-event lookup correctness
- cumulative FRP accumulation monotonicity
- resume skip behavior
- index/zarr alignment after interruption
- static layer extraction shape/projection checks

## 8) Performance Requirements
- Default to time-major mode.
- Parallelize field loads and patch builds.
- Keep missing-run and missing-field caches active.
- Log progress every 10–30s with:
  - days completed,
  - samples written,
  - samples/sec,
  - skipped due to missing runs/labels.

Use benchmark script to compare access paths before major refactor:
- `/Users/lorn/Code/gribcheck/scripts/hrrr_access_benchmark.py`

## 9) Initial Minimal Feature Set (MVP)
For first deliverable, do this and ship quickly:
- Inputs: `VIIRS_FRP_now`, `VIIRS_FRP_cumulative`, `UGRD_10m`, `VGRD_10m`, `TMP_2m`, `RH_2m`, `HPBL`, `PRATE`, `SOILW_top`, `TSOIL_top`, `landcover`, `burn_overlap_count`.
- Target: `target_next_frp_sum` only.
- Daily aggregation with sample hours `0,6,12,18`.
- `label_mode=next_detection`.
- Full resume support.

Then iterate to add detection count/binary targets and multi-satellite enhancements.

## 10) Known Risks / Pitfalls
- VIIRS detections are sparse and skewed; heavy class imbalance expected.
- Missing explicit non-detection overpass labels if using detection-only product.
- Multiple perimeter records may represent same incident; ensure stable `fire_id` dedupe logic.
- Static layer reprojection must match HRRR axes orientation and extent.
- Be careful not to exceed disk budget when adding static + history channels.

## 11) Done Criteria
Pipeline is “done” when all are true:
- `build-viirs-next-overpass-dataset` runs end-to-end with resume.
- Produces Zarr + Parquet index with documented schema.
- Logs throughput and missing-label stats.
- Unit tests and one integration smoke test pass.
- Build log includes sample count, channel list, skip counts, effective label lead distribution.

## 12) Suggested First Run Command
After implementation:

```bash
/Users/lorn/Code/gribcheck/.venv/bin/gribcheck \
  --config /Users/lorn/Code/gribcheck/config/pipeline_config.yaml \
  build-viirs-next-overpass-dataset \
  --workers 8 \
  --time-major \
  --resume \
  --verbose-progress \
  --progress-interval-seconds 10
```

If you need a fast smoke run, add:
- `--max-fires 5`
- `--max-samples 20`
