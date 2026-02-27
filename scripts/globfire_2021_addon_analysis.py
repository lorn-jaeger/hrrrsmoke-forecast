#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd
import pyproj
import shapefile  # pyshp
from shapely.geometry import shape

from scripts.mtbs_wfigs_coherence_analysis import (
    FireIncident,
    compute_detection_metrics,
    load_mtbs_fires,
    load_wfigs_fires,
    repair_geometry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add GlobFire to 2021 coherence comparison (MTBS/WFIGS/GlobFire)")
    parser.add_argument(
        "--wfigs-geojson",
        type=Path,
        default=Path("/Users/lorn/Downloads/WFIGS_Interagency_Perimeters_6781836551080060975.geojson"),
    )
    parser.add_argument(
        "--mtbs-zip",
        type=Path,
        default=Path("/Users/lorn/Code/gribcheck/data/external/mtbs_perimeter_data.zip"),
    )
    parser.add_argument(
        "--globfire-zip",
        type=Path,
        default=Path("/Users/lorn/Code/gribcheck/data/external/Final_GlobFirev3_GWIS_MCD64A1__2021.zip"),
    )
    parser.add_argument(
        "--viirs-zip",
        type=Path,
        default=Path("/Users/lorn/Downloads/DL_FIRE_J1V-C2_718831.zip"),
    )
    parser.add_argument("--min-acres", type=float, default=1000.0)
    parser.add_argument("--window-pad-days", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/lorn/Code/gribcheck/reports/mtbs_wfigs_globfire_2021"),
    )
    return parser.parse_args()


def load_globfire_us_2021_fires(path: Path, min_acres: float) -> list[FireIncident]:
    # Broad US + AK + HI envelope to align with MTBS/WFIGS U.S.-focused scope.
    us_bbox = (-170.0, 18.0, -65.0, 72.0)
    reader = shapefile.Reader(str(path))
    fields = [f[0] for f in reader.fields[1:]]
    idx = {name: i for i, name in enumerate(fields)}
    geod = pyproj.Geod(ellps="WGS84")

    fires: list[FireIncident] = []
    for sr in reader.iterShapeRecords():
        xmin, ymin, xmax, ymax = sr.shape.bbox
        if xmax < us_bbox[0] or xmin > us_bbox[2] or ymax < us_bbox[1] or ymin > us_bbox[3]:
            continue

        rec = sr.record
        start_d = rec[idx["IDate"]]
        end_d = rec[idx["FDate"]]
        if start_d is None:
            continue
        if start_d.year != 2021:
            continue
        if end_d is None or end_d < start_d:
            end_d = start_d

        geom = repair_geometry(shape(sr.shape.__geo_interface__))
        if geom is None:
            continue

        area_m2, _ = geod.geometry_area_perimeter(geom)
        area_acres = abs(area_m2) / 4046.8564224
        if area_acres < min_acres:
            continue

        fire_id = str(rec[idx["Id"]])
        fires.append(
            FireIncident(
                dataset="GlobFire",
                fire_id=fire_id,
                irwin_id=None,
                name=f"GlobFire_{fire_id}",
                state=None,
                start_date=start_d,
                end_date=end_d,
                end_date_is_proxy=False,
                size_acres=float(area_acres),
                geometry=geom,
            )
        )

    fires.sort(key=lambda r: (r.start_date, -r.size_acres, r.fire_id))
    return fires


def summarize_only(summary: dict) -> dict:
    keep = [
        "dataset",
        "fire_count",
        "mean_size_acres",
        "median_size_acres",
        "mean_active_window_days",
        "in_window_detection_share_total",
        "noise_share_total",
        "total_in_window_detections",
        "total_pre_start_detections",
        "total_post_end_detections",
        "pct_active_days_with_any_hn_weighted",
        "pct_active_days_few_hn_le3_weighted",
    ]
    return {k: summary[k] for k in keep}


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading 2021 MTBS/WFIGS...")
    mtbs_fires = load_mtbs_fires(args.mtbs_zip, start_year=2021, end_year=2021, min_acres=args.min_acres)
    wfigs_fires = load_wfigs_fires(args.wfigs_geojson, start_year=2021, end_year=2021, min_acres=args.min_acres)
    print(f"MTBS fires: {len(mtbs_fires):,}")
    print(f"WFIGS fires: {len(wfigs_fires):,}")

    print("Loading 2021 GlobFire (US envelope, geodesic area filter)...")
    globfire_fires = load_globfire_us_2021_fires(args.globfire_zip, min_acres=args.min_acres)
    print(f"GlobFire fires: {len(globfire_fires):,}")

    print("Scoring VIIRS coherence for WFIGS...")
    wfigs_df, wfigs_summary, _ = compute_detection_metrics(
        fires=wfigs_fires,
        viirs_zip=args.viirs_zip,
        window_pad_days=args.window_pad_days,
        start_year=2021,
        end_year=2021,
        label="WFIGS-2021",
    )

    print("Scoring VIIRS coherence for MTBS...")
    mtbs_df, mtbs_summary, _ = compute_detection_metrics(
        fires=mtbs_fires,
        viirs_zip=args.viirs_zip,
        window_pad_days=args.window_pad_days,
        start_year=2021,
        end_year=2021,
        label="MTBS-2021",
    )

    print("Scoring VIIRS coherence for GlobFire...")
    globfire_df, globfire_summary, _ = compute_detection_metrics(
        fires=globfire_fires,
        viirs_zip=args.viirs_zip,
        window_pad_days=args.window_pad_days,
        start_year=2021,
        end_year=2021,
        label="GlobFire-2021",
    )

    print("Writing outputs...")
    mtbs_df.to_csv(out_dir / "mtbs_2021_fire_metrics.csv", index=False)
    wfigs_df.to_csv(out_dir / "wfigs_2021_fire_metrics.csv", index=False)
    globfire_df.to_csv(out_dir / "globfire_2021_fire_metrics.csv", index=False)

    summary_rows = [
        summarize_only(mtbs_summary),
        summarize_only(wfigs_summary),
        summarize_only(globfire_summary),
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary_2021_mtbs_wfigs_globfire.csv", index=False)
    with (out_dir / "summary_2021_mtbs_wfigs_globfire.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "MTBS_2021": mtbs_summary,
                "WFIGS_2021": wfigs_summary,
                "GlobFire_2021": globfire_summary,
                "globfire_coverage_note": "Dataset release provides yearly files through 2021 only.",
            },
            fp,
            indent=2,
        )

    print("Done.")
    print(f"Summary CSV: {out_dir / 'summary_2021_mtbs_wfigs_globfire.csv'}")


if __name__ == "__main__":
    main()
