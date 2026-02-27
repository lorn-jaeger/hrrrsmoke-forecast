#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import shapefile  # pyshp
from shapely.geometry import shape

from scripts.mtbs_wfigs_coherence_analysis import (
    FireIncident,
    add_pairwise_differences,
    compute_detection_metrics,
    load_mtbs_fires,
    load_wfigs_fires,
    match_mtbs_wfigs,
    repair_geometry,
    summarize_matches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare MTBS/GlobFire to WFIGS with custom date rules "
            "(window +/- days, MTBS end cap)."
        )
    )
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
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--globfire-year", type=int, default=2021)
    parser.add_argument("--min-acres", type=float, default=1000.0)
    parser.add_argument("--window-pad-days", type=int, default=30)
    parser.add_argument("--event-pad-days", type=int, default=4)
    parser.add_argument("--mtbs-end-cap-days", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/lorn/Code/gribcheck/reports/wfigs_mtbs_globfire_custom_rules"),
    )
    return parser.parse_args()


def load_globfire_us_fires(path: Path, year: int, min_acres: float) -> list[FireIncident]:
    # Broad US + AK + HI bbox to align with MTBS/WFIGS footprint.
    us_bbox = (-170.0, 18.0, -65.0, 72.0)
    geod = pyproj.Geod(ellps="WGS84")

    reader = shapefile.Reader(str(path))
    fields = [f[0] for f in reader.fields[1:]]
    idx = {name: i for i, name in enumerate(fields)}

    out: list[FireIncident] = []
    for sr in reader.iterShapeRecords():
        xmin, ymin, xmax, ymax = sr.shape.bbox
        if xmax < us_bbox[0] or xmin > us_bbox[2] or ymax < us_bbox[1] or ymin > us_bbox[3]:
            continue

        rec = sr.record
        start_d = rec[idx["IDate"]]
        end_d = rec[idx["FDate"]]
        if start_d is None or start_d.year != year:
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
        out.append(
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

    out.sort(key=lambda r: (r.start_date, -r.size_acres, r.fire_id))
    return out


def apply_custom_windows(
    fires: list[FireIncident],
    event_pad_days: int,
    mtbs_end_cap_days: int,
) -> list[FireIncident]:
    padded: list[FireIncident] = []
    for fire in fires:
        start_raw = fire.start_date
        end_raw = fire.end_date or fire.start_date

        # User rule: cap MTBS end at start+30 days regardless of proxy end date.
        if fire.dataset.upper() == "MTBS":
            end_cap = start_raw + timedelta(days=mtbs_end_cap_days)
            end_raw = min(end_raw, end_cap)
            if end_raw < start_raw:
                end_raw = start_raw

        # User rule: +/- event pad applies only to MTBS and GlobFire, not WFIGS.
        if fire.dataset.upper() in {"MTBS", "GLOBFIRE"}:
            start_adj = start_raw - timedelta(days=event_pad_days)
            end_adj = end_raw + timedelta(days=event_pad_days)
        else:
            start_adj = start_raw
            end_adj = end_raw

        padded.append(
            FireIncident(
                dataset=fire.dataset,
                fire_id=fire.fire_id,
                irwin_id=fire.irwin_id,
                name=fire.name,
                state=fire.state,
                start_date=start_adj,
                end_date=end_adj,
                end_date_is_proxy=fire.end_date_is_proxy,
                size_acres=fire.size_acres,
                geometry=fire.geometry,
            )
        )
    return padded


def dataset_row(summary: dict, df: pd.DataFrame) -> dict:
    return {
        "dataset": summary["dataset"],
        "fire_count": int(summary["fire_count"]),
        "mean_size_acres": float(summary["mean_size_acres"]),
        "median_size_acres": float(summary["median_size_acres"]),
        "mean_active_window_days": float(summary["mean_active_window_days"]),
        "in_window_share_total": float(summary["in_window_detection_share_total"]),
        "noise_share_total": float(summary["noise_share_total"]),
        "total_in_window_detections": int(summary["total_in_window_detections"]),
        "total_pre_start_detections": int(summary["total_pre_start_detections"]),
        "total_post_end_detections": int(summary["total_post_end_detections"]),
        "pct_active_days_with_any_detection_weighted": float(
            df["days_with_any_detection_in_window"].sum() / max(df["active_window_days"].sum(), 1)
        ),
        "pct_active_days_with_any_hn_weighted": float(summary["pct_active_days_with_any_hn_weighted"]),
        "pct_active_days_few_hn_le3_weighted": float(summary["pct_active_days_few_hn_le3_weighted"]),
        "median_in_window_detection_span_frac": float(df["in_window_detection_span_frac"].median(skipna=True)),
        "median_in_window_peak_relative_position": float(df["in_window_peak_relative_position"].median(skipna=True)),
    }


def to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.4f}"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def write_report(
    output_path: Path,
    mtbs_wfigs_df: pd.DataFrame,
    globfire_wfigs_df: pd.DataFrame,
    mtbs_wfigs_match_summary: dict,
    rules: dict,
) -> None:
    lines: list[str] = []
    lines.append("# WFIGS vs MTBS and GlobFire (Custom Rules)")
    lines.append("")
    lines.append("## Rules Applied")
    lines.append(f"- Year filter for MTBS/WFIGS: {rules['start_year']}-01-01 to {rules['end_year']}-12-31")
    lines.append(f"- GlobFire comparison year: {rules['globfire_year']}")
    lines.append(f"- Minimum fire size: >= {rules['min_acres']:g} acres")
    lines.append(
        f"- Event-window pad: +/- {rules['event_pad_days']} days for MTBS and GlobFire only (WFIGS unchanged)"
    )
    lines.append(f"- Noise pad for pre/post counting: +/- {rules['window_pad_days']} days")
    lines.append(f"- MTBS end cap: start + {rules['mtbs_end_cap_days']} days (before +/- event pad)")
    lines.append("")

    lines.append("## MTBS vs WFIGS (2021-2023)")
    lines.append(to_markdown_table(mtbs_wfigs_df))
    lines.append("")

    lines.append("## WFIGS vs GlobFire (2021)")
    lines.append(to_markdown_table(globfire_wfigs_df))
    lines.append("")

    lines.append("## Shared Fire Footprints (MTBS vs WFIGS)")
    lines.append(f"- Matched pairs: {mtbs_wfigs_match_summary.get('matched_pair_count', 0)}")
    lines.append(f"- MTBS matched: {100.0 * mtbs_wfigs_match_summary.get('pct_mtbs_matched', 0.0):.1f}%")
    lines.append(f"- WFIGS matched: {100.0 * mtbs_wfigs_match_summary.get('pct_wfigs_matched', 0.0):.1f}%")
    if mtbs_wfigs_match_summary.get("matched_pair_count", 0) > 0:
        lines.append(f"- Median overlap ratio: {mtbs_wfigs_match_summary.get('median_overlap_ratio', float('nan')):.3f}")
        lines.append(
            f"- Median |start-date diff| (WFIGS - MTBS): {mtbs_wfigs_match_summary.get('median_abs_start_date_diff_days', float('nan')):.1f} days"
        )
        lines.append(
            f"- Median absolute size difference: {100.0 * mtbs_wfigs_match_summary.get('median_abs_size_pct_diff', float('nan')):.1f}%"
        )
    lines.append("")

    lines.append("## Coverage Caveat")
    lines.append("- Local GlobFire source provides yearly file for 2021 only in this workspace.")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MTBS/WFIGS raw cohorts...")
    mtbs_raw = load_mtbs_fires(args.mtbs_zip, args.start_year, args.end_year, args.min_acres)
    wfigs_raw = load_wfigs_fires(args.wfigs_geojson, args.start_year, args.end_year, args.min_acres)
    print(f"MTBS raw fires: {len(mtbs_raw):,}")
    print(f"WFIGS raw fires: {len(wfigs_raw):,}")

    print("Applying custom date rules (event pad and MTBS end cap)...")
    mtbs_adj = apply_custom_windows(mtbs_raw, args.event_pad_days, args.mtbs_end_cap_days)
    wfigs_adj = apply_custom_windows(wfigs_raw, args.event_pad_days, args.mtbs_end_cap_days)

    print("Computing MTBS metrics with custom windows...")
    mtbs_df, mtbs_summary, mtbs_profile = compute_detection_metrics(
        fires=mtbs_adj,
        viirs_zip=args.viirs_zip,
        window_pad_days=args.window_pad_days,
        start_year=args.start_year,
        end_year=args.end_year,
        label="MTBS",
    )

    print("Computing WFIGS metrics with custom windows...")
    wfigs_df, wfigs_summary, wfigs_profile = compute_detection_metrics(
        fires=wfigs_adj,
        viirs_zip=args.viirs_zip,
        window_pad_days=args.window_pad_days,
        start_year=args.start_year,
        end_year=args.end_year,
        label="WFIGS",
    )

    print("Matching shared MTBS/WFIGS incidents...")
    match_seed = match_mtbs_wfigs(mtbs_fires=mtbs_raw, wfigs_fires=wfigs_raw)
    mtbs_wfigs_pairs = add_pairwise_differences(match_seed, mtbs_fires=mtbs_raw, wfigs_fires=wfigs_raw)
    mtbs_wfigs_match_summary = summarize_matches(mtbs_wfigs_pairs, len(mtbs_raw), len(wfigs_raw))

    print("Loading GlobFire 2021 and WFIGS 2021 cohort...")
    globfire_raw = load_globfire_us_fires(args.globfire_zip, args.globfire_year, args.min_acres)
    wfigs_2021_raw = load_wfigs_fires(args.wfigs_geojson, args.globfire_year, args.globfire_year, args.min_acres)
    print(f"GlobFire {args.globfire_year} fires: {len(globfire_raw):,}")
    print(f"WFIGS {args.globfire_year} fires: {len(wfigs_2021_raw):,}")

    globfire_adj = apply_custom_windows(globfire_raw, args.event_pad_days, args.mtbs_end_cap_days)
    wfigs_2021_adj = apply_custom_windows(wfigs_2021_raw, args.event_pad_days, args.mtbs_end_cap_days)

    print("Computing GlobFire metrics with custom windows...")
    globfire_df, globfire_summary, globfire_profile = compute_detection_metrics(
        fires=globfire_adj,
        viirs_zip=args.viirs_zip,
        window_pad_days=args.window_pad_days,
        start_year=args.globfire_year,
        end_year=args.globfire_year,
        label="GlobFire",
    )

    print("Computing WFIGS(2021) metrics with custom windows...")
    wfigs_2021_df, wfigs_2021_summary, wfigs_2021_profile = compute_detection_metrics(
        fires=wfigs_2021_adj,
        viirs_zip=args.viirs_zip,
        window_pad_days=args.window_pad_days,
        start_year=args.globfire_year,
        end_year=args.globfire_year,
        label="WFIGS_2021",
    )

    mtbs_wfigs_summary_df = pd.DataFrame(
        [
            dataset_row(mtbs_summary, mtbs_df),
            dataset_row(wfigs_summary, wfigs_df),
        ]
    )
    globfire_wfigs_summary_df = pd.DataFrame(
        [
            dataset_row(globfire_summary, globfire_df),
            dataset_row(wfigs_2021_summary, wfigs_2021_df),
        ]
    )

    all_summary = {
        "rules": {
            "start_year": args.start_year,
            "end_year": args.end_year,
            "globfire_year": args.globfire_year,
            "min_acres": args.min_acres,
            "event_pad_days": args.event_pad_days,
            "window_pad_days": args.window_pad_days,
            "mtbs_end_cap_days": args.mtbs_end_cap_days,
        },
        "mtbs_vs_wfigs_2021_2023": {
            "MTBS": mtbs_summary,
            "WFIGS": wfigs_summary,
            "match_summary": mtbs_wfigs_match_summary,
        },
        "globfire_vs_wfigs_2021": {
            "GlobFire": globfire_summary,
            "WFIGS_2021": wfigs_2021_summary,
        },
    }

    print("Writing outputs...")
    mtbs_df.to_csv(out_dir / "mtbs_2021_2023_custom_fire_metrics.csv", index=False)
    wfigs_df.to_csv(out_dir / "wfigs_2021_2023_custom_fire_metrics.csv", index=False)
    globfire_df.to_csv(out_dir / "globfire_2021_custom_fire_metrics.csv", index=False)
    wfigs_2021_df.to_csv(out_dir / "wfigs_2021_custom_fire_metrics.csv", index=False)

    pd.concat([mtbs_profile, wfigs_profile], ignore_index=True).to_csv(
        out_dir / "mtbs_wfigs_2021_2023_relative_profile_hn_custom.csv", index=False
    )
    pd.concat([globfire_profile, wfigs_2021_profile], ignore_index=True).to_csv(
        out_dir / "globfire_wfigs_2021_relative_profile_hn_custom.csv", index=False
    )

    mtbs_wfigs_summary_df.to_csv(out_dir / "summary_mtbs_wfigs_2021_2023_custom.csv", index=False)
    globfire_wfigs_summary_df.to_csv(out_dir / "summary_globfire_wfigs_2021_custom.csv", index=False)
    mtbs_wfigs_pairs.to_csv(out_dir / "mtbs_wfigs_matched_pairs_2021_2023.csv", index=False)

    with (out_dir / "summary_custom_rules.json").open("w", encoding="utf-8") as fp:
        json.dump(all_summary, fp, indent=2)

    write_report(
        output_path=out_dir / "custom_rules_report.md",
        mtbs_wfigs_df=mtbs_wfigs_summary_df,
        globfire_wfigs_df=globfire_wfigs_summary_df,
        mtbs_wfigs_match_summary=mtbs_wfigs_match_summary,
        rules=all_summary["rules"],
    )

    print("Done.")
    print(f"Report: {out_dir / 'custom_rules_report.md'}")


if __name__ == "__main__":
    main()
