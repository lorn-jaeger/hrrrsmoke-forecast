#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from shapely import points as shapely_points
from shapely.strtree import STRtree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.mtbs_wfigs_coherence_analysis import DAY0, FireIncident, day_int_from_date, load_mtbs_fires, load_wfigs_fires


@dataclass
class FireAddonMetrics:
    total_window_detections: int = 0
    in_window_detections: int = 0
    pre_start_detections: int = 0
    post_end_detections: int = 0
    in_window_days_any: set[int] = field(default_factory=set)
    in_window_frp_sum: float = 0.0
    in_window_frp_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add-on metrics: any-confidence active detection percentages + FRP-over-time stats"
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
        "--viirs-zip",
        type=Path,
        default=Path("/Users/lorn/Downloads/DL_FIRE_J1V-C2_718831.zip"),
    )
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--min-acres", type=float, default=1000.0)
    parser.add_argument("--window-pad-days", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/lorn/Code/gribcheck/reports/mtbs_wfigs_coherence_2021_2023_addons"),
    )
    return parser.parse_args()


def iterate_viirs_chunks_with_frp(
    viirs_zip: Path,
    min_day_int: int,
    max_day_int: int,
    chunksize: int = 250_000,
):
    usecols = ["latitude", "longitude", "acq_date", "frp"]
    with zipfile.ZipFile(viirs_zip) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv") and "archive" in n.lower()]
        if not members:
            raise FileNotFoundError(f"No archive CSV found in {viirs_zip}")
        for member in members:
            with zf.open(member) as fp:
                for chunk in pd.read_csv(fp, usecols=usecols, chunksize=chunksize, low_memory=False):
                    day_vals = pd.to_datetime(chunk["acq_date"], errors="coerce").to_numpy(dtype="datetime64[D]")
                    lat_vals = pd.to_numeric(chunk["latitude"], errors="coerce").to_numpy(dtype=np.float64)
                    lon_vals = pd.to_numeric(chunk["longitude"], errors="coerce").to_numpy(dtype=np.float64)
                    frp_vals = pd.to_numeric(chunk["frp"], errors="coerce").to_numpy(dtype=np.float64)

                    valid = ~np.isnat(day_vals)
                    valid &= np.isfinite(lat_vals) & np.isfinite(lon_vals) & np.isfinite(frp_vals)
                    valid &= (lat_vals >= -90) & (lat_vals <= 90) & (lon_vals >= -180) & (lon_vals <= 180)
                    valid &= frp_vals >= 0.0
                    if not np.any(valid):
                        continue

                    day_int = (day_vals[valid] - DAY0).astype(np.int32)
                    keep = (day_int >= min_day_int) & (day_int <= max_day_int)
                    if not np.any(keep):
                        continue

                    yield (
                        lon_vals[valid][keep],
                        lat_vals[valid][keep],
                        day_int[keep],
                        frp_vals[valid][keep],
                    )


def compute_anyconf_and_frp(
    fires: list[FireIncident],
    viirs_zip: Path,
    start_year: int,
    end_year: int,
    window_pad_days: int,
    label: str,
):
    n = len(fires)
    geoms = [f.geometry for f in fires]
    tree = STRtree(geoms)

    start_day = np.array([day_int_from_date(f.start_date) for f in fires], dtype=np.int32)
    end_day = np.array([day_int_from_date(f.end_date or f.start_date) for f in fires], dtype=np.int32)
    active_days = np.maximum(end_day - start_day + 1, 1).astype(np.int32)
    duration_days = active_days.copy()
    window_start = start_day - int(window_pad_days)
    window_end = end_day + int(window_pad_days)

    viirs_min = day_int_from_date(date(start_year, 1, 1) - timedelta(days=window_pad_days + 2))
    viirs_max = day_int_from_date(date(end_year, 12, 31) + timedelta(days=450))

    fire_metrics = [FireAddonMetrics() for _ in fires]
    year_frp_values: defaultdict[int, list[float]] = defaultdict(list)
    rel_bin_frp_values: list[list[float]] = [[] for _ in range(10)]
    chunk_i = 0
    total_assigned = 0

    for lon_vals, lat_vals, day_vals, frp_vals in iterate_viirs_chunks_with_frp(
        viirs_zip=viirs_zip,
        min_day_int=viirs_min,
        max_day_int=viirs_max,
    ):
        chunk_i += 1
        pts = shapely_points(lon_vals, lat_vals)
        pairs = tree.query(pts, predicate="within")
        if pairs.shape[1] == 0:
            continue

        p_idx = pairs[0].astype(np.int32)
        f_idx = pairs[1].astype(np.int32)
        d = day_vals[p_idx]

        keep = (d >= window_start[f_idx]) & (d <= window_end[f_idx])
        if not np.any(keep):
            continue

        p_idx = p_idx[keep]
        f_idx = f_idx[keep]
        d = d[keep]

        s = start_day[f_idx]
        e = end_day[f_idx]
        pre = d < s
        post = d > e
        off = np.where(pre, s - d, np.where(post, d - e, 0))
        score = off.astype(np.int64) * 10_000 + np.minimum(duration_days[f_idx], 9_999).astype(np.int64)

        order = np.lexsort((f_idx, score, p_idx))
        p_sorted = p_idx[order]
        f_sorted = f_idx[order]
        d_sorted = d[order]
        first_mask = np.r_[True, p_sorted[1:] != p_sorted[:-1]]

        p_sel = p_sorted[first_mask]
        f_sel = f_sorted[first_mask]
        d_sel = d_sorted[first_mask]
        frp_sel = frp_vals[p_sel]
        total_assigned += len(p_sel)

        years_sel = (DAY0 + d_sel.astype("timedelta64[D]")).astype("datetime64[Y]").astype(int) + 1970

        for fi, di, frpi, yi in zip(f_sel, d_sel, frp_sel, years_sel, strict=True):
            fire_i = int(fi)
            day_i = int(di)
            frp = float(frpi)
            yr = int(yi)

            m = fire_metrics[fire_i]
            m.total_window_detections += 1

            s_i = int(start_day[fire_i])
            e_i = int(end_day[fire_i])
            if day_i < s_i:
                m.pre_start_detections += 1
                continue
            if day_i > e_i:
                m.post_end_detections += 1
                continue

            m.in_window_detections += 1
            m.in_window_days_any.add(day_i)
            m.in_window_frp_sum += frp
            m.in_window_frp_count += 1
            year_frp_values[yr].append(frp)

            denom = max(int(active_days[fire_i]) - 1, 1)
            rel = (day_i - s_i) / denom
            bin_i = int(np.clip(np.floor(rel * 10.0), 0, 9))
            rel_bin_frp_values[bin_i].append(frp)

        if chunk_i % 10 == 0:
            print(f"[{label}] chunk {chunk_i}, assigned detections so far: {total_assigned:,}")

    rows = []
    for i, fire in enumerate(fires):
        m = fire_metrics[i]
        awd = int(active_days[i])
        rows.append(
            {
                "dataset": fire.dataset,
                "fire_id": fire.fire_id,
                "start_date": fire.start_date,
                "end_date": fire.end_date or fire.start_date,
                "size_acres": fire.size_acres,
                "active_window_days": awd,
                "total_window_detections_anyconf": m.total_window_detections,
                "in_window_detections_anyconf": m.in_window_detections,
                "pre_start_detections_anyconf": m.pre_start_detections,
                "post_end_detections_anyconf": m.post_end_detections,
                "days_with_any_detection_in_window_anyconf": len(m.in_window_days_any),
                "pct_active_days_with_any_detection_anyconf": (len(m.in_window_days_any) / awd) if awd > 0 else np.nan,
                "mean_in_window_frp": (m.in_window_frp_sum / m.in_window_frp_count) if m.in_window_frp_count > 0 else np.nan,
            }
        )

    per_fire_df = pd.DataFrame(rows)
    total_window = int(per_fire_df["total_window_detections_anyconf"].sum())
    in_window = int(per_fire_df["in_window_detections_anyconf"].sum())
    pre = int(per_fire_df["pre_start_detections_anyconf"].sum())
    post = int(per_fire_df["post_end_detections_anyconf"].sum())

    summary = {
        "dataset": label,
        "fire_count": int(len(per_fire_df)),
        "weighted_pct_active_days_with_any_detection_anyconf": float(
            per_fire_df["days_with_any_detection_in_window_anyconf"].sum()
            / max(per_fire_df["active_window_days"].sum(), 1)
        ),
        "in_window_detection_share_total_anyconf": float(in_window / max(total_window, 1)),
        "noise_share_total_anyconf": float((pre + post) / max(total_window, 1)),
        "total_window_detections_anyconf": total_window,
        "in_window_detections_anyconf": in_window,
        "pre_start_detections_anyconf": pre,
        "post_end_detections_anyconf": post,
        "mean_frp_in_window_all_detections": float(
            per_fire_df["mean_in_window_frp"].mean(skipna=True)
        ),
    }

    year_rows = []
    for year in sorted(year_frp_values):
        vals = np.asarray(year_frp_values[year], dtype=np.float64)
        year_rows.append(
            {
                "dataset": label,
                "year": int(year),
                "in_window_detection_count": int(vals.size),
                "mean_frp": float(np.mean(vals)),
                "median_frp": float(np.median(vals)),
                "p90_frp": float(np.quantile(vals, 0.9)),
            }
        )
    year_df = pd.DataFrame(year_rows)

    rel_rows = []
    for i, vals_list in enumerate(rel_bin_frp_values):
        vals = np.asarray(vals_list, dtype=np.float64)
        rel_rows.append(
            {
                "dataset": label,
                "bin_index": i,
                "bin_start_rel": i / 10.0,
                "bin_end_rel": (i + 1) / 10.0,
                "detection_count": int(vals.size),
                "mean_frp": float(np.mean(vals)) if vals.size > 0 else np.nan,
                "median_frp": float(np.median(vals)) if vals.size > 0 else np.nan,
                "p90_frp": float(np.quantile(vals, 0.9)) if vals.size > 0 else np.nan,
            }
        )
    rel_df = pd.DataFrame(rel_rows)

    return per_fire_df, summary, year_df, rel_df


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MTBS and WFIGS fires...")
    mtbs_fires = load_mtbs_fires(args.mtbs_zip, args.start_year, args.end_year, args.min_acres)
    wfigs_fires = load_wfigs_fires(args.wfigs_geojson, args.start_year, args.end_year, args.min_acres)
    print(f"MTBS fires: {len(mtbs_fires):,}")
    print(f"WFIGS fires: {len(wfigs_fires):,}")

    print("Computing add-on metrics for MTBS...")
    mtbs_fire_df, mtbs_summary, mtbs_year_df, mtbs_rel_df = compute_anyconf_and_frp(
        fires=mtbs_fires,
        viirs_zip=args.viirs_zip,
        start_year=args.start_year,
        end_year=args.end_year,
        window_pad_days=args.window_pad_days,
        label="MTBS",
    )

    print("Computing add-on metrics for WFIGS...")
    wfigs_fire_df, wfigs_summary, wfigs_year_df, wfigs_rel_df = compute_anyconf_and_frp(
        fires=wfigs_fires,
        viirs_zip=args.viirs_zip,
        start_year=args.start_year,
        end_year=args.end_year,
        window_pad_days=args.window_pad_days,
        label="WFIGS",
    )

    summary_df = pd.DataFrame([mtbs_summary, wfigs_summary])
    year_df = pd.concat([mtbs_year_df, wfigs_year_df], ignore_index=True)
    rel_df = pd.concat([mtbs_rel_df, wfigs_rel_df], ignore_index=True)

    mtbs_fire_df.to_csv(out_dir / "mtbs_anyconf_frp_fire_metrics.csv", index=False)
    wfigs_fire_df.to_csv(out_dir / "wfigs_anyconf_frp_fire_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "summary_anyconf_frp.csv", index=False)
    year_df.to_csv(out_dir / "yearly_frp_anyconf.csv", index=False)
    rel_df.to_csv(out_dir / "relative_window_frp_anyconf.csv", index=False)

    with (out_dir / "summary_anyconf_frp.json").open("w", encoding="utf-8") as fp:
        json.dump({"MTBS": mtbs_summary, "WFIGS": wfigs_summary}, fp, indent=2)

    print("Done.")
    print(f"Summary CSV: {out_dir / 'summary_anyconf_frp.csv'}")


if __name__ == "__main__":
    main()
