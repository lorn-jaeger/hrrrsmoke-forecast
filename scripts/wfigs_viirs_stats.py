#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import ijson
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import points as shapely_points
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.mtbs_wfigs_coherence_analysis import (  # noqa: E402
    DAY0,
    choose_wfigs_end,
    day_int_from_date,
    parse_wfigs_dt,
    repair_geometry,
    to_float,
)


@dataclass
class WfigsFire:
    fire_id: str
    name: str
    state: str
    start_date: date
    end_date: date
    size_acres: float
    geometry: BaseGeometry
    cause_general: str
    cause_specific: str
    incident_kind: str
    fuel_group: str


def clean_text(value, default: str = "Unknown") -> str:
    text = str(value or "").strip()
    return text if text else default


def size_bucket(acres: float) -> str:
    if acres < 5000:
        return "1k-5k"
    if acres < 20000:
        return "5k-20k"
    if acres < 100000:
        return "20k-100k"
    return "100k+"


def load_wfigs_fires(
    path: Path,
    start_year: int,
    end_year: int,
    min_acres: float,
) -> list[WfigsFire]:
    start_limit = date(start_year, 1, 1)
    end_limit = date(end_year, 12, 31)
    by_id: dict[str, WfigsFire] = {}

    with path.open("rb") as fp:
        for feature in ijson.items(fp, "features.item"):
            props = feature.get("properties", {})
            if clean_text(props.get("attr_IncidentTypeCategory"), "") != "WF":
                continue

            start_dt = parse_wfigs_dt(props.get("attr_FireDiscoveryDateTime"))
            if start_dt is None:
                continue
            start_d = start_dt.date()
            if start_d < start_limit or start_d > end_limit:
                continue

            end_dt = choose_wfigs_end(props)
            end_d = end_dt.date() if end_dt is not None else start_d
            if end_d < start_d:
                end_d = start_d

            size = to_float(props.get("attr_IncidentSize"))
            if size is None or size <= 0:
                size = to_float(props.get("poly_GISAcres"))
            if size is None or size < min_acres:
                continue

            fire_id = (
                clean_text(props.get("attr_UniqueFireIdentifier"), "")
                or clean_text(props.get("poly_IRWINID"), "")
                or clean_text(props.get("attr_LocalIncidentIdentifier"), "")
            )
            if fire_id == "":
                continue

            geom_obj = feature.get("geometry")
            if geom_obj is None:
                continue
            geom = repair_geometry(shape(geom_obj))
            if geom is None:
                continue

            rec = WfigsFire(
                fire_id=fire_id,
                name=clean_text(props.get("attr_IncidentName") or props.get("poly_IncidentName")),
                state=clean_text(props.get("attr_POOState")),
                start_date=start_d,
                end_date=end_d,
                size_acres=float(size),
                geometry=geom,
                cause_general=clean_text(props.get("attr_FireCauseGeneral")),
                cause_specific=clean_text(props.get("attr_FireCauseSpecific")),
                incident_kind=clean_text(props.get("attr_IncidentTypeKind")),
                fuel_group=clean_text(props.get("attr_PredominantFuelGroup")),
            )

            prev = by_id.get(fire_id)
            if prev is None:
                by_id[fire_id] = rec
                continue

            by_id[fire_id] = WfigsFire(
                fire_id=fire_id,
                name=(rec.name if len(rec.name) >= len(prev.name) else prev.name),
                state=(rec.state if rec.state != "Unknown" else prev.state),
                start_date=min(rec.start_date, prev.start_date),
                end_date=max(rec.end_date, prev.end_date),
                size_acres=max(rec.size_acres, prev.size_acres),
                geometry=(rec.geometry if rec.size_acres >= prev.size_acres else prev.geometry),
                cause_general=(rec.cause_general if rec.cause_general != "Unknown" else prev.cause_general),
                cause_specific=(rec.cause_specific if rec.cause_specific != "Unknown" else prev.cause_specific),
                incident_kind=(rec.incident_kind if rec.incident_kind != "Unknown" else prev.incident_kind),
                fuel_group=(rec.fuel_group if rec.fuel_group != "Unknown" else prev.fuel_group),
            )

    fires = list(by_id.values())
    fires.sort(key=lambda f: (f.start_date, -f.size_acres, f.fire_id))
    return fires


def iterate_viirs_chunks(viirs_zip: Path, min_day: int, max_day: int, chunksize: int = 300_000):
    usecols = ["latitude", "longitude", "acq_date", "frp", "confidence", "daynight"]
    with zipfile.ZipFile(viirs_zip) as zf:
        csv_members = [n for n in zf.namelist() if n.lower().endswith(".csv") and "archive" in n.lower()]
        if not csv_members:
            raise FileNotFoundError(f"No archive CSV in {viirs_zip}")
        for member in csv_members:
            with zf.open(member) as fp:
                for chunk in pd.read_csv(fp, usecols=usecols, chunksize=chunksize, low_memory=False):
                    dt = pd.to_datetime(chunk["acq_date"], errors="coerce").to_numpy(dtype="datetime64[D]")
                    lat = pd.to_numeric(chunk["latitude"], errors="coerce").to_numpy(dtype=np.float64)
                    lon = pd.to_numeric(chunk["longitude"], errors="coerce").to_numpy(dtype=np.float64)
                    frp = pd.to_numeric(chunk["frp"], errors="coerce").to_numpy(dtype=np.float64)
                    conf = chunk["confidence"].astype(str).str.strip().str.lower().str.slice(0, 1).to_numpy(dtype="<U1")
                    dn = chunk["daynight"].astype(str).str.strip().str.upper().str.slice(0, 1).to_numpy(dtype="<U1")

                    valid = ~np.isnat(dt)
                    valid &= np.isfinite(lat) & np.isfinite(lon) & np.isfinite(frp)
                    valid &= (lat >= -90) & (lat <= 90) & (lon >= -180) & (lon <= 180)
                    valid &= frp >= 0.0
                    if not np.any(valid):
                        continue

                    day_int = (dt[valid] - DAY0).astype(np.int32)
                    keep = (day_int >= min_day) & (day_int <= max_day)
                    if not np.any(keep):
                        continue

                    yield (
                        lon[valid][keep],
                        lat[valid][keep],
                        day_int[keep],
                        frp[valid][keep],
                        conf[valid][keep],
                        dn[valid][keep],
                    )


def to_markdown_table(df: pd.DataFrame, rows: int = 15) -> str:
    if df.empty:
        return "_No rows_"
    view = df.head(rows).copy()

    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return f"{float(x):.3f}"
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        return str(x)

    cols = list(view.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in view.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def run_analysis(
    fires: list[WfigsFire],
    viirs_zip: Path,
    start_year: int,
    end_year: int,
    min_acres: float,
    pad_days: int,
    output_dir: Path,
) -> None:
    n = len(fires)
    fire_df = pd.DataFrame(
        {
            "fire_idx": np.arange(n, dtype=np.int32),
            "fire_id": [f.fire_id for f in fires],
            "name": [f.name for f in fires],
            "state": [f.state for f in fires],
            "start_date": [f.start_date for f in fires],
            "end_date": [f.end_date for f in fires],
            "size_acres": [f.size_acres for f in fires],
            "cause_general": [f.cause_general for f in fires],
            "cause_specific": [f.cause_specific for f in fires],
            "incident_kind": [f.incident_kind for f in fires],
            "fuel_group": [f.fuel_group for f in fires],
            "centroid_lon": [f.geometry.centroid.x for f in fires],
            "centroid_lat": [f.geometry.centroid.y for f in fires],
        }
    )
    fire_df["size_bucket"] = fire_df["size_acres"].map(size_bucket)

    geoms = [f.geometry for f in fires]
    tree = STRtree(geoms)

    start_day = np.array([day_int_from_date(f.start_date) for f in fires], dtype=np.int32)
    end_day = np.array([day_int_from_date(f.end_date) for f in fires], dtype=np.int32)
    duration_days = np.maximum(end_day - start_day + 1, 1).astype(np.int32)

    min_day = day_int_from_date(date(start_year, 1, 1) - timedelta(days=pad_days + 2))
    max_day = day_int_from_date(date(end_year, 12, 31) + timedelta(days=400))

    pix_count = np.zeros(n, dtype=np.int64)
    frp_sum = np.zeros(n, dtype=np.float64)
    frp_min = np.full(n, np.inf, dtype=np.float64)
    frp_max = np.zeros(n, dtype=np.float64)
    det_min_lon = np.full(n, np.inf, dtype=np.float64)
    det_max_lon = np.full(n, -np.inf, dtype=np.float64)
    det_min_lat = np.full(n, np.inf, dtype=np.float64)
    det_max_lat = np.full(n, -np.inf, dtype=np.float64)
    pre_count = np.zeros(n, dtype=np.int64)
    post_count = np.zeros(n, dtype=np.int64)

    assigned_fire_chunks: list[np.ndarray] = []
    assigned_day_chunks: list[np.ndarray] = []
    assigned_frp_chunks: list[np.ndarray] = []
    assigned_conf_chunks: list[np.ndarray] = []
    assigned_dn_chunks: list[np.ndarray] = []
    fire_day_chunks: list[np.ndarray] = []

    processed_chunks = 0
    for lon, lat, day, frp, conf, dn in iterate_viirs_chunks(viirs_zip, min_day=min_day, max_day=max_day):
        processed_chunks += 1
        pts = shapely_points(lon, lat)
        pairs = tree.query(pts, predicate="within")
        if pairs.shape[1] == 0:
            continue

        p_idx = pairs[0].astype(np.int32)
        f_idx = pairs[1].astype(np.int32)
        d = day[p_idx]

        in_plusminus = (d >= (start_day[f_idx] - pad_days)) & (d <= (end_day[f_idx] + pad_days))
        if not np.any(in_plusminus):
            continue
        p_idx = p_idx[in_plusminus]
        f_idx = f_idx[in_plusminus]
        d = d[in_plusminus]

        # Resolve overlaps: assign one fire per VIIRS pixel-day (prefer smaller fire area, then shorter duration).
        order = np.lexsort((duration_days[f_idx], fire_df["size_acres"].to_numpy(dtype=np.float64)[f_idx], p_idx))
        p_sorted = p_idx[order]
        f_sorted = f_idx[order]
        d_sorted = d[order]
        first_mask = np.r_[True, p_sorted[1:] != p_sorted[:-1]]
        p_sel = p_sorted[first_mask]
        f_sel = f_sorted[first_mask]
        d_sel = d_sorted[first_mask]

        frp_sel = frp[p_sel].astype(np.float32)
        conf_sel = conf[p_sel]
        dn_sel = dn[p_sel]
        lon_sel = lon[p_sel]
        lat_sel = lat[p_sel]

        s = start_day[f_sel]
        e = end_day[f_sel]
        is_pre = d_sel < s
        is_post = d_sel > e
        is_in = ~is_pre & ~is_post

        if np.any(is_pre):
            np.add.at(pre_count, f_sel[is_pre], 1)
        if np.any(is_post):
            np.add.at(post_count, f_sel[is_post], 1)

        if np.any(is_in):
            fi = f_sel[is_in]
            di = d_sel[is_in]
            frpi = frp_sel[is_in]
            confi = conf_sel[is_in]
            dni = dn_sel[is_in]
            loni = lon_sel[is_in]
            lati = lat_sel[is_in]

            np.add.at(pix_count, fi, 1)
            np.add.at(frp_sum, fi, frpi)
            np.minimum.at(frp_min, fi, frpi)
            np.maximum.at(frp_max, fi, frpi)
            np.minimum.at(det_min_lon, fi, loni)
            np.maximum.at(det_max_lon, fi, loni)
            np.minimum.at(det_min_lat, fi, lati)
            np.maximum.at(det_max_lat, fi, lati)

            assigned_fire_chunks.append(fi.astype(np.int32))
            assigned_day_chunks.append(di.astype(np.int32))
            assigned_frp_chunks.append(frpi.astype(np.float32))
            assigned_conf_chunks.append(confi)
            assigned_dn_chunks.append(dni)
            fire_day_chunks.append(np.column_stack((fi.astype(np.int32), di.astype(np.int32))))

        if processed_chunks % 10 == 0:
            print(f"processed VIIRS chunks: {processed_chunks}")

    if not assigned_fire_chunks:
        raise RuntimeError("No VIIRS detections were matched to WFIGS fires.")

    fire_idx_all = np.concatenate(assigned_fire_chunks)
    day_all = np.concatenate(assigned_day_chunks)
    frp_all = np.concatenate(assigned_frp_chunks)
    conf_all = np.concatenate(assigned_conf_chunks)
    dn_all = np.concatenate(assigned_dn_chunks)

    fire_day_pairs = np.vstack(fire_day_chunks)
    unique_fire_day = np.unique(fire_day_pairs, axis=0)
    days_with_det = np.bincount(unique_fire_day[:, 0], minlength=n)

    fire_df["active_window_days"] = duration_days
    fire_df["pixels_in_window_anyconf"] = pix_count
    fire_df["frp_sum_in_window"] = frp_sum
    fire_df["frp_mean_in_window"] = np.where(pix_count > 0, frp_sum / pix_count, np.nan)
    fire_df["frp_min_in_window"] = np.where(np.isfinite(frp_min), frp_min, np.nan)
    fire_df["frp_max_in_window"] = np.where(pix_count > 0, frp_max, np.nan)
    fire_df["pre_start_pixels_pad"] = pre_count
    fire_df["post_end_pixels_pad"] = post_count
    fire_df["days_with_pixels_anyconf"] = days_with_det
    fire_df["pct_active_days_with_pixels_anyconf"] = np.where(
        fire_df["active_window_days"] > 0,
        fire_df["days_with_pixels_anyconf"] / fire_df["active_window_days"],
        np.nan,
    )
    fire_df["pixels_per_1000_acres"] = np.where(
        fire_df["size_acres"] > 0,
        fire_df["pixels_in_window_anyconf"] / (fire_df["size_acres"] / 1000.0),
        np.nan,
    )
    fire_df["frp_per_acre"] = np.where(
        fire_df["size_acres"] > 0,
        fire_df["frp_sum_in_window"] / fire_df["size_acres"],
        np.nan,
    )
    fire_df["det_bbox_lon_span_deg"] = np.where(
        np.isfinite(det_min_lon),
        det_max_lon - det_min_lon,
        np.nan,
    )
    fire_df["det_bbox_lat_span_deg"] = np.where(
        np.isfinite(det_min_lat),
        det_max_lat - det_min_lat,
        np.nan,
    )

    assigned_df = pd.DataFrame(
        {
            "fire_idx": fire_idx_all,
            "day_int": day_all,
            "frp": frp_all.astype(np.float64),
            "confidence": conf_all,
            "daynight": dn_all,
        }
    ).merge(
        fire_df[
            [
                "fire_idx",
                "fire_id",
                "name",
                "state",
                "size_bucket",
                "cause_general",
                "incident_kind",
                "fuel_group",
                "start_date",
                "active_window_days",
            ]
        ],
        on="fire_idx",
        how="left",
    )

    rel = (
        (assigned_df["day_int"].to_numpy(dtype=np.int32) - start_day[assigned_df["fire_idx"].to_numpy(dtype=np.int32)])
        / np.maximum(
            duration_days[assigned_df["fire_idx"].to_numpy(dtype=np.int32)] - 1,
            1,
        )
    )
    assigned_df["rel_bin"] = np.clip(np.floor(rel * 10.0), 0, 9).astype(np.int32)
    assigned_df["year"] = (
        (DAY0 + assigned_df["day_int"].to_numpy(dtype=np.int32).astype("timedelta64[D]")).astype("datetime64[Y]").astype(int)
        + 1970
    )

    overall = {
        "fire_count": int(len(fire_df)),
        "fires_with_any_pixels": int((fire_df["pixels_in_window_anyconf"] > 0).sum()),
        "total_pixels_in_window_anyconf": int(len(assigned_df)),
        "mean_pixels_per_fire": float(fire_df["pixels_in_window_anyconf"].mean()),
        "median_pixels_per_fire": float(fire_df["pixels_in_window_anyconf"].median()),
        "p90_pixels_per_fire": float(fire_df["pixels_in_window_anyconf"].quantile(0.9)),
        "weighted_pct_active_days_with_pixels_anyconf": float(
            fire_df["days_with_pixels_anyconf"].sum() / max(fire_df["active_window_days"].sum(), 1)
        ),
        "frp_mean_pixel": float(assigned_df["frp"].mean()),
        "frp_median_pixel": float(assigned_df["frp"].median()),
        "frp_p90_pixel": float(assigned_df["frp"].quantile(0.9)),
        "frp_p99_pixel": float(assigned_df["frp"].quantile(0.99)),
        "day_fraction": float((assigned_df["daynight"] == "D").mean()),
        "night_fraction": float((assigned_df["daynight"] == "N").mean()),
    }

    by_cause = (
        assigned_df.groupby("cause_general", dropna=False)
        .agg(
            pixel_count=("frp", "size"),
            frp_mean=("frp", "mean"),
            frp_median=("frp", "median"),
            frp_p90=("frp", lambda s: s.quantile(0.9)),
            fire_count=("fire_id", "nunique"),
        )
        .reset_index()
        .sort_values("pixel_count", ascending=False)
    )
    by_cause["pixel_share"] = by_cause["pixel_count"] / len(assigned_df)

    by_size_bucket = (
        assigned_df.groupby("size_bucket", dropna=False)
        .agg(
            pixel_count=("frp", "size"),
            frp_mean=("frp", "mean"),
            frp_median=("frp", "median"),
            frp_p90=("frp", lambda s: s.quantile(0.9)),
            fire_count=("fire_id", "nunique"),
        )
        .reset_index()
    )
    by_size_bucket["pixel_share"] = by_size_bucket["pixel_count"] / len(assigned_df)
    bucket_order = {"1k-5k": 0, "5k-20k": 1, "20k-100k": 2, "100k+": 3}
    by_size_bucket["__order"] = by_size_bucket["size_bucket"].map(bucket_order).fillna(9)
    by_size_bucket = by_size_bucket.sort_values("__order").drop(columns="__order")

    by_state = (
        assigned_df.groupby("state", dropna=False)
        .agg(
            fire_count=("fire_id", "nunique"),
            pixel_count=("frp", "size"),
            frp_mean=("frp", "mean"),
            frp_median=("frp", "median"),
            frp_p90=("frp", lambda s: s.quantile(0.9)),
        )
        .reset_index()
        .sort_values("pixel_count", ascending=False)
    )
    by_state["pixel_share"] = by_state["pixel_count"] / len(assigned_df)

    by_rel_bin = (
        assigned_df.groupby("rel_bin")
        .agg(
            pixel_count=("frp", "size"),
            frp_mean=("frp", "mean"),
            frp_median=("frp", "median"),
            frp_p90=("frp", lambda s: s.quantile(0.9)),
        )
        .reset_index()
    )
    by_rel_bin["bin_start_rel"] = by_rel_bin["rel_bin"] / 10.0
    by_rel_bin["bin_end_rel"] = (by_rel_bin["rel_bin"] + 1) / 10.0

    by_year = (
        assigned_df.groupby("year")
        .agg(
            pixel_count=("frp", "size"),
            frp_mean=("frp", "mean"),
            frp_median=("frp", "median"),
            frp_p90=("frp", lambda s: s.quantile(0.9)),
        )
        .reset_index()
        .sort_values("year")
    )

    top_by_pixels = fire_df.sort_values("pixels_in_window_anyconf", ascending=False).head(25)
    top_by_density = fire_df.sort_values("pixels_per_1000_acres", ascending=False).head(25)
    top_by_mean_frp = (
        fire_df[fire_df["pixels_in_window_anyconf"] >= 50]
        .sort_values("frp_mean_in_window", ascending=False)
        .head(25)
    )

    out_dir = output_dir
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fire_df.to_csv(out_dir / "wfigs_fire_level_viirs_stats.csv", index=False)
    assigned_df.to_csv(out_dir / "wfigs_assigned_viirs_pixels.csv", index=False)
    by_cause.to_csv(out_dir / "wfigs_viirs_by_cause_general.csv", index=False)
    by_size_bucket.to_csv(out_dir / "wfigs_viirs_by_size_bucket.csv", index=False)
    by_state.to_csv(out_dir / "wfigs_viirs_by_state.csv", index=False)
    by_rel_bin.to_csv(out_dir / "wfigs_viirs_frp_by_relative_window_bin.csv", index=False)
    by_year.to_csv(out_dir / "wfigs_viirs_frp_by_year.csv", index=False)
    top_by_pixels.to_csv(out_dir / "wfigs_top_fires_by_pixels.csv", index=False)
    top_by_density.to_csv(out_dir / "wfigs_top_fires_by_pixel_density.csv", index=False)
    top_by_mean_frp.to_csv(out_dir / "wfigs_top_fires_by_mean_frp.csv", index=False)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(overall, fp, indent=2)

    # Spatial view: centroids colored by VIIRS pixel counts.
    plot_df = fire_df[fire_df["pixels_in_window_anyconf"] > 0].copy()
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(
        plot_df["centroid_lon"],
        plot_df["centroid_lat"],
        c=np.log10(plot_df["pixels_in_window_anyconf"] + 1),
        s=np.clip(np.sqrt(plot_df["size_acres"]) / 2.0, 8, 80),
        cmap="viridis",
        alpha=0.7,
        linewidths=0.0,
    )
    cb = plt.colorbar(sc)
    cb.set_label("log10(VIIRS pixels + 1)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"WFIGS Fires ({start_year}-{end_year}, >={min_acres:g} ac): Centroids Colored by VIIRS Pixel Count")
    plt.tight_layout()
    plt.savefig(fig_dir / "wfigs_fire_centroids_pixels.png", dpi=140)
    plt.close()

    # FRP vs pixel count by fire.
    plot_df2 = fire_df[fire_df["pixels_in_window_anyconf"] > 0].copy()
    plt.figure(figsize=(8, 6))
    plt.scatter(
        np.log10(plot_df2["pixels_in_window_anyconf"]),
        plot_df2["frp_mean_in_window"],
        alpha=0.55,
        s=14,
    )
    plt.xlabel("log10(VIIRS pixels in window)")
    plt.ylabel("Mean FRP per pixel (MW)")
    plt.title("Per-fire Mean FRP vs Pixel Count")
    plt.tight_layout()
    plt.savefig(fig_dir / "wfigs_fire_mean_frp_vs_pixels.png", dpi=140)
    plt.close()

    report_lines = []
    report_lines.append(f"# WFIGS + VIIRS Stats ({start_year}-{end_year}, >={min_acres:g} acres)")
    report_lines.append("")
    report_lines.append("## Cohort")
    report_lines.append("- WFIGS wildfire incidents only (`attr_IncidentTypeCategory = WF`).")
    report_lines.append(
        f"- Time filter by discovery date: {start_year}-01-01 to {end_year}-12-31."
    )
    report_lines.append(f"- Fire size filter: >= {min_acres:g} acres.")
    report_lines.append("- VIIRS detections assigned only when point is inside perimeter and date is within start/end window.")
    report_lines.append("- Overlap rule: one fire per VIIRS pixel-day (prefer smaller fire area, then shorter duration).")
    report_lines.append("")
    report_lines.append("## Overall VIIRS Stats")
    for k, v in overall.items():
        if isinstance(v, float):
            report_lines.append(f"- `{k}`: {v:.4f}")
        else:
            report_lines.append(f"- `{k}`: {v}")
    report_lines.append("")
    report_lines.append("## By Cause (Top Rows)")
    report_lines.append(to_markdown_table(by_cause, rows=12))
    report_lines.append("")
    report_lines.append("## By Size Bucket")
    report_lines.append(to_markdown_table(by_size_bucket, rows=10))
    report_lines.append("")
    report_lines.append("## By State (Top Rows)")
    report_lines.append(to_markdown_table(by_state, rows=15))
    report_lines.append("")
    report_lines.append("## FRP Over Fire Progress (Relative Window Bins)")
    report_lines.append(to_markdown_table(by_rel_bin, rows=10))
    report_lines.append("")
    report_lines.append("## Top Fires by VIIRS Pixel Count")
    report_lines.append(to_markdown_table(top_by_pixels[["fire_id", "name", "state", "size_acres", "pixels_in_window_anyconf", "frp_mean_in_window"]], rows=20))
    report_lines.append("")
    report_lines.append("## Top Fires by Pixel Density")
    report_lines.append(to_markdown_table(top_by_density[["fire_id", "name", "state", "size_acres", "pixels_per_1000_acres", "frp_mean_in_window"]], rows=20))
    report_lines.append("")

    (out_dir / "wfigs_viirs_stats_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WFIGS-only VIIRS pixel and FRP statistics")
    parser.add_argument(
        "--wfigs-geojson",
        type=Path,
        default=Path("/Users/lorn/Downloads/WFIGS_Interagency_Perimeters_6781836551080060975.geojson"),
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
        default=Path("/Users/lorn/Code/gribcheck/reports/wfigs_viirs_stats_2021_2023_1000ac"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading WFIGS cohort...")
    fires = load_wfigs_fires(
        path=args.wfigs_geojson,
        start_year=args.start_year,
        end_year=args.end_year,
        min_acres=args.min_acres,
    )
    print(f"Loaded fires: {len(fires):,}")
    print("Running VIIRS assignment and statistics...")
    run_analysis(
        fires=fires,
        viirs_zip=args.viirs_zip,
        start_year=args.start_year,
        end_year=args.end_year,
        min_acres=args.min_acres,
        pad_days=args.window_pad_days,
        output_dir=args.output_dir,
    )
    print("Done.")
    print(f"Report: {args.output_dir / 'wfigs_viirs_stats_report.md'}")


if __name__ == "__main__":
    main()
