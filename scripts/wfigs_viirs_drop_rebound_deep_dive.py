#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DAY0 = np.datetime64("1970-01-01", "D")


def to_day_int(series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(series, errors="coerce").to_numpy(dtype="datetime64[D]")
    out = (dt - DAY0).astype(np.int32)
    return out


def read_inputs(report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fire_path = report_dir / "wfigs_fire_level_viirs_stats.csv"
    pix_path = report_dir / "wfigs_assigned_viirs_pixels.csv"
    if not fire_path.exists() or not pix_path.exists():
        missing = [str(p) for p in (fire_path, pix_path) if not p.exists()]
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    fire_cols = [
        "fire_idx",
        "fire_id",
        "name",
        "state",
        "start_date",
        "end_date",
        "size_acres",
        "size_bucket",
        "active_window_days",
    ]
    fire_df = pd.read_csv(fire_path, usecols=fire_cols, low_memory=False)
    fire_df = fire_df.drop_duplicates("fire_idx").copy()
    fire_df["start_date"] = pd.to_datetime(fire_df["start_date"], errors="coerce")
    fire_df["end_date"] = pd.to_datetime(fire_df["end_date"], errors="coerce")
    fire_df = fire_df[fire_df["start_date"].notna() & fire_df["end_date"].notna()].copy()
    fire_df["start_day"] = to_day_int(fire_df["start_date"])
    fire_df["end_day"] = to_day_int(fire_df["end_date"])
    fire_df = fire_df[fire_df["end_day"] >= fire_df["start_day"]].copy()
    fire_df["active_window_days"] = (
        pd.to_numeric(fire_df["active_window_days"], errors="coerce")
        .fillna((fire_df["end_day"] - fire_df["start_day"] + 1).astype(np.int32))
        .astype(np.int32)
    )

    pix_cols = ["fire_idx", "day_int", "frp"]
    pix_df = pd.read_csv(pix_path, usecols=pix_cols, low_memory=False)
    pix_df["day_int"] = pd.to_numeric(pix_df["day_int"], errors="coerce").astype("Int64")
    pix_df["frp"] = pd.to_numeric(pix_df["frp"], errors="coerce")
    pix_df = pix_df[pix_df["day_int"].notna() & pix_df["frp"].notna() & (pix_df["frp"] >= 0)].copy()
    pix_df["day_int"] = pix_df["day_int"].astype(np.int32)
    pix_df = pix_df[pix_df["fire_idx"].isin(fire_df["fire_idx"])].copy()

    return fire_df, pix_df


def build_daily_timeseries(fire_df: pd.DataFrame, pix_df: pd.DataFrame) -> pd.DataFrame:
    daily_detect = (
        pix_df.groupby(["fire_idx", "day_int"], as_index=False)
        .agg(
            pixel_count=("frp", "size"),
            frp_total=("frp", "sum"),
            frp_mean_pixel=("frp", "mean"),
        )
        .astype({"pixel_count": np.int32})
    )

    lengths = (fire_df["end_day"] - fire_df["start_day"] + 1).astype(np.int32).to_numpy()
    fire_idx_rep = np.repeat(fire_df["fire_idx"].to_numpy(dtype=np.int32), lengths)
    start_rep = np.repeat(fire_df["start_day"].to_numpy(dtype=np.int32), lengths)
    offsets = np.concatenate([np.arange(n, dtype=np.int32) for n in lengths])
    day_int = start_rep + offsets

    full = pd.DataFrame({"fire_idx": fire_idx_rep, "day_int": day_int})
    full = full.merge(daily_detect, on=["fire_idx", "day_int"], how="left")
    full["pixel_count"] = full["pixel_count"].fillna(0).astype(np.int32)
    full["frp_total"] = full["frp_total"].fillna(0.0)
    full["frp_mean_pixel"] = full["frp_mean_pixel"].fillna(0.0)

    meta_cols = [
        "fire_idx",
        "fire_id",
        "name",
        "state",
        "size_acres",
        "size_bucket",
        "start_day",
        "end_day",
        "active_window_days",
    ]
    full = full.merge(fire_df[meta_cols], on="fire_idx", how="left")
    full["day_of_fire"] = (full["day_int"] - full["start_day"]).astype(np.int32)
    denom = np.maximum(full["active_window_days"] - 1, 1)
    full["rel_progress"] = (full["day_of_fire"] / denom).clip(0, 1)
    full["rel_bin20"] = np.minimum((full["rel_progress"] * 20).astype(np.int32), 19)
    full = full.sort_values(["fire_idx", "day_int"]).reset_index(drop=True)
    full["day_pos"] = full.groupby("fire_idx").cumcount().astype(np.int32)
    return full


def add_transitions(full: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
    prev_col = f"{prefix}_prev"
    delta_col = f"{prefix}_delta"
    pct_col = f"{prefix}_pct_change"
    drop_col = f"{prefix}_is_drop"
    inc_col = f"{prefix}_is_increase"

    prev = full.groupby("fire_idx")[value_col].shift(1)
    full[prev_col] = prev
    full[delta_col] = full[value_col] - full[prev_col]
    full[pct_col] = np.where(full[prev_col] > 0, full[delta_col] / full[prev_col], np.nan)
    full[drop_col] = (full[prev_col] > 0) & (full[value_col] < full[prev_col])
    full[inc_col] = full[value_col] > full[prev_col]
    return full


def classify_pixel_transition(row: pd.Series) -> str:
    prev = row["pix_prev"]
    cur = row["pixel_count"]
    if pd.isna(prev):
        return "first_day"
    if prev == 0 and cur == 0:
        return "zero_to_zero"
    if prev == 0 and cur > 0:
        return "start_from_zero"
    if prev > 0 and cur == 0:
        return "drop_to_zero"
    if prev > 0 and cur > 0 and cur < prev:
        return "partial_drop"
    if prev > 0 and cur > 0 and cur > prev:
        return "increase_positive"
    return "stable_positive"


def drop_events(
    full: pd.DataFrame,
    value_col: str,
    prev_col: str,
    drop_col: str,
    label: str,
) -> pd.DataFrame:
    events = full[full[drop_col]].copy()
    events["metric"] = label
    events["drop_from"] = events[prev_col].astype(float)
    events["drop_to"] = events[value_col].astype(float)
    events["drop_amount"] = events["drop_from"] - events["drop_to"]
    events["drop_pct"] = np.where(events["drop_from"] > 0, events["drop_amount"] / events["drop_from"], np.nan)
    events["drop_to_zero"] = events["drop_to"] == 0
    events["steep_drop_25"] = events["drop_pct"] >= 0.25
    events["steep_drop_50"] = events["drop_pct"] >= 0.50
    events["steep_drop_75"] = events["drop_pct"] >= 0.75
    keep = [
        "metric",
        "fire_idx",
        "fire_id",
        "name",
        "state",
        "size_bucket",
        "day_int",
        "day_of_fire",
        "rel_progress",
        "rel_bin20",
        "drop_from",
        "drop_to",
        "drop_amount",
        "drop_pct",
        "drop_to_zero",
        "steep_drop_25",
        "steep_drop_50",
        "steep_drop_75",
    ]
    return events[keep].reset_index(drop=True)


def add_rebound_metrics(
    full: pd.DataFrame,
    events: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    values_by_fire: dict[int, np.ndarray] = {
        int(fid): grp[value_col].to_numpy(dtype=float)
        for fid, grp in full.groupby("fire_idx", sort=False)
    }
    pos_by_fire_day: dict[tuple[int, int], int] = {
        (int(fid), int(day)): int(pos)
        for fid, day, pos in full[["fire_idx", "day_int", "day_pos"]].itertuples(index=False, name=None)
    }

    out = events.copy()
    next_up = np.zeros(len(out), dtype=bool)
    up3 = np.zeros(len(out), dtype=bool)
    up7 = np.zeros(len(out), dtype=bool)
    rec3 = np.zeros(len(out), dtype=bool)
    rec7 = np.zeros(len(out), dtype=bool)
    best3 = np.zeros(len(out), dtype=float)
    best7 = np.zeros(len(out), dtype=float)

    for i, row in enumerate(out.itertuples(index=False)):
        fid = int(row.fire_idx)
        day = int(row.day_int)
        drop_to = float(row.drop_to)
        drop_from = float(row.drop_from)
        arr = values_by_fire[fid]
        pos = pos_by_fire_day[(fid, day)]

        w1 = arr[pos + 1 : pos + 2]
        w3 = arr[pos + 1 : pos + 4]
        w7 = arr[pos + 1 : pos + 8]

        if w1.size:
            next_up[i] = bool(w1[0] > drop_to)
        if w3.size:
            up3[i] = bool(np.any(w3 > drop_to))
            rec3[i] = bool(np.any(w3 >= drop_from))
            best3[i] = float(np.max(w3) - drop_to)
        if w7.size:
            up7[i] = bool(np.any(w7 > drop_to))
            rec7[i] = bool(np.any(w7 >= drop_from))
            best7[i] = float(np.max(w7) - drop_to)

    out["rebound_next_day"] = next_up
    out["rebound_within_3d"] = up3
    out["rebound_within_7d"] = up7
    out["recover_pre_drop_within_3d"] = rec3
    out["recover_pre_drop_within_7d"] = rec7
    out["best_rebound_amount_3d"] = best3
    out["best_rebound_amount_7d"] = best7
    return out


def quantile_series(x: pd.Series, q: float) -> float:
    if x.empty:
        return float("nan")
    return float(x.quantile(q))


def summarize_drops(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    def agg_block(df: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "drop_events": int(len(df)),
                "drop_to_zero_pct": float(df["drop_to_zero"].mean() * 100.0),
                "steep_drop_25_pct": float(df["steep_drop_25"].mean() * 100.0),
                "steep_drop_50_pct": float(df["steep_drop_50"].mean() * 100.0),
                "steep_drop_75_pct": float(df["steep_drop_75"].mean() * 100.0),
                "drop_amount_mean": float(df["drop_amount"].mean()),
                "drop_amount_median": float(df["drop_amount"].median()),
                "drop_amount_p90": quantile_series(df["drop_amount"], 0.9),
                "drop_pct_mean": float(df["drop_pct"].mean() * 100.0),
                "drop_pct_median": float(df["drop_pct"].median() * 100.0),
                "drop_pct_p90": quantile_series(df["drop_pct"], 0.9) * 100.0,
            }
        )

    out = events.groupby(["metric", "size_bucket"], as_index=False).apply(agg_block, include_groups=False)
    return out.reset_index(drop=True)


def summarize_rebounds(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    def agg_block(df: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "drop_events": int(len(df)),
                "rebound_next_day_pct": float(df["rebound_next_day"].mean() * 100.0),
                "rebound_within_3d_pct": float(df["rebound_within_3d"].mean() * 100.0),
                "rebound_within_7d_pct": float(df["rebound_within_7d"].mean() * 100.0),
                "recover_pre_drop_within_3d_pct": float(df["recover_pre_drop_within_3d"].mean() * 100.0),
                "recover_pre_drop_within_7d_pct": float(df["recover_pre_drop_within_7d"].mean() * 100.0),
                "best_rebound_amount_3d_mean": float(df["best_rebound_amount_3d"].mean()),
                "best_rebound_amount_7d_mean": float(df["best_rebound_amount_7d"].mean()),
            }
        )

    out = events.groupby(["metric", "size_bucket"], as_index=False).apply(agg_block, include_groups=False)
    return out.reset_index(drop=True)


def summarize_transition_types(full: pd.DataFrame) -> pd.DataFrame:
    x = full[full["pix_prev"].notna()].copy()
    x["transition"] = x.apply(classify_pixel_transition, axis=1)

    counts = (
        x.groupby(["size_bucket", "transition"], as_index=False)
        .agg(days=("transition", "size"))
        .sort_values(["size_bucket", "days"], ascending=[True, False])
    )
    totals = counts.groupby("size_bucket", as_index=False)["days"].sum().rename(columns={"days": "total_days"})
    out = counts.merge(totals, on="size_bucket", how="left")
    out["pct_days"] = out["days"] / out["total_days"] * 100.0
    return out


def build_drop_streaks(
    full: pd.DataFrame,
    value_col: str,
    prev_col: str,
    drop_col: str,
    label: str,
) -> pd.DataFrame:
    rows: list[dict] = []

    for fid, grp in full.groupby("fire_idx", sort=False):
        g = grp.sort_values("day_int")
        is_drop = g[drop_col].to_numpy(dtype=bool)
        if not np.any(is_drop):
            continue
        cur = g[value_col].to_numpy(dtype=float)
        prev = g[prev_col].to_numpy(dtype=float)
        day_int = g["day_int"].to_numpy(dtype=np.int32)
        day_of_fire = g["day_of_fire"].to_numpy(dtype=np.int32)
        rel_progress = g["rel_progress"].to_numpy(dtype=float)
        size_bucket = str(g["size_bucket"].iloc[0])
        fire_id = str(g["fire_id"].iloc[0])
        name = str(g["name"].iloc[0])
        state = str(g["state"].iloc[0])

        idxs = np.where(is_drop)[0]
        starts = [idxs[0]]
        ends: list[int] = []
        for i in range(1, len(idxs)):
            if idxs[i] != idxs[i - 1] + 1:
                ends.append(idxs[i - 1])
                starts.append(idxs[i])
        ends.append(idxs[-1])

        for s, e in zip(starts, ends):
            drop_from = float(prev[s])
            drop_to = float(cur[e])
            drop_amt = float(drop_from - drop_to)
            drop_pct = float(drop_amt / drop_from) if drop_from > 0 else np.nan
            rows.append(
                {
                    "metric": label,
                    "fire_idx": int(fid),
                    "fire_id": fire_id,
                    "name": name,
                    "state": state,
                    "size_bucket": size_bucket,
                    "streak_start_day_int": int(day_int[s]),
                    "streak_end_day_int": int(day_int[e]),
                    "streak_start_day_of_fire": int(day_of_fire[s]),
                    "streak_end_day_of_fire": int(day_of_fire[e]),
                    "streak_start_rel_progress": float(rel_progress[s]),
                    "streak_end_rel_progress": float(rel_progress[e]),
                    "streak_len_days": int(e - s + 1),
                    "drop_from": drop_from,
                    "drop_to": drop_to,
                    "streak_drop_amount": drop_amt,
                    "streak_drop_pct": drop_pct,
                    "streak_ends_at_zero": bool(drop_to == 0),
                }
            )

    return pd.DataFrame(rows)


def summarize_drop_streaks(streaks: pd.DataFrame) -> pd.DataFrame:
    if streaks.empty:
        return pd.DataFrame()

    def agg_block(df: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "streaks": int(len(df)),
                "streak_len_mean": float(df["streak_len_days"].mean()),
                "streak_len_median": float(df["streak_len_days"].median()),
                "streak_len_p90": quantile_series(df["streak_len_days"], 0.9),
                "streak_ends_at_zero_pct": float(df["streak_ends_at_zero"].mean() * 100.0),
                "streak_drop_amount_mean": float(df["streak_drop_amount"].mean()),
                "streak_drop_amount_median": float(df["streak_drop_amount"].median()),
                "streak_drop_amount_p90": quantile_series(df["streak_drop_amount"], 0.9),
                "streak_drop_pct_mean": float(df["streak_drop_pct"].mean() * 100.0),
                "streak_drop_pct_median": float(df["streak_drop_pct"].median() * 100.0),
                "streak_drop_pct_p90": quantile_series(df["streak_drop_pct"], 0.9) * 100.0,
            }
        )

    out = streaks.groupby(["metric", "size_bucket"], as_index=False).apply(agg_block, include_groups=False)
    return out.reset_index(drop=True)


def day_of_fire_curve(full: pd.DataFrame) -> pd.DataFrame:
    def mean_if_detected(df: pd.DataFrame, col: str) -> float:
        y = df.loc[df["pixel_count"] > 0, col]
        return float(y.mean()) if len(y) else 0.0

    grp = full.groupby(["size_bucket", "day_of_fire"], as_index=False)
    out = grp.agg(
        fire_days=("fire_idx", "size"),
        days_with_detections=("pixel_count", lambda s: int((s > 0).sum())),
        mean_pixels=("pixel_count", "mean"),
        median_pixels=("pixel_count", "median"),
        p90_pixels=("pixel_count", lambda s: quantile_series(s, 0.9)),
        mean_frp_total=("frp_total", "mean"),
        median_frp_total=("frp_total", "median"),
        drop_rate_pixels=("pix_is_drop", "mean"),
        drop_rate_frp_total=("frp_is_drop", "mean"),
    )
    out["pct_days_with_detections"] = out["days_with_detections"] / out["fire_days"] * 100.0
    det_mean = (
        full[full["pixel_count"] > 0]
        .groupby(["size_bucket", "day_of_fire"], as_index=False)
        .agg(mean_frp_pixel_on_detect_days=("frp_mean_pixel", "mean"))
    )
    out = out.merge(det_mean, on=["size_bucket", "day_of_fire"], how="left")
    out["mean_frp_pixel_on_detect_days"] = out["mean_frp_pixel_on_detect_days"].fillna(0.0)
    return out.sort_values(["size_bucket", "day_of_fire"]).reset_index(drop=True)


def relative_curve(full: pd.DataFrame) -> pd.DataFrame:
    grp = full.groupby(["size_bucket", "rel_bin20"], as_index=False)
    out = grp.agg(
        fire_days=("fire_idx", "size"),
        days_with_detections=("pixel_count", lambda s: int((s > 0).sum())),
        mean_pixels=("pixel_count", "mean"),
        median_pixels=("pixel_count", "median"),
        mean_frp_total=("frp_total", "mean"),
        mean_frp_pixel=("frp_mean_pixel", lambda s: float(s[s > 0].mean()) if np.any(s > 0) else 0.0),
        drop_rate_pixels=("pix_is_drop", "mean"),
        drop_rate_frp_total=("frp_is_drop", "mean"),
    )
    out["pct_days_with_detections"] = out["days_with_detections"] / out["fire_days"] * 100.0
    out["bin_start_rel"] = out["rel_bin20"] / 20.0
    out["bin_end_rel"] = (out["rel_bin20"] + 1) / 20.0
    return out.sort_values(["size_bucket", "rel_bin20"]).reset_index(drop=True)


def markdown_table(df: pd.DataFrame, n: int = 12) -> str:
    if df.empty:
        return "_No rows_"
    view = df.head(n).copy()
    cols = list(view.columns)

    def f(v):
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.3f}"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return str(v)

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in view.itertuples(index=False):
        lines.append("| " + " | ".join(f(v) for v in row) + " |")
    return "\n".join(lines)


def write_report(
    out_dir: Path,
    full: pd.DataFrame,
    drop_summary: pd.DataFrame,
    rebound_summary: pd.DataFrame,
    transition_summary: pd.DataFrame,
    drop_streak_summary: pd.DataFrame,
    rel_curve: pd.DataFrame,
) -> None:
    total_fire_days = int(len(full))
    days_with_det = int((full["pixel_count"] > 0).sum())
    pixel_drop_days = int(full["pix_is_drop"].sum())
    frp_drop_days = int(full["frp_is_drop"].sum())
    nonfirst_days = int(full["pix_prev"].notna().sum())

    lines: list[str] = []
    lines.append("# WFIGS VIIRS Drop/Rebound Deep Dive")
    lines.append("")
    lines.append("## Overall")
    lines.append(f"- total_fire_days: {total_fire_days}")
    lines.append(f"- fire_days_with_detections: {days_with_det} ({days_with_det / total_fire_days * 100.0:.2f}%)")
    lines.append(f"- pixel_drop_days: {pixel_drop_days} ({pixel_drop_days / nonfirst_days * 100.0:.2f}% of comparable days)")
    lines.append(f"- frp_total_drop_days: {frp_drop_days} ({frp_drop_days / nonfirst_days * 100.0:.2f}% of comparable days)")
    lines.append("")

    lines.append("## Drop-Only Summary by Size (pixels + FRP total)")
    lines.append(markdown_table(drop_summary, n=16))
    lines.append("")

    lines.append("## Rebound After Drop by Size")
    lines.append(markdown_table(rebound_summary, n=16))
    lines.append("")

    lines.append("## Consecutive Drop-Streaks by Size")
    lines.append(markdown_table(drop_streak_summary, n=16))
    lines.append("")

    lines.append("## Pixel Transition Mix by Size")
    lines.append(markdown_table(transition_summary, n=20))
    lines.append("")

    sample_rel = rel_curve[
        rel_curve["rel_bin20"].isin([0, 1, 2, 9, 10, 19])
    ].sort_values(["size_bucket", "rel_bin20"])
    lines.append("## Relative Progress Sample (early/mid/late)")
    lines.append(markdown_table(sample_rel, n=40))
    lines.append("")

    lines.append("## Files")
    lines.append("- wfigs_daily_fire_timeseries.csv")
    lines.append("- wfigs_pixel_drop_events.csv")
    lines.append("- wfigs_frp_total_drop_events.csv")
    lines.append("- wfigs_drop_summary_by_size_bucket.csv")
    lines.append("- wfigs_rebound_summary_by_size_bucket.csv")
    lines.append("- wfigs_drop_streaks.csv")
    lines.append("- wfigs_drop_streak_summary_by_size_bucket.csv")
    lines.append("- wfigs_pixel_transition_summary_by_size_bucket.csv")
    lines.append("- wfigs_day_of_fire_curve_by_size_bucket.csv")
    lines.append("- wfigs_relative_progress_curve_by_size_bucket.csv")

    (out_dir / "wfigs_viirs_drop_rebound_deep_dive_report.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deep drop/rebound analysis for WFIGS-assigned VIIRS detections.",
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/wfigs_viirs_stats_fullrange_2016_2026_1000ac"),
        help="Directory containing wfigs_assigned_viirs_pixels.csv and wfigs_fire_level_viirs_stats.csv",
    )
    p.add_argument(
        "--out-subdir",
        type=str,
        default="wfigs_viirs_stability_addon_v2",
        help="Subdirectory under --report-dir for outputs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir.resolve()
    out_dir = report_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    fire_df, pix_df = read_inputs(report_dir)
    full = build_daily_timeseries(fire_df, pix_df)
    full = add_transitions(full, value_col="pixel_count", prefix="pix")
    full = add_transitions(full, value_col="frp_total", prefix="frp")

    pixel_drop_events = drop_events(
        full=full,
        value_col="pixel_count",
        prev_col="pix_prev",
        drop_col="pix_is_drop",
        label="pixel_count",
    )
    frp_drop_events = drop_events(
        full=full,
        value_col="frp_total",
        prev_col="frp_prev",
        drop_col="frp_is_drop",
        label="frp_total",
    )
    pixel_drop_events = add_rebound_metrics(full, pixel_drop_events, value_col="pixel_count")
    frp_drop_events = add_rebound_metrics(full, frp_drop_events, value_col="frp_total")

    all_drop_events = pd.concat([pixel_drop_events, frp_drop_events], ignore_index=True)
    drop_summary = summarize_drops(all_drop_events)
    rebound_summary = summarize_rebounds(all_drop_events)
    transition_summary = summarize_transition_types(full)
    pixel_streaks = build_drop_streaks(
        full=full,
        value_col="pixel_count",
        prev_col="pix_prev",
        drop_col="pix_is_drop",
        label="pixel_count",
    )
    frp_streaks = build_drop_streaks(
        full=full,
        value_col="frp_total",
        prev_col="frp_prev",
        drop_col="frp_is_drop",
        label="frp_total",
    )
    drop_streaks = pd.concat([pixel_streaks, frp_streaks], ignore_index=True)
    drop_streak_summary = summarize_drop_streaks(drop_streaks)
    dof_curve = day_of_fire_curve(full)
    rel_curve = relative_curve(full)

    full.to_csv(out_dir / "wfigs_daily_fire_timeseries.csv", index=False)
    pixel_drop_events.to_csv(out_dir / "wfigs_pixel_drop_events.csv", index=False)
    frp_drop_events.to_csv(out_dir / "wfigs_frp_total_drop_events.csv", index=False)
    drop_summary.to_csv(out_dir / "wfigs_drop_summary_by_size_bucket.csv", index=False)
    rebound_summary.to_csv(out_dir / "wfigs_rebound_summary_by_size_bucket.csv", index=False)
    drop_streaks.to_csv(out_dir / "wfigs_drop_streaks.csv", index=False)
    drop_streak_summary.to_csv(out_dir / "wfigs_drop_streak_summary_by_size_bucket.csv", index=False)
    transition_summary.to_csv(out_dir / "wfigs_pixel_transition_summary_by_size_bucket.csv", index=False)
    dof_curve.to_csv(out_dir / "wfigs_day_of_fire_curve_by_size_bucket.csv", index=False)
    rel_curve.to_csv(out_dir / "wfigs_relative_progress_curve_by_size_bucket.csv", index=False)
    write_report(
        out_dir=out_dir,
        full=full,
        drop_summary=drop_summary,
        rebound_summary=rebound_summary,
        transition_summary=transition_summary,
        drop_streak_summary=drop_streak_summary,
        rel_curve=rel_curve,
    )


if __name__ == "__main__":
    main()
