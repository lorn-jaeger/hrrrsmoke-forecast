#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create simple WFIGS/VIIRS plots using continuous fire size (no fixed size buckets)."
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("reports/wfigs_viirs_stats_fullrange_2016_2026_1000ac/wfigs_viirs_stability_addon_v2"),
        help="Directory with deep-dive CSV outputs.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots (defaults to <input-dir>/figures_continuous_size).",
    )
    return p.parse_args()


def rolling_smooth(x: np.ndarray, y: np.ndarray, frac: float = 0.08, min_window: int = 75) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return np.array([]), np.array([])
    xs = x[m]
    ys = y[m]
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    n = len(xs)
    win = max(min_window, int(n * frac))
    if win % 2 == 0:
        win += 1
    if win > n:
        win = n if n % 2 == 1 else max(1, n - 1)

    if win <= 3:
        return xs, ys

    x_sm = pd.Series(xs).rolling(win, center=True, min_periods=max(5, win // 5)).mean().to_numpy()
    y_sm = pd.Series(ys).rolling(win, center=True, min_periods=max(5, win // 5)).mean().to_numpy()
    k = np.isfinite(x_sm) & np.isfinite(y_sm)
    return x_sm[k], y_sm[k]


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(input_dir / "wfigs_daily_fire_timeseries.csv", low_memory=False)
    pixel_drops = pd.read_csv(input_dir / "wfigs_pixel_drop_events.csv", low_memory=False)
    frp_drops = pd.read_csv(input_dir / "wfigs_frp_total_drop_events.csv", low_memory=False)
    return daily, pixel_drops, frp_drops


def build_fire_metrics(daily: pd.DataFrame, pixel_drops: pd.DataFrame, frp_drops: pd.DataFrame) -> pd.DataFrame:
    base = (
        daily.groupby("fire_idx", as_index=True)
        .agg(
            size_acres=("size_acres", "first"),
            active_window_days=("active_window_days", "first"),
            mean_pixels_per_day=("pixel_count", "mean"),
            mean_pixels_on_detect_days=("pixel_count", lambda s: float(s[s > 0].mean()) if np.any(s > 0) else 0.0),
            mean_frp_total_per_day=("frp_total", "mean"),
            mean_frp_total_on_detect_days=("frp_total", lambda s: float(s[s > 0].mean()) if np.any(s > 0) else 0.0),
        )
        .copy()
    )

    comp = (
        daily[daily["pix_prev"].notna()]
        .groupby("fire_idx", as_index=True)
        .agg(
            pixel_drop_rate=("pix_is_drop", "mean"),
            frp_drop_rate=("frp_is_drop", "mean"),
            pixel_increase_rate=("pix_is_increase", "mean"),
            frp_increase_rate=("frp_is_increase", "mean"),
        )
    )

    pix_rb = (
        pixel_drops.groupby("fire_idx", as_index=True)
        .agg(
            pixel_drop_events=("fire_idx", "size"),
            pixel_rebound7_rate=("rebound_within_7d", "mean"),
            pixel_recover7_rate=("recover_pre_drop_within_7d", "mean"),
            pixel_drop_to_zero_rate=("drop_to_zero", "mean"),
            pixel_drop_pct_median=("drop_pct", "median"),
        )
        .copy()
    )

    frp_rb = (
        frp_drops.groupby("fire_idx", as_index=True)
        .agg(
            frp_drop_events=("fire_idx", "size"),
            frp_rebound7_rate=("rebound_within_7d", "mean"),
            frp_recover7_rate=("recover_pre_drop_within_7d", "mean"),
            frp_drop_to_zero_rate=("drop_to_zero", "mean"),
            frp_drop_pct_median=("drop_pct", "median"),
        )
        .copy()
    )

    out = base.join(comp, how="left").join(pix_rb, how="left").join(frp_rb, how="left")
    fill_zero = [
        "pixel_drop_rate",
        "frp_drop_rate",
        "pixel_increase_rate",
        "frp_increase_rate",
        "pixel_drop_events",
        "pixel_rebound7_rate",
        "pixel_recover7_rate",
        "pixel_drop_to_zero_rate",
        "pixel_drop_pct_median",
        "frp_drop_events",
        "frp_rebound7_rate",
        "frp_recover7_rate",
        "frp_drop_to_zero_rate",
        "frp_drop_pct_median",
    ]
    for c in fill_zero:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)

    out = out[out["size_acres"] > 0].copy()
    out["log10_size"] = np.log10(out["size_acres"].astype(float))
    out.reset_index(inplace=True)
    return out


def build_progress_profile(daily: pd.DataFrame) -> pd.DataFrame:
    x = daily.copy()
    x["prog_bin"] = np.minimum((x["rel_progress"].clip(0, 1) * 100).astype(int), 99)
    prof = (
        x.groupby("prog_bin", as_index=False)
        .agg(
            fire_days=("fire_idx", "size"),
            days_with_detections=("pixel_count", lambda s: int((s > 0).sum())),
            mean_pixels=("pixel_count", "mean"),
            p90_pixels=("pixel_count", lambda s: float(s.quantile(0.9))),
            mean_frp_total=("frp_total", "mean"),
            p90_frp_total=("frp_total", lambda s: float(s.quantile(0.9))),
            pixel_drop_rate=("pix_is_drop", "mean"),
            frp_drop_rate=("frp_is_drop", "mean"),
        )
        .copy()
    )
    prof["pct_with_detections"] = prof["days_with_detections"] / prof["fire_days"] * 100.0
    prof["progress_mid"] = (prof["prog_bin"] + 0.5) / 100.0
    return prof.sort_values("prog_bin").reset_index(drop=True)


def plot_progress_profile(profile: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    x = profile["progress_mid"].to_numpy()

    axes[0, 0].plot(x, profile["mean_pixels"], lw=2.0, color="#1f77b4", label="Mean pixels/day")
    axes[0, 0].plot(x, profile["p90_pixels"], lw=1.5, color="#6baed6", label="P90 pixels/day")
    axes[0, 0].set_title("Pixels Over Relative Fire Progress")
    axes[0, 0].set_xlabel("Relative progress (0=start, 1=end)")
    axes[0, 0].set_ylabel("Pixel count")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(x, profile["mean_frp_total"], lw=2.0, color="#d62728", label="Mean FRP total/day")
    axes[0, 1].plot(x, profile["p90_frp_total"], lw=1.5, color="#f28e8c", label="P90 FRP total/day")
    axes[0, 1].set_title("FRP Total Over Relative Fire Progress")
    axes[0, 1].set_xlabel("Relative progress (0=start, 1=end)")
    axes[0, 1].set_ylabel("FRP total (MW/day)")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(x, profile["pct_with_detections"], lw=2.0, color="#2ca02c")
    axes[1, 0].set_title("Detection-Day Rate Over Relative Progress")
    axes[1, 0].set_xlabel("Relative progress")
    axes[1, 0].set_ylabel("% days with detections")
    axes[1, 0].set_ylim(0, max(100, profile["pct_with_detections"].max() * 1.1))

    axes[1, 1].plot(x, profile["pixel_drop_rate"] * 100.0, lw=2.0, color="#9467bd", label="Pixel drop-day rate")
    axes[1, 1].plot(x, profile["frp_drop_rate"] * 100.0, lw=1.7, color="#8c564b", label="FRP drop-day rate")
    axes[1, 1].set_title("Drop-Day Rate Over Relative Progress")
    axes[1, 1].set_xlabel("Relative progress")
    axes[1, 1].set_ylabel("Drop-day rate (%)")
    axes[1, 1].set_ylim(bottom=0)
    axes[1, 1].legend(frameon=False)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def scatter_with_smooth(ax, x, y, color, title, ylab, ylim=None, frac: float = 0.08) -> None:
    ax.scatter(x, y, s=12, alpha=0.2, color=color, edgecolors="none")
    xs, ys = rolling_smooth(x, y, frac=frac)
    if len(xs):
        ax.plot(xs, ys, color="black", lw=2.0)
    ax.set_title(title)
    ax.set_xlabel("log10(Fire size acres)")
    ax.set_ylabel(ylab)
    if ylim is not None:
        ax.set_ylim(*ylim)


def plot_size_relationships(fire_metrics: pd.DataFrame, out_pixels_path: Path, out_frp_path: Path) -> None:
    x = fire_metrics["log10_size"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    scatter_with_smooth(
        axes[0],
        x,
        fire_metrics["mean_pixels_per_day"].to_numpy(),
        color="#1f77b4",
        title="Mean Pixels/Day vs Fire Size",
        ylab="Mean pixels/day",
    )
    axes[0].set_yscale("log")

    scatter_with_smooth(
        axes[1],
        x,
        (fire_metrics["pixel_drop_rate"] * 100.0).to_numpy(),
        color="#9467bd",
        title="Pixel Drop Rate vs Fire Size",
        ylab="Drop-day rate (%)",
        ylim=(0, 100),
    )

    scatter_with_smooth(
        axes[2],
        x,
        (fire_metrics["pixel_rebound7_rate"] * 100.0).to_numpy(),
        color="#2ca02c",
        title="Pixel Rebound<=7d vs Fire Size",
        ylab="Rebound<=7d after drop (%)",
        ylim=(0, 100),
    )

    fig.savefig(out_pixels_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    scatter_with_smooth(
        axes[0],
        x,
        fire_metrics["mean_frp_total_per_day"].to_numpy(),
        color="#d62728",
        title="Mean FRP Total/Day vs Fire Size",
        ylab="Mean FRP total/day (MW)",
    )
    axes[0].set_yscale("log")

    scatter_with_smooth(
        axes[1],
        x,
        (fire_metrics["frp_drop_rate"] * 100.0).to_numpy(),
        color="#8c564b",
        title="FRP Drop Rate vs Fire Size",
        ylab="Drop-day rate (%)",
        ylim=(0, 100),
    )

    scatter_with_smooth(
        axes[2],
        x,
        (fire_metrics["frp_rebound7_rate"] * 100.0).to_numpy(),
        color="#17becf",
        title="FRP Rebound<=7d vs Fire Size",
        ylab="Rebound<=7d after drop (%)",
        ylim=(0, 100),
    )

    fig.savefig(out_frp_path, dpi=180)
    plt.close(fig)


def plot_progress_size_hexbin(
    daily: pd.DataFrame,
    pixel_drops: pd.DataFrame,
    fire_metrics: pd.DataFrame,
    out_path: Path,
) -> None:
    size_map = fire_metrics.set_index("fire_idx")["log10_size"]

    d = daily[daily["pix_prev"].notna()].copy()
    d["log10_size"] = d["fire_idx"].map(size_map)
    d = d[d["log10_size"].notna()].copy()

    pdrops = pixel_drops.copy()
    pdrops["log10_size"] = pdrops["fire_idx"].map(size_map)
    pdrops = pdrops[pdrops["log10_size"].notna()].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

    hb1 = axes[0].hexbin(
        d["rel_progress"],
        d["log10_size"],
        C=d["pix_is_drop"].astype(float),
        reduce_C_function=np.mean,
        gridsize=38,
        cmap="viridis",
        mincnt=20,
        vmin=0,
        vmax=1,
    )
    axes[0].set_title("Pixel Drop Probability")
    axes[0].set_xlabel("Relative progress")
    axes[0].set_ylabel("log10(Fire size acres)")
    cb1 = fig.colorbar(hb1, ax=axes[0])
    cb1.set_label("P(drop day)")

    hb2 = axes[1].hexbin(
        pdrops["rel_progress"],
        pdrops["log10_size"],
        C=pdrops["rebound_within_7d"].astype(float),
        reduce_C_function=np.mean,
        gridsize=38,
        cmap="plasma",
        mincnt=10,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Rebound<=7d Probability (After Pixel Drop)")
    axes[1].set_xlabel("Relative progress")
    axes[1].set_ylabel("log10(Fire size acres)")
    cb2 = fig.colorbar(hb2, ax=axes[1])
    cb2.set_label("P(rebound<=7d)")

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_plot_readme(out_dir: Path) -> None:
    lines = [
        "# Continuous-Size Plot Notes",
        "",
        "- Fire size is treated as a continuous variable using `log10(acres)`.",
        "- No fixed fire-size class buckets are used in these figures.",
        "- Smooth black curves are running means over sorted fire size.",
        "",
        "Files:",
        "- progress_profiles_simple.png",
        "- pixel_metrics_vs_fire_size_continuous.png",
        "- frp_metrics_vs_fire_size_continuous.png",
        "- drop_rebound_prob_progress_size_hexbin.png",
        "- plot_fire_metrics_continuous.csv",
        "- plot_progress_profile.csv",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    out_dir = (args.out_dir.resolve() if args.out_dir else (input_dir / "figures_continuous_size"))
    out_dir.mkdir(parents=True, exist_ok=True)

    daily, pixel_drops, frp_drops = load_inputs(input_dir)
    fire_metrics = build_fire_metrics(daily, pixel_drops, frp_drops)
    progress_profile = build_progress_profile(daily)

    fire_metrics.to_csv(out_dir / "plot_fire_metrics_continuous.csv", index=False)
    progress_profile.to_csv(out_dir / "plot_progress_profile.csv", index=False)

    plot_progress_profile(progress_profile, out_dir / "progress_profiles_simple.png")
    plot_size_relationships(
        fire_metrics,
        out_pixels_path=out_dir / "pixel_metrics_vs_fire_size_continuous.png",
        out_frp_path=out_dir / "frp_metrics_vs_fire_size_continuous.png",
    )
    plot_progress_size_hexbin(
        daily=daily,
        pixel_drops=pixel_drops,
        fire_metrics=fire_metrics,
        out_path=out_dir / "drop_rebound_prob_progress_size_hexbin.png",
    )
    write_plot_readme(out_dir)


if __name__ == "__main__":
    main()
