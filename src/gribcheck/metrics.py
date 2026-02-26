from __future__ import annotations

import math

import numpy as np
from scipy.stats import pearsonr, spearmanr


EPS = 1e-12


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.size == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "pearson_r": float("nan"),
            "spearman_r": float("nan"),
            "r2": float("nan"),
            "slope": float("nan"),
            "intercept": float("nan"),
        }

    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(math.sqrt(np.mean(diff**2)))
    bias = float(np.mean(diff))

    if y_true.size < 2:
        pearson = float("nan")
        spearman = float("nan")
    else:
        try:
            pearson = float(pearsonr(y_true, y_pred).statistic)
        except Exception:
            pearson = float("nan")
        try:
            spearman = float(spearmanr(y_true, y_pred).statistic)
        except Exception:
            spearman = float("nan")

    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    sse = float(np.sum((y_true - y_pred) ** 2))
    r2 = float("nan") if sst < EPS else float(1.0 - sse / sst)
    if sst < EPS:
        slope = float("nan")
        intercept = float("nan")
    else:
        x_mean = float(np.mean(y_true))
        y_mean = float(np.mean(y_pred))
        slope = float(np.sum((y_true - x_mean) * (y_pred - y_mean)) / np.sum((y_true - x_mean) ** 2))
        intercept = float(y_mean - slope * x_mean)

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "pearson_r": pearson,
        "spearman_r": spearman,
        "r2": r2,
        "slope": slope,
        "intercept": intercept,
    }
