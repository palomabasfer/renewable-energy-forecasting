"""Diagnostic error analysis module for forecast residual breakdown."""

from typing import Dict, Any
import numpy as np
import pandas as pd


def analyze_residual_statistics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute detailed residual distribution statistics.

    Residual = y_true - y_pred
    """
    residuals = y_true - y_pred
    mean_bias = float(np.mean(residuals))
    std_residual = float(np.std(residuals))
    skew = float(pd.Series(residuals).skew())
    kurtosis = float(pd.Series(residuals).kurtosis())

    # Durbin-Watson statistic for residual autocorrelation
    diff_res = np.diff(residuals)
    dw_stat = float(np.sum(diff_res ** 2) / (np.sum(residuals ** 2) + 1e-6))

    return {
        "mean_bias": round(mean_bias, 3),
        "std_residual": round(std_residual, 3),
        "skewness": round(skew, 3),
        "kurtosis": round(kurtosis, 3),
        "durbin_watson": round(dw_stat, 3),
    }


def analyze_diurnal_errors(
    timestamps: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> pd.DataFrame:
    """Break down forecast MAE and RMSE by hour of the day (0..23)."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "actual": y_true,
            "pred": y_pred,
        }
    )
    df["hour"] = df["timestamp"].dt.hour
    df["abs_error"] = np.abs(df["actual"] - df["pred"])
    df["sq_error"] = (df["actual"] - df["pred"]) ** 2

    grouped = (
        df.groupby("hour")
        .agg(
            mae=("abs_error", "mean"),
            rmse=("sq_error", lambda x: np.sqrt(np.mean(x))),
            count=("actual", "count"),
        )
        .reset_index()
        .round(3)
    )
    return grouped


def analyze_regime_errors(
    y_true: np.ndarray, y_pred: np.ndarray, capacity_mw: float = 250.0
) -> pd.DataFrame:
    """Break down forecast errors by power generation regime (Low, Medium, High)."""
    abs_errors = np.abs(y_true - y_pred)
    regimes = []
    for val in y_true:
        ratio = val / (capacity_mw + 1e-6)
        if ratio < 0.2:
            regimes.append("Low (<20%)")
        elif ratio < 0.7:
            regimes.append("Medium (20-70%)")
        else:
            regimes.append("High (>70%)")

    df = pd.DataFrame({"regime": regimes, "actual": y_true, "abs_error": abs_errors})

    return (
        df.groupby("regime")
        .agg(
            mean_actual=("actual", "mean"),
            mae=("abs_error", "mean"),
            count=("actual", "count"),
        )
        .reset_index()
        .round(3)
    )
