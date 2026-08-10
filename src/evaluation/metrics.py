"""Evaluation metrics for time series point and probabilistic forecasts."""

from typing import Dict, Union, Optional
import numpy as np


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (MAE)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (sMAPE in %)."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + 1e-6
    return float(np.mean(np.abs(y_pred - y_true) / denominator) * 100.0)


def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (WAPE in %)."""
    total_abs_actual = np.sum(np.abs(y_true)) + 1e-6
    return float((np.sum(np.abs(y_true - y_pred)) / total_abs_actual) * 100.0)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.5) -> float:
    """Pinball Loss (Quantile Loss) for a given quantile alpha.

    L_q(y, f) = max(q * (y - f), (q - 1) * (y - f))
    """
    errors = y_true - y_pred
    return float(np.mean(np.maximum(quantile * errors, (quantile - 1) * errors)))


def crps_score(
    y_true: np.ndarray, p10: np.ndarray, p50: np.ndarray, p90: np.ndarray
) -> float:
    """Continuous Ranked Probability Score (CRPS) approximation via 3 quantiles.

    CRPS ≈ (Pinball_0.1 + Pinball_0.5 + Pinball_0.9) / 3.0
    """
    l10 = pinball_loss(y_true, p10, 0.1)
    l50 = pinball_loss(y_true, p50, 0.5)
    l90 = pinball_loss(y_true, p90, 0.9)
    return float((l10 + l50 + l90) / 3.0)


def evaluate_forecast_metrics(
    y_true: np.ndarray,
    p50: np.ndarray,
    p10: Optional[np.ndarray] = None,
    p90: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute comprehensive point and probabilistic evaluation metrics."""
    metrics = {
        "MAE": round(calculate_mae(y_true, p50), 3),
        "RMSE": round(calculate_rmse(y_true, p50), 3),
        "sMAPE": round(calculate_smape(y_true, p50), 2),
        "WAPE": round(calculate_wape(y_true, p50), 2),
    }

    if p10 is not None and p90 is not None:
        metrics["Pinball_P10"] = round(pinball_loss(y_true, p10, 0.1), 3)
        metrics["Pinball_P50"] = round(pinball_loss(y_true, p50, 0.5), 3)
        metrics["Pinball_P90"] = round(pinball_loss(y_true, p90, 0.9), 3)
        metrics["CRPS"] = round(crps_score(y_true, p10, p50, p90), 3)

    return metrics
