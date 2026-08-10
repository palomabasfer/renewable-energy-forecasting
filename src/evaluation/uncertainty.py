"""Forecast uncertainty estimation and prediction interval calibration module."""

from typing import Dict, Tuple

import numpy as np


def evaluate_interval_coverage(
    y_true: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray
) -> Dict[str, float]:
    """Evaluate empirical coverage rate and average width of prediction intervals.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth actual values.
    lower_bound : np.ndarray
        Lower bound of prediction interval (e.g., P10).
    upper_bound : np.ndarray
        Upper bound of prediction interval (e.g., P90).

    Returns
    -------
    dict
        Empirical coverage fraction (0.0 to 1.0) and mean interval width.
    """
    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    mean_width = np.mean(upper_bound - lower_bound)

    return {
        "empirical_coverage": round(float(coverage), 4),
        "mean_interval_width": round(float(mean_width), 3),
    }


class ResidualConformalCalibrator:
    """Conformal prediction interval calibrator using historical residual quantiles."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha  # 80% coverage interval for alpha=0.2 (P10 to P90)
        self.lower_quantile_error: float = 0.0
        self.upper_quantile_error: float = 0.0

    def fit(self, y_val_true: np.ndarray, y_val_pred: np.ndarray) -> "ResidualConformalCalibrator":
        """Fit empirical residual quantiles on validation set."""
        residuals = y_val_true - y_val_pred
        self.lower_quantile_error = float(np.quantile(residuals, self.alpha / 2.0))
        self.upper_quantile_error = float(np.quantile(residuals, 1.0 - self.alpha / 2.0))
        return self

    def calibrate_intervals(
        self, point_forecast: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calibrate point forecasts into lower and upper bounds with empirical coverage guarantee."""
        p10 = np.maximum(0.0, point_forecast + self.lower_quantile_error)
        p90 = point_forecast + self.upper_quantile_error
        return p10, p90
