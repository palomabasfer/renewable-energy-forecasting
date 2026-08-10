import numpy as np
import pandas as pd

from src.evaluation.error_analysis import (
    analyze_diurnal_errors,
    analyze_regime_errors,
    analyze_residual_statistics,
)
from src.evaluation.metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_smape,
    calculate_wape,
    evaluate_forecast_metrics,
)


def test_metrics_calculation():
    y_true = np.array([100.0, 110.0, 105.0])
    y_pred = np.array([98.0, 112.0, 104.0])

    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    smape = calculate_smape(y_true, y_pred)
    wape = calculate_wape(y_true, y_pred)

    assert mae > 0
    assert rmse >= mae
    assert 0 <= smape <= 100
    assert 0 <= wape <= 100

    metrics = evaluate_forecast_metrics(y_true, y_pred, y_pred - 5, y_pred + 5)
    assert "MAE" in metrics
    assert "CRPS" in metrics


def test_error_analysis():
    timestamps = pd.date_range("2026-01-01", periods=48, freq="h")
    y_true = np.random.uniform(50, 150, 48)
    y_pred = y_true + np.random.normal(0, 5, 48)

    stats = analyze_residual_statistics(y_true, y_pred)
    assert "mean_bias" in stats
    assert "durbin_watson" in stats

    diurnal = analyze_diurnal_errors(timestamps, y_true, y_pred)
    assert len(diurnal) == 24
    assert "mae" in diurnal.columns

    regimes = analyze_regime_errors(y_true, y_pred)
    assert "regime" in regimes.columns
