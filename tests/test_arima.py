import numpy as np

from src.models.arima_baseline import (
    ARIMAForecaster,
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)


def test_naive_forecaster():
    series = np.arange(100, 200, dtype=float)
    model = NaiveForecaster()
    model.fit(series)
    q = model.predict_quantiles(horizon=24)
    assert len(q["p50"]) == 24
    assert np.all(q["p50"] == 199.0)
    assert np.all(q["p90"] >= q["p10"])


def test_seasonal_naive_forecaster():
    series = np.tile(np.sin(np.linspace(0, 2 * np.pi, 24)), 10) + 50.0
    model = SeasonalNaiveForecaster(seasonal_period=24)
    model.fit(series)
    q = model.predict_quantiles(horizon=24)
    assert len(q["p50"]) == 24
    assert np.all(q["p90"] >= q["p10"])


def test_moving_average_forecaster():
    series = np.ones(100) * 50.0
    model = MovingAverageForecaster(window=24)
    model.fit(series)
    q = model.predict_quantiles(horizon=12)
    assert len(q["p50"]) == 12
    assert np.all(q["p50"] == 50.0)


def test_arima_forecaster():
    series = np.sin(np.linspace(0, 4 * np.pi, 100)) * 20.0 + 100.0
    model = ARIMAForecaster(order=(1, 0, 0), seasonal_order=None)
    model.fit(series)
    q = model.predict_quantiles(horizon=12)
    assert len(q["p50"]) == 12
    assert np.all(q["p90"] >= q["p10"])
