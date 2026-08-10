"""Statistical forecasting baselines (Naive, Seasonal Naive, Moving Average, ARIMA/SARIMAX)."""

from typing import Dict, Optional, Tuple

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


class BaseForecaster:
    """Abstract base class for time series forecasters."""

    def fit(self, series: np.ndarray, exog: Optional[np.ndarray] = None) -> "BaseForecaster":
        return self

    def predict_quantiles(
        self, horizon: int = 24, exog_future: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError


class NaiveForecaster(BaseForecaster):
    """Naive Forecaster: assumes future generation equals the last observed value."""

    def __init__(self, std_scale: float = 10.0):
        self.last_val: float = 0.0
        self.history_std: float = std_scale

    def fit(self, series: np.ndarray, exog: Optional[np.ndarray] = None) -> "NaiveForecaster":
        clean = series[~np.isnan(series)]
        self.last_val = float(clean[-1]) if len(clean) > 0 else 0.0
        self.history_std = float(np.std(np.diff(clean))) if len(clean) > 1 else self.history_std
        return self

    def predict_quantiles(
        self, horizon: int = 24, exog_future: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        p50 = np.full(horizon, self.last_val)
        # Standard error grows with sqrt(horizon)
        uncertainty = 1.28 * self.history_std * np.sqrt(np.arange(1, horizon + 1) / 24.0)
        p10 = np.maximum(0, p50 - uncertainty)
        p90 = p50 + uncertainty
        return {"p10": p10, "p50": p50, "p90": p90}


class SeasonalNaiveForecaster(BaseForecaster):
    """Seasonal Naive Forecaster: assumes future value equals value from 1 seasonal cycle ago (e.g. 24h)."""

    def __init__(self, seasonal_period: int = 24):
        self.seasonal_period = seasonal_period
        self.last_season: np.ndarray = np.array([])
        self.residuals_std: float = 5.0

    def fit(self, series: np.ndarray, exog: Optional[np.ndarray] = None) -> "SeasonalNaiveForecaster":
        clean = series[~np.isnan(series)]
        if len(clean) >= self.seasonal_period:
            self.last_season = clean[-self.seasonal_period :]
            # Estimate residual variance between consecutive seasons
            if len(clean) >= 2 * self.seasonal_period:
                diff = clean[self.seasonal_period :] - clean[: -self.seasonal_period]
                self.residuals_std = float(np.std(diff))
        else:
            self.last_season = np.full(self.seasonal_period, clean[-1] if len(clean) > 0 else 0.0)
        return self

    def predict_quantiles(
        self, horizon: int = 24, exog_future: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        n_repeats = (horizon // self.seasonal_period) + 1
        p50 = np.tile(self.last_season, n_repeats)[:horizon]
        p10 = np.maximum(0, p50 - 1.28 * self.residuals_std)
        p90 = p50 + 1.28 * self.residuals_std
        return {"p10": p10, "p50": p50, "p90": p90}


class MovingAverageForecaster(BaseForecaster):
    """Moving Average Forecaster: predicts mean of last W observations."""

    def __init__(self, window: int = 24):
        self.window = window
        self.ma_val: float = 0.0
        self.history_std: float = 5.0

    def fit(self, series: np.ndarray, exog: Optional[np.ndarray] = None) -> "MovingAverageForecaster":
        clean = series[~np.isnan(series)]
        w = min(len(clean), self.window)
        self.ma_val = float(np.mean(clean[-w:])) if w > 0 else 0.0
        self.history_std = float(np.std(clean[-w:])) if w > 1 else 5.0
        return self

    def predict_quantiles(
        self, horizon: int = 24, exog_future: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        p50 = np.full(horizon, self.ma_val)
        p10 = np.maximum(0, p50 - 1.28 * self.history_std)
        p90 = p50 + 1.28 * self.history_std
        return {"p10": p10, "p50": p50, "p90": p90}


class ARIMAForecaster(BaseForecaster):
    """Auto-Regressive Integrated Moving Average (ARIMA / SARIMAX) forecaster using statsmodels."""

    def __init__(
        self,
        order: Tuple[int, int, int] = (2, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = (1, 0, 1, 24),
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_model = None
        self.last_val: float = 0.0

    def fit(self, series: np.ndarray, exog: Optional[np.ndarray] = None) -> "ARIMAForecaster":
        clean = series[~np.isnan(series)]
        self.last_val = float(clean[-1]) if len(clean) > 0 else 0.0

        # Subsample or fit lightweight SARIMAX to prevent slow training in test suite
        train_data = clean[-336:] if len(clean) > 336 else clean

        try:
            model = SARIMAX(
                train_data,
                order=self.order,
                seasonal_order=self.seasonal_order if len(train_data) >= 48 else None,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.fitted_model = model.fit(disp=False, maxiter=50)
        except Exception:
            # Fallback if SARIMAX optimization fails
            self.fitted_model = None

        return self

    def predict_quantiles(
        self, horizon: int = 24, exog_future: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        if self.fitted_model is not None:
            try:
                forecast_res = self.fitted_model.get_forecast(steps=horizon)
                p50 = forecast_res.predicted_mean
                conf_int = forecast_res.conf_int(alpha=0.2)  # ~P10 to P90
                p10 = np.maximum(0, conf_int[:, 0])
                p90 = conf_int[:, 1]
                return {"p10": p10, "p50": p50, "p90": p90}
            except Exception:
                pass

        # Fallback if fitted model is None or forecast fails
        p50 = np.full(horizon, self.last_val)
        p10 = np.maximum(0, p50 - 10.0)
        p90 = p50 + 10.0
        return {"p10": p10, "p50": p50, "p90": p90}
