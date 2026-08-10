"""XGBoost probabilistic quantile forecaster module with robust fallback support."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

HAS_XGBOOST = False
try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except Exception:
    from sklearn.ensemble import GradientBoostingRegressor

    HAS_XGBOOST = False


class XGBoostQuantileForecaster:
    """Multi-quantile probabilistic forecaster powered by XGBoost (with Scikit-Learn GradientBoosting fallback).

    Fits three independent regressors for target quantiles (P10, P50, P90).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        quantiles: Optional[List[float]] = None,
        random_state: int = 42,
    ):
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.quantiles = quantiles
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.feature_names: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None,
    ) -> "XGBoostQuantileForecaster":
        """Train quantile regressors on feature matrix X and target y."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y)

        for q in self.quantiles:
            q_key = f"p{int(q * 100)}"
            if HAS_XGBOOST:
                model = XGBRegressor(
                    objective="reg:quantileerror",
                    quantile_alpha=q,
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
            else:
                model = GradientBoostingRegressor(
                    loss="quantile",
                    alpha=q,
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                )
            model.fit(X_arr, y_arr)
            self.models[q_key] = model

        return self

    def predict_quantiles(self, X_future: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate quantile forecasts for future features."""
        if isinstance(X_future, pd.DataFrame):
            X_arr = X_future.values
        else:
            X_arr = np.asarray(X_future)

        preds = {}
        for q in self.quantiles:
            q_key = f"p{int(q * 100)}"
            if q_key in self.models:
                p = self.models[q_key].predict(X_arr)
                preds[q_key] = np.maximum(0, p)

        # Enforce monotonic quantile order: P10 <= P50 <= P90
        if "p10" in preds and "p50" in preds and "p90" in preds:
            preds["p10"] = np.minimum(preds["p10"], preds["p50"])
            preds["p90"] = np.maximum(preds["p90"], preds["p50"])

        return preds

    def get_feature_importances(self) -> Dict[str, float]:
        """Extract feature importances from P50 median model."""
        if "p50" not in self.models or not self.feature_names:
            return {}
        imp = self.models["p50"].feature_importances_
        return dict(zip(self.feature_names, imp))
