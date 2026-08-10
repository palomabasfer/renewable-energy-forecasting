import numpy as np
import pandas as pd
from src.models.xgboost_forecaster import XGBoostQuantileForecaster
from src.models.lightgbm_forecaster import LightGBMQuantileForecaster


def test_xgboost_fit_predict():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f_{i}" for i in range(5)])
    y = pd.Series(np.maximum(0, X["f_0"] * 10 + 50 + np.random.randn(100)))

    model = XGBoostQuantileForecaster(n_estimators=10, max_depth=3)
    model.fit(X, y)
    preds = model.predict_quantiles(X.iloc[:10])

    assert "p10" in preds and "p50" in preds and "p90" in preds
    assert len(preds["p50"]) == 10
    assert np.all(preds["p90"] >= preds["p50"])
    assert np.all(preds["p50"] >= preds["p10"])

    importances = model.get_feature_importances()
    assert len(importances) == 5


def test_lightgbm_fit_predict():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f_{i}" for i in range(5)])
    y = pd.Series(np.maximum(0, X["f_0"] * 10 + 50 + np.random.randn(100)))

    model = LightGBMQuantileForecaster(n_estimators=10, max_depth=3)
    model.fit(X, y)
    preds = model.predict_quantiles(X.iloc[:10])

    assert "p10" in preds and "p50" in preds and "p90" in preds
    assert len(preds["p50"]) == 10
    assert np.all(preds["p90"] >= preds["p50"])
    assert np.all(preds["p50"] >= preds["p10"])
