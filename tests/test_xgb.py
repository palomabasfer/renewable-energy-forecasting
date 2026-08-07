import numpy as np
from src.models.xgboost_forecaster import XGBoostQuantileForecaster

def test_xgboost():
    model = XGBoostQuantileForecaster()
    q = model.predict_quantiles(np.zeros((10, 5)))
    assert len(q['p50']) == 10
