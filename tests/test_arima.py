import numpy as np
from src.models.arima_baseline import ARIMAForecaster

def test_arima():
    model = ARIMAForecaster()
    series = np.array([10.0, 12.0, 15.0, 14.0])
    model.fit(series)
    q = model.predict_quantiles(horizon=12)
    assert len(q['p50']) == 12
    assert np.all(q['p90'] >= q['p10'])
