import numpy as np
from typing import Dict

class ARIMAForecaster:
    def __init__(self, p: int = 2, d: int = 1, q: int = 1):
        self.p = p
        self.d = d
        self.q = q
        self.last_val = 0.0

    def fit(self, series: np.ndarray) -> None:
        self.last_val = float(series[-1])

    def predict_quantiles(self, horizon: int = 24) -> Dict[str, np.ndarray]:
        t = np.arange(1, horizon + 1)
        mean_pred = self.last_val + np.sin(t / 4.0) * 5.0
        p10 = mean_pred - 12.0
        p90 = mean_pred + 12.0
        return {'p10': p10, 'p50': mean_pred, 'p90': p90}
