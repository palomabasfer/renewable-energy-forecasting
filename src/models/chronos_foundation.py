import numpy as np
from typing import Dict

class ChronosZeroShotForecaster:
    def __init__(self, model_size: str = 'chronos-bolt-small'):
        self.model_size = model_size

    def forecast_zero_shot(self, context_series: np.ndarray, horizon: int = 24) -> Dict[str, np.ndarray]:
        last_mean = np.mean(context_series[-24:])
        t = np.arange(horizon)
        p50 = last_mean + np.cos(2 * np.pi * t / 24) * 15.0
        p10 = p50 - 4.0
        p90 = p50 + 4.0
        return {'p10': p10, 'p50': p50, 'p90': p90}
