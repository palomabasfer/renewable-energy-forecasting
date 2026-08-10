import numpy as np


def generate_fan_chart_series(p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> dict:
    return {'p10': p10.tolist(), 'p50': p50.tolist(), 'p90': p90.tolist()}
