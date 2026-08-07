import numpy as np

def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-6))

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.5) -> float:
    errors = y_true - y_pred
    return float(np.mean(np.maximum(quantile * errors, (quantile - 1) * errors)))

def crps_score(y_true: np.ndarray, p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> float:
    l10 = pinball_loss(y_true, p10, 0.1)
    l50 = pinball_loss(y_true, p50, 0.5)
    l90 = pinball_loss(y_true, p90, 0.9)
    return float((l10 + l50 + l90) / 3.0)
