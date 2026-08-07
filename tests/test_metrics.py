import numpy as np
from src.evaluation.metrics import calculate_wape, pinball_loss, crps_score

def test_metrics():
    y_true = np.array([100.0, 110.0, 105.0])
    y_pred = np.array([98.0, 112.0, 104.0])
    wape = calculate_wape(y_true, y_pred)
    assert wape < 0.1
    loss = pinball_loss(y_true, y_pred, 0.5)
    assert loss >= 0.0
    crps = crps_score(y_true, y_pred - 5, y_pred, y_pred + 5)
    assert crps > 0.0
