import numpy as np
from src.evaluation.uncertainty import evaluate_interval_coverage, ResidualConformalCalibrator


def test_evaluate_interval_coverage():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    p10 = np.array([5.0, 15.0, 25.0, 35.0])
    p90 = np.array([15.0, 25.0, 35.0, 45.0])

    res = evaluate_interval_coverage(y_true, p10, p90)
    assert res["empirical_coverage"] == 1.0
    assert res["mean_interval_width"] == 10.0


def test_residual_conformal_calibrator():
    np.random.seed(42)
    y_val = np.random.uniform(50, 150, 200)
    y_pred = y_val + np.random.normal(0, 5, 200)

    calibrator = ResidualConformalCalibrator(alpha=0.2)
    calibrator.fit(y_val, y_pred)

    test_pred = np.array([100.0, 120.0])
    p10, p90 = calibrator.calibrate_intervals(test_pred)

    assert len(p10) == 2
    assert len(p90) == 2
    assert np.all(p90 > p10)
