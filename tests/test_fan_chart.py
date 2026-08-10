import numpy as np

from src.evaluation.fan_chart import generate_fan_chart_series


def test_fan_chart():
    res = generate_fan_chart_series(np.array([1]), np.array([2]), np.array([3]))
    assert res['p10'] == [1]
