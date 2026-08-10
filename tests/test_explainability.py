import numpy as np
import pandas as pd
from src.models.xgboost_forecaster import XGBoostQuantileForecaster
from src.interpretation.explainability import (
    get_model_feature_importance,
    compute_permutation_importance,
)
from src.evaluation.metrics import calculate_mae


def test_permutation_importance():
    np.random.seed(42)
    X = pd.DataFrame({
        "informative": np.random.randn(100),
        "noise": np.random.randn(100),
    })
    y = pd.Series(X["informative"] * 20.0 + 50.0 + np.random.normal(0, 1, 100))

    model = XGBoostQuantileForecaster(n_estimators=10, max_depth=3)
    model.fit(X, y)

    imps = get_model_feature_importance(model)
    assert len(imps) == 2

    perm_imps = compute_permutation_importance(model, X, y, eval_metric_fn=calculate_mae, n_repeats=2)
    assert len(perm_imps) == 2
    assert perm_imps["informative"] >= perm_imps["noise"]
