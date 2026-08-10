"""Model interpretability and feature importance analysis module.

NOTE ON INTERPRETABILITY:
Feature importance metrics (gain, split frequency, permutation importance) measure
PREDICTIVE IMPORTANCE—the degree to which a feature improves out-of-sample forecast accuracy.
They DO NOT imply CAUSAL INFLUENCE. For instance, solar irradiance is strongly predictive of solar power output,
but correlated features like temperature also receive predictive weight due to co-linearity.
"""

from typing import Dict, List, Callable, Any
import numpy as np
import pandas as pd


def get_model_feature_importance(model: Any) -> Dict[str, float]:
    """Extract native feature importance dictionary from fitted model."""
    if hasattr(model, "get_feature_importances"):
        return model.get_feature_importances()

    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        feature_names = getattr(model, "feature_names_in_", [f"f_{i}" for i in range(len(imps))])
        return dict(zip(feature_names, imps))

    return {}


def compute_permutation_importance(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    eval_metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_repeats: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute out-of-sample Permutation Feature Importance for any forecaster.

    Parameters
    ----------
    model : fitted forecaster object
    X_val : pd.DataFrame
        Validation feature matrix.
    y_val : pd.Series or np.ndarray
        Validation target values.
    eval_metric_fn : callable(y_true, y_pred) -> float
        Metric function where lower is better (e.g. MAE, RMSE).
    n_repeats : int, default=5
        Number of shuffle repetitions per feature.
    seed : int, default=42
        Random seed.

    Returns
    -------
    dict of str -> float
        Mean increase in error metric when feature column is shuffled.
    """
    np.random.seed(seed)
    y_arr = np.asarray(y_val)

    # Baseline performance
    baseline_preds = model.predict_quantiles(X_val)["p50"]
    baseline_score = eval_metric_fn(y_arr, baseline_preds)

    importance_scores = {}
    X_temp = X_val.copy()

    for col in X_val.columns:
        original_col = X_temp[col].values.copy()
        scores = []

        for _ in range(n_repeats):
            shuffled = np.random.permutation(original_col)
            X_temp[col] = shuffled
            shuffled_preds = model.predict_quantiles(X_temp)["p50"]
            score = eval_metric_fn(y_arr, shuffled_preds)
            scores.append(score - baseline_score)

        importance_scores[col] = float(np.mean(scores))
        X_temp[col] = original_col  # restore column

    # Sort descending by importance score
    sorted_importance = dict(
        sorted(importance_scores.items(), key=lambda item: item[1], reverse=True)
    )
    return sorted_importance
