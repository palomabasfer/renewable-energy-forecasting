"""Model interpretability package."""

from src.interpretation.explainability import (
    get_model_feature_importance,
    compute_permutation_importance,
)

__all__ = [
    "get_model_feature_importance",
    "compute_permutation_importance",
]
