"""Utility module for setting global random seeds for reproducibility."""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set global random seed across Python standard library, NumPy, and environment."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
