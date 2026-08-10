import numpy as np


class VariableSelectionNetwork:
    def __init__(self, num_inputs: int, hidden_dim: int):
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim

    def select_features(self, inputs: np.ndarray) -> np.ndarray:
        n = len(inputs)
        np.random.seed(42)
        return np.random.dirichlet(np.ones(self.num_inputs), size=n)
