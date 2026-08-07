import numpy as np

class GatedResidualNetwork:
    def __init__(self, input_dim: int, hidden_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x) * (1.0 / (1.0 + np.exp(-x)))
