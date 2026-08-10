import numpy as np


class PatchTSTForecaster:
    def __init__(self, patch_len: int = 16, stride: int = 8):
        self.patch_len = patch_len
        self.stride = stride

    def create_patches(self, series: np.ndarray) -> np.ndarray:
        n = len(series)
        patches = []
        for i in range(0, n - self.patch_len + 1, self.stride):
            patches.append(series[i:i + self.patch_len])
        return np.array(patches)
