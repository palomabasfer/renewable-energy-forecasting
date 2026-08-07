import numpy as np
from src.models.grn_module import GatedResidualNetwork
from src.models.vsn_module import VariableSelectionNetwork
from src.models.patchtst_forecaster import PatchTSTForecaster

def test_grn():
    grn = GatedResidualNetwork(10, 20)
    out = grn.forward(np.array([1.0, 2.0]))
    assert out.shape == (2,)

def test_vsn():
    vsn = VariableSelectionNetwork(5, 10)
    weights = vsn.select_features(np.zeros((3, 5)))
    assert weights.shape == (3, 5)

def test_patchtst():
    p = PatchTSTForecaster(16, 8)
    patches = p.create_patches(np.zeros(100))
    assert len(patches) > 0
