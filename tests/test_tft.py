from src.models.tft_forecaster import TemporalFusionTransformer

def test_tft():
    model = TemporalFusionTransformer()
    q = model.predict_quantiles(horizon=24)
    assert len(q['p50']) == 24
