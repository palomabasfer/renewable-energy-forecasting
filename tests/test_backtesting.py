import numpy as np
import pandas as pd
from src.data.synthetic_grid_data import generate_renewable_timeseries
from src.features.engineering import create_feature_pipeline
from src.models.xgboost_forecaster import XGBoostQuantileForecaster
from src.evaluation.backtesting import WalkForwardBacktester


def test_walk_forward_backtest():
    df = generate_renewable_timeseries(n_hours=500)
    feat_df = create_feature_pipeline(df, drop_na=True)
    feature_cols = [c for c in feat_df.columns if c not in ["timestamp", "total_renewable_mw", "solar_power_mw", "wind_power_mw"]]

    backtester = WalkForwardBacktester(
        train_window_hours=200,
        test_window_hours=24,
        step_hours=24,
        min_train_hours=100,
    )

    def model_factory():
        return XGBoostQuantileForecaster(n_estimators=10, max_depth=3)

    res = backtester.backtest_tabular_model(
        model_factory=model_factory,
        df=feat_df,
        feature_cols=feature_cols,
        target_col="total_renewable_mw",
        time_col="timestamp",
    )

    assert res["n_windows"] > 0
    assert "mean_mae" in res
    assert "mean_crps" in res
    assert len(res["p50"]) == res["n_windows"] * 24
    assert np.all(res["p90"] >= res["p10"])
