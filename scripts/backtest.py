#!/usr/bin/env python3
"""CLI script to run walk-forward model backtest benchmark."""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_raw_telemetry, clean_telemetry_pipeline
from src.features import create_feature_pipeline
from src.models.xgboost_forecaster import XGBoostQuantileForecaster
from src.models.lightgbm_forecaster import LightGBMQuantileForecaster
from src.evaluation.backtesting import WalkForwardBacktester


def main():
    print("📈 Ingesting and processing telemetry data for backtest benchmark...")
    raw_df = load_raw_telemetry(n_hours=1200)
    cleaned_df = clean_telemetry_pipeline(raw_df)
    feat_df = create_feature_pipeline(cleaned_df, drop_na=True)

    feature_cols = [c for c in feat_df.columns if c not in ["timestamp", "total_renewable_mw", "solar_power_mw", "wind_power_mw"]]

    backtester = WalkForwardBacktester(
        train_window_hours=336,
        test_window_hours=24,
        step_hours=24,
        min_train_hours=168,
    )

    models_to_evaluate = {
        "XGBoost Quantile": lambda: XGBoostQuantileForecaster(n_estimators=50, max_depth=4),
        "LightGBM Quantile": lambda: LightGBMQuantileForecaster(n_estimators=50, max_depth=4),
    }

    results = []

    for name, factory in models_to_evaluate.items():
        print(f"🔄 Executing walk-forward backtest for {name}...")
        res = backtester.backtest_tabular_model(
            model_factory=factory,
            df=feat_df,
            feature_cols=feature_cols,
            target_col="total_renewable_mw",
        )
        results.append({
            "Model Architecture": name,
            "WAPE (%)": f"{res['mean_wape']}%",
            "MAE (MW)": res["mean_mae"],
            "RMSE (MW)": res["mean_rmse"],
            "CRPS Score": res["mean_crps"],
            "Windows": res["n_windows"],
        })

    summary_df = pd.DataFrame(results)
    print("\n🏆 --- MODEL BACKTEST BENCHMARK SUMMARY ---")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
