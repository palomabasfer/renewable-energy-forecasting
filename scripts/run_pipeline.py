#!/usr/bin/env python3
"""Master CLI script to execute the end-to-end forecasting pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import clean_telemetry_pipeline, load_raw_telemetry, validate_time_series_data
from src.evaluation.metrics import evaluate_forecast_metrics
from src.features import create_feature_pipeline
from src.models.xgboost_forecaster import XGBoostQuantileForecaster


def main():
    print("🚀 --- EXECUTING RENEWABLE ENERGY FORECASTING PIPELINE ---")

    # 1. Validation
    print("\nStep 1/4: Ingesting & Validating Telemetry...")
    raw_df = load_raw_telemetry(n_hours=1000)
    report = validate_time_series_data(raw_df)
    print(f"Validation Status: {'PASS' if report.is_valid else 'WARN'} ({report.n_observations} hours)")

    # 2. Features
    print("\nStep 2/4: Feature Engineering (Lags, Rolling, Cyclical Encodings)...")
    cleaned_df = clean_telemetry_pipeline(raw_df)
    feat_df = create_feature_pipeline(cleaned_df, drop_na=True)
    feature_cols = [c for c in feat_df.columns if c not in ["timestamp", "total_renewable_mw", "solar_power_mw", "wind_power_mw"]]

    # 3. Model Training
    print("\nStep 3/4: Training XGBoost Quantile Regressor...")
    X_train = feat_df.iloc[:-24][feature_cols]
    y_train = feat_df.iloc[:-24]["total_renewable_mw"]
    X_future = feat_df.iloc[-24:][feature_cols]
    y_actual = feat_df.iloc[-24:]["total_renewable_mw"].values

    model = XGBoostQuantileForecaster(n_estimators=50, max_depth=4)
    model.fit(X_train, y_train)

    # 4. Forecast & Evaluation
    print("\nStep 4/4: Generating 24-Hour Probabilistic Forecast...")
    preds = model.predict_quantiles(X_future)
    metrics = evaluate_forecast_metrics(y_actual, preds["p50"], preds["p10"], preds["p90"])

    print("\n📊 --- 24-HOUR FORECAST EVALUATION METRICS ---")
    for k, v in metrics.items():
        print(f" • {k}: {v}")

    print("\n🎉 Pipeline Execution Completed Successfully!")


if __name__ == "__main__":
    main()
