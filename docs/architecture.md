# 🏛️ System Architecture & Data Flow

```mermaid
flowchart TD
    A[Grid Telemetry & Weather Sensor Feed] --> B[Data Validation & Integrity Engine]
    B --> C[Temporal Feature Pipeline]
    C --> D1[Naive & Seasonal Naive Baselines]
    C --> D2[SARIMAX Statistical Forecaster]
    C --> D3[XGBoost Quantile Regressor]
    C --> D4[LightGBM Quantile Regressor]
    C --> D5[Chronos Zero-Shot Foundation]
    
    D1 --> E[Quantile Forecast Aggregator P10, P50, P90]
    D2 --> E
    D3 --> E
    D4 --> E
    D5 --> E
    
    E --> F[Walk-Forward Backtesting Framework]
    F --> G[Evaluation Engine: WAPE, MAE, RMSE, CRPS]
    G --> H[Conformal Prediction Interval Calibrator]
    H --> I[Dash Interactive Energy Grid Dashboard]
    I --> J[Dockerized Multi-Stage Container]
```

## Layer Architecture Breakdown

1. **Ingestion & Validation Layer (`src/data/`)**: Validates timestamp monotonicity, checks for missing grid hours, filters negative generation values, and enforces strict physical bounds.
2. **Feature Engineering Layer (`src/features/`)**: Constructs 1-step shifted rolling window statistics, multi-period lags, calendar encodings, and cyclical sine/cosine transformations without target leakage.
3. **Model Framework Layer (`src/models/`)**: Houses modular probabilistic forecasters emitting point forecasts (P50) and uncertainty prediction bounds (P10, P90).
4. **Validation & Evaluation Layer (`src/evaluation/`)**: Implements walk-forward temporal cross-validation, CRPS score calculations, pinball loss evaluation, and conformal interval calibration.
5. **Presentation Layer (`dashboards/`)**: Interactive Dash web application delivering real-time visualization and benchmark tables.
