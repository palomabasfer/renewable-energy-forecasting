# 🎴 Model Card — Probabilistic Renewable Energy Forecasters

## Model Details
- **Developer**: Paloma Bas Fernández
- **Model Type**: Multi-Quantile Gradient Boosted Decision Trees (XGBoost & LightGBM) + SARIMAX / Chronos Zero-Shot
- **Model Version**: 1.0.0
- **License**: MIT

## Intended Use
- **Primary Use**: Day-ahead (24-hour horizon) probabilistic wind and solar energy generation forecasting for electrical grid integration.
- **Target Audience**: Energy market operators, transmission system operators (TSOs), renewable asset managers, and energy traders.
- **Out-of-Scope Use**: Long-term climate projection (>1 year ahead) or microsecond power quality transient stability analysis.

## Training & Validation Data
- **Source**: Grid telemetry feeds with regional weather numerical weather predictions (NWP).
- **Sampling Frequency**: Hourly (`1h`).
- **Validation Strategy**: 14-day expanding walk-forward cross-validation without temporal data leakage.

## Quantitative Metrics (24-Hour Horizon)
- **WAPE**: 6.4% – 8.6%
- **CRPS Score**: 3.65 – 4.82
- **Pinball Loss (P10)**: 1.35 – 1.85
- **Pinball Loss (P90)**: 1.52 – 2.10

## Risk Mitigation & Assumptions
- **Non-Negativity**: Output bounds are strictly clipped at $\ge 0.0$ MW.
- **Conformal Calibration**: Residual quantiles ensure empirical 80% coverage interval reliability ($P10 \le y \le P90$).
