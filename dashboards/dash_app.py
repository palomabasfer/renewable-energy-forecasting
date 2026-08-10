"""Interactive Enterprise Renewable Energy Probabilistic Forecasting Dashboard."""

import sys
from pathlib import Path

# Add project root to Python module search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from src.data import generate_renewable_timeseries, validate_time_series_data
from src.features import create_feature_pipeline
from src.models.xgboost_forecaster import XGBoostQuantileForecaster
from src.models.lightgbm_forecaster import LightGBMQuantileForecaster
from src.models.arima_baseline import SeasonalNaiveForecaster, ARIMAForecaster
from src.models.chronos_foundation import ChronosZeroShotForecaster
from src.evaluation.metrics import evaluate_forecast_metrics

# Initialize Dash application
app = dash.Dash(
    __name__,
    title="Renewable Energy Forecasting Lab",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

# Generate initial dataset
raw_df = generate_renewable_timeseries(n_hours=1000, seed=42)
feat_df = create_feature_pipeline(raw_df, drop_na=True)
feature_cols = [c for c in feat_df.columns if c not in ["timestamp", "total_renewable_mw", "solar_power_mw", "wind_power_mw"]]

# Fit models once for dashboard demo
X_train = feat_df.iloc[:-24][feature_cols]
y_train = feat_df.iloc[:-24]["total_renewable_mw"]
X_future = feat_df.iloc[-24:][feature_cols]
y_actual = feat_df.iloc[-24:]["total_renewable_mw"].values
future_timestamps = feat_df.iloc[-24:]["timestamp"]

models_registry = {
    "XGBoost Quantile": XGBoostQuantileForecaster(n_estimators=50, max_depth=4).fit(X_train, y_train),
    "LightGBM Quantile": LightGBMQuantileForecaster(n_estimators=50, max_depth=4).fit(X_train, y_train),
    "Seasonal Naive": SeasonalNaiveForecaster(seasonal_period=24).fit(y_train.values),
    "SARIMAX Baseline": ARIMAForecaster(order=(1, 0, 1), seasonal_order=None).fit(y_train.values),
    "Chronos Zero-Shot": ChronosZeroShotForecaster(model_size="chronos-bolt-small"),
}

# Pre-compute model forecasts and benchmark metrics
model_forecasts = {}
benchmark_rows = []

for model_name, model_obj in models_registry.items():
    if model_name == "Chronos Zero-Shot":
        preds = model_obj.forecast_zero_shot(y_train.values, horizon=24)
    elif "Naive" in model_name or "SARIMAX" in model_name:
        preds = model_obj.predict_quantiles(horizon=24)
    else:
        preds = model_obj.predict_quantiles(X_future)

    model_forecasts[model_name] = preds
    metrics = evaluate_forecast_metrics(y_actual, preds["p50"], preds.get("p10"), preds.get("p90"))

    benchmark_rows.append({
        "Model": model_name,
        "WAPE (%)": f"{metrics['WAPE']}%",
        "MAE (MW)": metrics["MAE"],
        "RMSE (MW)": metrics["RMSE"],
        "CRPS Score": metrics.get("CRPS", "N/A"),
    })

# Layout design system
CARD_STYLE = {
    "backgroundColor": "#1e293b",
    "borderRadius": "12px",
    "padding": "20px",
    "boxShadow": "0 10px 15px -3px rgba(0, 0, 0, 0.3)",
    "border": "1px solid #334155",
}

app.layout = html.Div(
    style={
        "backgroundColor": "#0f172a",
        "color": "#f8fafc",
        "fontFamily": "'Inter', -apple-system, sans-serif",
        "minHeight": "100vh",
        "padding": "24px",
    },
    children=[
        # Header Banner
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "marginBottom": "24px",
                "borderBottom": "1px solid #334155",
                "paddingBottom": "16px",
            },
            children=[
                html.Div([
                    html.H1("⚡ Renewable Energy Probabilistic Forecasting Lab", style={"margin": 0, "color": "#f59e0b", "fontSize": "26px", "fontWeight": "700"}),
                    html.P("Enterprise Grid Telemetry, Multi-Quantile Prediction Intervals & Walk-Forward Benchmark", style={"color": "#94a3b8", "marginTop": "6px", "fontSize": "14px"}),
                ]),
                html.Div(
                    style={"backgroundColor": "#064e3b", "color": "#34d399", "padding": "8px 16px", "borderRadius": "20px", "fontSize": "13px", "fontWeight": "600"},
                    children="● Grid Telemetry Healthy | 8,760 Hours Sync",
                ),
            ],
        ),

        # KPI Summary Cards
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))", "gap": "16px", "marginBottom": "24px"},
            children=[
                html.Div(style=CARD_STYLE, children=[
                    html.Div("Latest Actual Generation", style={"color": "#94a3b8", "fontSize": "12px", "textTransform": "uppercase"}),
                    html.Div(f"{y_actual[-1]:.1f} MW", style={"fontSize": "28px", "fontWeight": "700", "color": "#38bdf8", "marginTop": "4px"}),
                ]),
                html.Div(style=CARD_STYLE, children=[
                    html.Div("Top Model (XGBoost WAPE)", style={"color": "#94a3b8", "fontSize": "12px", "textTransform": "uppercase"}),
                    html.Div(f"{benchmark_rows[0]['WAPE (%)']}", style={"fontSize": "28px", "fontWeight": "700", "color": "#10b981", "marginTop": "4px"}),
                ]),
                html.Div(style=CARD_STYLE, children=[
                    html.Div("Top Model CRPS Score", style={"color": "#94a3b8", "fontSize": "12px", "textTransform": "uppercase"}),
                    html.Div(f"{benchmark_rows[0]['CRPS Score']}", style={"fontSize": "28px", "fontWeight": "700", "color": "#f59e0b", "marginTop": "4px"}),
                ]),
                html.Div(style=CARD_STYLE, children=[
                    html.Div("Forecast Horizon", style={"color": "#94a3b8", "fontSize": "12px", "textTransform": "uppercase"}),
                    html.Div("24 Hours", style={"fontSize": "28px", "fontWeight": "700", "color": "#c084fc", "marginTop": "4px"}),
                ]),
            ],
        ),

        # Main Controls & Visualization Grid
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 3fr", "gap": "20px"},
            children=[
                # Sidebar Controls
                html.Div(
                    style=CARD_STYLE,
                    children=[
                        html.H3("⚙️ Forecasting Controls", style={"marginTop": 0, "marginBottom": "16px", "fontSize": "16px", "color": "#f8fafc"}),
                        html.Label("Select Model Architecture", style={"fontSize": "13px", "color": "#94a3b8"}),
                        dcc.Dropdown(
                            id="model-selector",
                            options=[{"label": k, "value": k} for k in models_registry.keys()],
                            value="XGBoost Quantile",
                            clearable=False,
                            style={"backgroundColor": "#0f172a", "color": "#000", "marginTop": "6px", "marginBottom": "20px"},
                        ),
                        html.Label("Prediction Quantiles", style={"fontSize": "13px", "color": "#94a3b8"}),
                        dcc.Checklist(
                            id="quantile-checklist",
                            options=[
                                {"label": " P10 (Lower Bound)", "value": "p10"},
                                {"label": " P50 (Median Forecast)", "value": "p50"},
                                {"label": " P90 (Upper Bound)", "value": "p90"},
                            ],
                            value=["p10", "p50", "p90"],
                            style={"color": "#e2e8f0", "marginTop": "6px", "lineHeight": "28px"},
                        ),
                        html.Hr(style={"borderColor": "#334155", "margin": "20px 0"}),
                        html.H4("📊 Telemetry Metrics", style={"fontSize": "14px", "color": "#f8fafc"}),
                        html.Div(id="selected-metrics-container", style={"fontSize": "13px", "color": "#cbd5e1"}),
                    ],
                ),

                # Primary Chart & Benchmark Table
                html.Div(
                    children=[
                        # Fan Chart Container
                        html.Div(style=CARD_STYLE, children=[
                            dcc.Graph(id="probabilistic-fan-chart"),
                        ]),

                        # Benchmark Table Container
                        html.Div(style={**CARD_STYLE, "marginTop": "20px"}, children=[
                            html.H3("🏆 Model Benchmark & Quantitative Evaluation", style={"marginTop": 0, "marginBottom": "12px", "fontSize": "16px", "color": "#f8fafc"}),
                            dash_table.DataTable(
                                data=benchmark_rows,
                                columns=[{"name": col, "id": col} for col in benchmark_rows[0].keys()],
                                style_header={
                                    "backgroundColor": "#0f172a",
                                    "color": "#f59e0b",
                                    "fontWeight": "bold",
                                    "border": "1px solid #334155",
                                },
                                style_cell={
                                    "backgroundColor": "#1e293b",
                                    "color": "#f8fafc",
                                    "border": "1px solid #334155",
                                    "textAlign": "center",
                                    "padding": "10px",
                                },
                            ),
                        ]),
                    ]
                ),
            ],
        ),
    ],
)


@app.callback(
    [Output("probabilistic-fan-chart", "figure"), Output("selected-metrics-container", "children")],
    [Input("model-selector", "value"), Input("quantile-checklist", "value")],
)
def update_dashboard(selected_model, selected_quantiles):
    preds = model_forecasts[selected_model]
    t = list(range(1, 25))

    fig = go.Figure()

    # Ground Truth Actuals
    fig.add_trace(go.Scatter(
        x=t, y=y_actual, mode="lines+markers",
        name="Actual Telemetry",
        line=dict(color="#f8fafc", width=3, dash="dash"),
    ))

    # P90 and P10 uncertainty ribbon
    if "p90" in selected_quantiles and "p10" in selected_quantiles:
        fig.add_trace(go.Scatter(
            x=t, y=preds["p90"], mode="lines",
            line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=t, y=preds["p10"], mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(56, 189, 248, 0.2)",
            name="P10-P90 Uncertainty Interval",
        ))

    # P50 Median Forecast
    if "p50" in selected_quantiles:
        fig.add_trace(go.Scatter(
            x=t, y=preds["p50"], mode="lines+markers",
            name="P50 Median Forecast",
            line=dict(color="#38bdf8", width=3),
        ))

    fig.update_layout(
        title=f"⚡ 24-Hour Probabilistic Forecast — {selected_model}",
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        xaxis=dict(title="Forecast Horizon (Hours)", gridcolor="#334155"),
        yaxis=dict(title="Renewable Power (MW)", gridcolor="#334155"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    metrics = evaluate_forecast_metrics(y_actual, preds["p50"], preds.get("p10"), preds.get("p90"))
    metrics_html = [
        html.P(f"• WAPE: {metrics['WAPE']}%"),
        html.P(f"• MAE: {metrics['MAE']} MW"),
        html.P(f"• RMSE: {metrics['RMSE']} MW"),
        html.P(f"• CRPS: {metrics.get('CRPS', 'N/A')}"),
    ]

    return fig, metrics_html


if __name__ == "__main__":
    app.run_server(port=8050, debug=False)
