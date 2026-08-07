import dash
from dash import dcc, html
import plotly.graph_objects as go
import numpy as np

app = dash.Dash(__name__)

t = np.arange(24)
p50 = 120.0 + 25.0 * np.sin(2 * np.pi * t / 24)
p10 = p50 - 8.0
p90 = p50 + 8.0

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=p90, mode='lines', line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=t, y=p10, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(56, 189, 248, 0.2)', name='P10-P90 Uncertainty Interval'))
fig.add_trace(go.Scatter(x=t, y=p50, mode='lines+markers', line=dict(color='#38bdf8', width=3), name='P50 Median Forecast'))

fig.update_layout(
    title='⚡ 24-Hour Probabilistic Grid Power Forecast (TFT vs Chronos)',
    paper_bgcolor='#0f172a',
    plot_bgcolor='#0f172a',
    font=dict(color='#e2e8f0'),
    xaxis_title='Forecast Horizon (Hours)',
    yaxis_title='Generation Power (MW)'
)

app.layout = html.Div(style={'backgroundColor': '#0f172a', 'color': '#e2e8f0', 'padding': '20px'}, children=[
    html.H1("⚡ Renewable Energy Probabilistic Forecasting Lab", style={'color': '#f59e0b'}),
    html.P("Comparing ARIMA, XGBoost, Temporal Fusion Transformer (TFT), PatchTST & Chronos/Bolt Zero-Shot"),
    dcc.Graph(id='probabilistic-forecast-graph', figure=fig)
])

if __name__ == '__main__':
    app.run_server(port=8050, debug=False)
