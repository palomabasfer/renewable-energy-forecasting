from src.analysis.eda import (
    compute_autocorrelations,
    compute_diurnal_profile,
    compute_time_series_summary,
    compute_weather_correlations,
)
from src.data.synthetic_grid_data import generate_renewable_timeseries


def test_eda_summary():
    df = generate_renewable_timeseries(n_hours=168)
    summary = compute_time_series_summary(df)
    assert "total_renewable_mw" in summary.columns
    assert "mean" in summary.index
    assert "std" in summary.index


def test_diurnal_profile():
    df = generate_renewable_timeseries(n_hours=168)
    diurnal = compute_diurnal_profile(df)
    assert len(diurnal) == 24
    assert "hour" in diurnal.columns
    assert "total_renewable_mw_mean" in diurnal.columns


def test_autocorrelations():
    df = generate_renewable_timeseries(n_hours=168)
    series = df["total_renewable_mw"].values
    acf_vals, pacf_vals = compute_autocorrelations(series, nlags=24)
    assert len(acf_vals) == 25
    assert len(pacf_vals) == 25


def test_weather_correlations():
    df = generate_renewable_timeseries(n_hours=168)
    corr = compute_weather_correlations(df)
    assert corr.shape[0] == corr.shape[1]
    assert "total_renewable_mw" in corr.columns
