"""Exploratory Time Series Analysis and Data Profiling Module."""

from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf


def compute_time_series_summary(
    df: pd.DataFrame,
    target_columns: Tuple[str, ...] = ("total_renewable_mw", "solar_power_mw", "wind_power_mw"),
) -> pd.DataFrame:
    """Compute statistical summary for target generation columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_columns : tuple of str
        Columns to analyze.

    Returns
    -------
    pd.DataFrame
        DataFrame with rows as metrics and columns as targets.
    """
    stats = {}
    for col in target_columns:
        if col in df.columns:
            s = df[col].dropna()
            stats[col] = {
                "count": len(s),
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "p25": s.quantile(0.25),
                "median": s.median(),
                "p75": s.quantile(0.75),
                "max": s.max(),
                "skewness": s.skew(),
                "kurtosis": s.kurtosis(),
                "zero_pct": (s == 0).mean() * 100.0,
            }
    return pd.DataFrame(stats).round(2)


def compute_diurnal_profile(
    df: pd.DataFrame,
    time_column: str = "timestamp",
    target_columns: Tuple[str, ...] = ("solar_power_mw", "wind_power_mw", "total_renewable_mw"),
) -> pd.DataFrame:
    """Compute mean and standard deviation of generation by hour of the day.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    time_column : str
        Timestamp column name.
    target_columns : tuple of str
        Target columns.

    Returns
    -------
    pd.DataFrame
        Hourly profiles grouped by hour (0..23).
    """
    data = df.copy()
    data[time_column] = pd.to_datetime(data[time_column])
    data["hour"] = data[time_column].dt.hour

    agg_dict = {}
    for col in target_columns:
        if col in data.columns:
            agg_dict[f"{col}_mean"] = (col, "mean")
            agg_dict[f"{col}_std"] = (col, "std")

    return data.groupby("hour").agg(**agg_dict).reset_index().round(2)


def compute_autocorrelations(
    series: np.ndarray, nlags: int = 48
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ACF and PACF values for a time series.

    Parameters
    ----------
    series : np.ndarray
        1D target time series array.
    nlags : int, default=48
        Number of lags to calculate.

    Returns
    -------
    acf_vals : np.ndarray
    pacf_vals : np.ndarray
    """
    clean_series = series[~np.isnan(series)]
    acf_vals = acf(clean_series, nlags=nlags, fft=True)
    pacf_vals = pacf(clean_series, nlags=nlags, method="yw")
    return acf_vals, pacf_vals


def compute_weather_correlations(
    df: pd.DataFrame,
    feature_cols: Tuple[str, ...] = (
        "solar_power_mw",
        "wind_power_mw",
        "total_renewable_mw",
        "temperature_c",
        "solar_irradiance_wm2",
        "wind_speed_ms",
    ),
) -> pd.DataFrame:
    """Compute Pearson correlation matrix between generation and weather variables."""
    available_cols = [c for c in feature_cols if c in df.columns]
    return df[available_cols].corr().round(3)
