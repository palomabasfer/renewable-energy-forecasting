"""Time series feature engineering module with strict temporal boundary enforcement."""

from typing import List, Optional

import numpy as np
import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    target_columns: List[str],
    lags: List[int],
) -> pd.DataFrame:
    """Create lag features for specified target columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_columns : list of str
        Target column names to create lags for.
    lags : list of int
        Lag steps (e.g., [1, 2, 24, 168]).

    Returns
    -------
    pd.DataFrame
        DataFrame with lag columns added.
    """
    data = df.copy()
    for col in target_columns:
        if col in data.columns:
            for lag in lags:
                data[f"{col}_lag_{lag}"] = data[col].shift(lag)
    return data


def add_rolling_features(
    df: pd.DataFrame,
    target_columns: List[str],
    windows: List[int],
    metrics: List[str] = ("mean", "std"),
) -> pd.DataFrame:
    """Create rolling window summary features with 1-step shift to prevent target leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_columns : list of str
        Target columns to compute rolling statistics for.
    windows : list of int
        Rolling window sizes in hours (e.g., [6, 24, 168]).
    metrics : list of str, default=('mean', 'std')
        Rolling summary metrics to compute.

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling features added.
    """
    data = df.copy()
    for col in target_columns:
        if col in data.columns:
            # Shift by 1 step so observation at time t ONLY uses data up to t-1
            shifted_series = data[col].shift(1)
            for w in windows:
                roll = shifted_series.rolling(window=w, min_periods=max(1, w // 2))
                if "mean" in metrics:
                    data[f"{col}_roll_{w}_mean"] = roll.mean()
                if "std" in metrics:
                    data[f"{col}_roll_{w}_std"] = roll.std().fillna(0.0)
                if "min" in metrics:
                    data[f"{col}_roll_{w}_min"] = roll.min()
                if "max" in metrics:
                    data[f"{col}_roll_{w}_max"] = roll.max()
    return data


def add_calendar_cyclical_features(
    df: pd.DataFrame,
    time_column: str = "timestamp",
) -> pd.DataFrame:
    """Extract calendar features and encode cyclical periodicities using sine/cosine.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing timestamp column.
    time_column : str, default='timestamp'
        Name of timestamp column.

    Returns
    -------
    pd.DataFrame
        DataFrame with calendar and cyclical features.
    """
    data = df.copy()
    timestamps = pd.to_datetime(data[time_column])

    # Calendar integer attributes
    hour = timestamps.dt.hour.values
    dayofweek = timestamps.dt.dayofweek.values
    month = timestamps.dt.month.values
    is_weekend = (dayofweek >= 5).astype(float)

    data["hour"] = hour
    data["dayofweek"] = dayofweek
    data["month"] = month
    data["is_weekend"] = is_weekend

    # Cyclical sin/cos encodings
    data["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    data["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)

    data["sin_month"] = np.sin(2 * np.pi * month / 12.0)
    data["cos_month"] = np.cos(2 * np.pi * month / 12.0)

    data["sin_dayofweek"] = np.sin(2 * np.pi * dayofweek / 7.0)
    data["cos_dayofweek"] = np.cos(2 * np.pi * dayofweek / 7.0)

    return data


def create_feature_pipeline(
    df: pd.DataFrame,
    time_column: str = "timestamp",
    target_column: str = "total_renewable_mw",
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Execute complete feature engineering pipeline on telemetry DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input cleaned telemetry DataFrame.
    time_column : str
        Timestamp column name.
    target_column : str
        Primary forecasting target variable.
    lags : list of int, optional
        Lags to build. Defaults to [1, 2, 3, 6, 12, 24, 48, 168].
    windows : list of int, optional
        Rolling windows to build. Defaults to [6, 24, 168].
    drop_na : bool, default=True
        Whether to drop rows containing NaN values from lagging.

    Returns
    -------
    pd.DataFrame
        Feature-augmented DataFrame.
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 48, 168]
    if windows is None:
        windows = [6, 24, 168]

    target_cols = [target_column]
    if "solar_power_mw" in df.columns:
        target_cols.append("solar_power_mw")
    if "wind_power_mw" in df.columns:
        target_cols.append("wind_power_mw")

    data = add_calendar_cyclical_features(df, time_column=time_column)
    data = add_lag_features(data, target_columns=target_cols, lags=lags)
    data = add_rolling_features(data, target_columns=[target_column], windows=windows)

    if drop_na:
        data = data.dropna().reset_index(drop=True)

    return data
