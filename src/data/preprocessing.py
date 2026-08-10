"""Data preprocessing and temporal cleaning pipeline."""

from typing import List, Optional

import pandas as pd


def clean_telemetry_pipeline(
    df: pd.DataFrame,
    time_column: str = "timestamp",
    power_columns: Optional[List[str]] = None,
    clip_negative: bool = True,
    expected_freq: str = "1h",
) -> pd.DataFrame:
    """Preprocess and clean raw renewable telemetry data into a regular time series.

    Parameters
    ----------
    df : pd.DataFrame
        Raw telemetry DataFrame.
    time_column : str, default='timestamp'
        Name of timestamp column.
    power_columns : list of str, optional
        Power generation columns to enforce physical boundaries on.
    clip_negative : bool, default=True
        Whether to clip negative generation values to zero.
    expected_freq : str, default='1h'
        Target sampling frequency.

    Returns
    -------
    pd.DataFrame
        Cleaned, chronologically sorted DataFrame with datetime index.
    """
    if power_columns is None:
        power_columns = ["total_renewable_mw", "solar_power_mw", "wind_power_mw"]

    data = df.copy()

    # Parse and sort timestamps
    data[time_column] = pd.to_datetime(data[time_column])
    data = data.drop_duplicates(subset=[time_column]).sort_values(time_column)

    # Set datetime index
    data = data.set_index(time_column)

    # Resample to regular hourly frequency and linearly interpolate small gaps
    data = data.resample(expected_freq).mean()
    data = data.interpolate(method="time")

    # Clip physical generation boundaries
    if clip_negative:
        for col in power_columns:
            if col in data.columns:
                data[col] = data[col].clip(lower=0.0)

    # Reset index to retain timestamp column
    cleaned_df = data.reset_index()

    return cleaned_df
