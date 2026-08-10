"""Time series data validation module for grid telemetry integrity."""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd


@dataclass
class ValidationReport:
    """Summary dataclass for time series validation results."""

    is_valid: bool
    n_observations: int
    start_timestamp: Optional[pd.Timestamp]
    end_timestamp: Optional[pd.Timestamp]
    frequency: str
    duplicate_timestamps: int
    missing_timestamps: int
    negative_power_values: int
    missing_value_count: int
    issues: List[str]


def validate_time_series_data(
    df: pd.DataFrame,
    time_column: str = "timestamp",
    power_columns: Optional[List[str]] = None,
    expected_freq: str = "1h",
) -> ValidationReport:
    """Perform strict temporal and physical sanity checks on renewable power telemetry data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing grid telemetry data.
    time_column : str, default='timestamp'
        Name of the timestamp column.
    power_columns : list of str, optional
        List of power generation columns to validate for non-negativity.
    expected_freq : str, default='1h'
        Expected pandas datetime frequency string.

    Returns
    -------
    ValidationReport
        Dataclass containing detailed validation diagnostic results.
    """
    if power_columns is None:
        power_columns = ["total_renewable_mw", "solar_power_mw", "wind_power_mw"]

    issues = []

    if time_column not in df.columns:
        issues.append(f"Missing time column: '{time_column}'")
        return ValidationReport(
            is_valid=False,
            n_observations=len(df),
            start_timestamp=None,
            end_timestamp=None,
            frequency=expected_freq,
            duplicate_timestamps=0,
            missing_timestamps=0,
            negative_power_values=0,
            missing_value_count=df.isna().sum().sum(),
            issues=issues,
        )

    # Convert to datetime if not already
    timestamps = pd.to_datetime(df[time_column])
    start_ts = timestamps.min()
    end_ts = timestamps.max()

    # Check for duplicates
    n_duplicates = int(timestamps.duplicated().sum())
    if n_duplicates > 0:
        issues.append(f"Found {n_duplicates} duplicate timestamps.")

    # Check chronological ordering
    if not timestamps.is_monotonic_increasing:
        issues.append("Timestamps are not strictly in chronological order.")

    # Check missing timestamps in regular grid
    expected_range = pd.date_range(start=start_ts, end=end_ts, freq=expected_freq, tz=timestamps.dt.tz)
    missing_ts_count = len(expected_range) - len(timestamps.unique())
    if missing_ts_count > 0:
        issues.append(f"Found {missing_ts_count} missing timestamps in expected range.")

    # Check non-negativity of power generation
    negative_counts = 0
    for col in power_columns:
        if col in df.columns:
            n_neg = int((df[col] < 0).sum())
            if n_neg > 0:
                negative_counts += n_neg
                issues.append(f"Column '{col}' has {n_neg} negative power generation values.")

    # Check overall missing values
    total_missing = int(df.isna().sum().sum())
    if total_missing > 0:
        issues.append(f"DataFrame contains {total_missing} missing (NaN) values.")

    is_valid = len(issues) == 0

    return ValidationReport(
        is_valid=is_valid,
        n_observations=len(df),
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        frequency=expected_freq,
        duplicate_timestamps=n_duplicates,
        missing_timestamps=max(0, missing_ts_count),
        negative_power_values=negative_counts,
        missing_value_count=total_missing,
        issues=issues,
    )
