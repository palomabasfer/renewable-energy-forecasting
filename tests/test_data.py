import pandas as pd
from src.data.synthetic_grid_data import generate_renewable_timeseries
from src.data.validation import validate_time_series_data
from src.data.preprocessing import clean_telemetry_pipeline


def test_data_generation():
    df = generate_renewable_timeseries(n_hours=100)
    assert len(df) == 100
    assert 'solar_power_mw' in df.columns
    assert 'wind_power_mw' in df.columns
    assert 'total_renewable_mw' in df.columns


def test_data_validation():
    df = generate_renewable_timeseries(n_hours=100)
    report = validate_time_series_data(df)
    assert report.is_valid is True
    assert report.n_observations == 100
    assert report.duplicate_timestamps == 0
    assert report.missing_value_count == 0


def test_clean_telemetry_pipeline():
    df = generate_renewable_timeseries(n_hours=100)
    cleaned = clean_telemetry_pipeline(df)
    assert len(cleaned) == 100
    assert (cleaned['solar_power_mw'] >= 0).all()
    assert (cleaned['wind_power_mw'] >= 0).all()
