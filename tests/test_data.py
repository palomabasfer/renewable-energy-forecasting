from src.data.synthetic_grid_data import generate_renewable_timeseries

def test_data_generation():
    df = generate_renewable_timeseries(n_hours=100)
    assert len(df) == 100
    assert 'solar_power_mw' in df.columns
    assert 'wind_power_mw' in df.columns
