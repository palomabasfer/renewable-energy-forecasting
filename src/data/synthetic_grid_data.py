import numpy as np
import pandas as pd

def generate_renewable_timeseries(n_hours: int = 720) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range(start='2026-01-01', periods=n_hours, freq='h')
    t = np.arange(n_hours)
    solar_base = np.maximum(0, np.sin(2 * np.pi * (t - 6) / 24)) * 100.0
    solar_noise = np.random.normal(0, 5, n_hours)
    solar_power = np.maximum(0, solar_base + solar_noise)

    wind_base = 50.0 + 30.0 * np.sin(2 * np.pi * t / 168) + 15.0 * np.cos(2 * np.pi * t / 24)
    wind_noise = np.random.normal(0, 8, n_hours)
    wind_power = np.maximum(0, wind_base + wind_noise)
    temp = 15.0 + 10.0 * np.sin(2 * np.pi * (t - 9) / 24) + np.random.normal(0, 2, n_hours)

    return pd.DataFrame({
        'timestamp': dates,
        'solar_power_mw': solar_power,
        'wind_power_mw': wind_power,
        'temperature_c': temp,
        'total_renewable_mw': solar_power + wind_power
    })
