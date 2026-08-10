"""Synthetic renewable energy grid telemetry generator."""

import numpy as np
import pandas as pd


def generate_renewable_timeseries(
    n_hours: int = 8760,
    start_date: str = "2025-01-01 00:00:00",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic hourly synthetic wind, solar, and temperature telemetry data.

    Parameters
    ----------
    n_hours : int, default=8760
        Number of hourly observations to generate (8760 hours = 1 year).
    start_date : str, default='2025-01-01 00:00:00'
        Start timestamp for the synthetic series.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, solar_power_mw, wind_power_mw,
        temperature_c, solar_irradiance_wm2, wind_speed_ms, total_renewable_mw.
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, periods=n_hours, freq="h", tz="UTC")
    t = np.arange(n_hours)

    # Solar power pattern: diurnal cycle + seasonal variation (higher in summer)
    day_of_year = dates.dayofyear.values
    solar_seasonal = 0.7 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    solar_diurnal = np.maximum(0, np.sin(2 * np.pi * (dates.hour.values - 6) / 24))
    solar_irradiance = solar_diurnal * solar_seasonal * 1000.0  # W/m2
    solar_noise = np.random.normal(0, 15, n_hours)
    solar_power = np.maximum(0, solar_irradiance * 0.15 + solar_noise * solar_diurnal)

    # Wind power pattern: synoptic scale variations (3-7 days) + diurnal breeze
    wind_synoptic = 30.0 * np.sin(2 * np.pi * t / (24 * 5))
    wind_diurnal = 15.0 * np.cos(2 * np.pi * (dates.hour.values - 15) / 24)
    wind_seasonal = 10.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    wind_speed = np.maximum(0, 8.0 + (wind_synoptic + wind_diurnal + wind_seasonal) / 5.0 + np.random.normal(0, 1.5, n_hours))
    # Wind power cubic relationship with speed (truncated at rated capacity ~200MW)
    wind_power = np.clip(0.5 * (wind_speed ** 3) * 0.1, 0, 200.0)

    # Temperature: annual cycle + diurnal cycle
    temp_annual = 15.0 - 10.0 * np.cos(2 * np.pi * (day_of_year - 20) / 365)
    temp_diurnal = 5.0 * np.sin(2 * np.pi * (dates.hour.values - 9) / 24)
    temperature = temp_annual + temp_diurnal + np.random.normal(0, 1.5, n_hours)

    # Total renewable generation
    total_renewable = solar_power + wind_power

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "solar_power_mw": np.round(solar_power, 2),
            "wind_power_mw": np.round(wind_power, 2),
            "temperature_c": np.round(temperature, 2),
            "solar_irradiance_wm2": np.round(solar_irradiance, 2),
            "wind_speed_ms": np.round(wind_speed, 2),
            "total_renewable_mw": np.round(total_renewable, 2),
        }
    )

    return df
