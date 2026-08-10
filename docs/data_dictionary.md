# 📖 Renewable Energy Telemetry Data Dictionary

| Variable Name | Type | Description | Unit | Frequency | Expected Range | Known at Forecast Time | Leakage Risk |
| ------------- | ---- | ----------- | ---- | --------- | -------------- | ---------------------- | ------------ |
| `timestamp` | Datetime (UTC) | Telemetry observation timestamp | ISO 8601 | Hourly (`1h`) | 2025-01-01 to Present | Yes | None |
| `total_renewable_mw` | Float64 | Total combined renewable power output | Megawatts (MW) | Hourly (`1h`) | $[0.0, 500.0]$ | No (Target Variable) | High if shifted $\le 0$ |
| `solar_power_mw` | Float64 | Solar photovoltaic grid power generation | Megawatts (MW) | Hourly (`1h`) | $[0.0, 250.0]$ | No (Sub-target) | High if unshifted |
| `wind_power_mw` | Float64 | Wind turbine farm power generation | Megawatts (MW) | Hourly (`1h`) | $[0.0, 250.0]$ | No (Sub-target) | High if unshifted |
| `temperature_c` | Float64 | Ambient air temperature at grid node | Degrees Celsius (°C) | Hourly (`1h`) | $[-15.0, 45.0]$ | Yes (NWP Forecast) | Low |
| `solar_irradiance_wm2` | Float64 | Global Horizontal Irradiance (GHI) | $W/m^2$ | Hourly (`1h`) | $[0.0, 1200.0]$ | Yes (NWP Forecast) | Low |
| `wind_speed_ms` | Float64 | Wind speed at 100m turbine hub height | $m/s$ | Hourly (`1h`) | $[0.0, 35.0]$ | Yes (NWP Forecast) | Low |
| `sin_hour` / `cos_hour` | Float64 | Cyclical sine/cosine hour of day encoding | Continuous $[-1, 1]$ | Hourly (`1h`) | $[-1.0, 1.0]$ | Yes (Deterministic) | None |
| `total_renewable_mw_lag_24` | Float64 | Diurnal 24-hour historical generation lag | Megawatts (MW) | Hourly (`1h`) | $[0.0, 500.0]$ | Yes ($t-24$) | None |
| `total_renewable_mw_roll_24_mean` | Float64 | 24-hour historical rolling mean generation | Megawatts (MW) | Hourly (`1h`) | $[0.0, 500.0]$ | Yes (Shifted 1 step) | None |
