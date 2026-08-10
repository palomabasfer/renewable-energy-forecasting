"""Data loading utilities for grid telemetry datasets."""

from pathlib import Path
from typing import Optional
import pandas as pd

from src.config import DATA_DIR
from src.data.synthetic_grid_data import generate_renewable_timeseries


def load_raw_telemetry(
    file_path: Optional[Path] = None,
    generate_if_missing: bool = True,
    n_hours: int = 8760,
) -> pd.DataFrame:
    """Load raw telemetry data from CSV or generate synthetic data if missing.

    Parameters
    ----------
    file_path : Path, optional
        Path to CSV raw telemetry dataset. If None, defaults to DATA_DIR / 'raw' / 'grid_telemetry.csv'.
    generate_if_missing : bool, default=True
        Whether to auto-generate synthetic grid telemetry if the file does not exist.
    n_hours : int, default=8760
        Number of hours for synthetic generation if needed.

    Returns
    -------
    pd.DataFrame
        Raw telemetry DataFrame.
    """
    if file_path is None:
        file_path = DATA_DIR / "raw" / "grid_telemetry.csv"

    file_path = Path(file_path)

    if file_path.exists():
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        return df

    if generate_if_missing:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df = generate_renewable_timeseries(n_hours=n_hours)
        df.to_csv(file_path, index=False)
        return df

    raise FileNotFoundError(f"Telemetry file not found: {file_path}")
