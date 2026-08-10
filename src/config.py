"""Centralized configuration loader and path manager for renewable-energy-forecasting."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "configs"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"


def load_yaml_config(config_name: str) -> Dict[str, Any]:
    """Load a YAML configuration file from the configs directory."""
    if not config_name.endswith(".yaml") and not config_name.endswith(".yml"):
        config_name = f"{config_name}.yaml"

    config_path = CONFIG_DIR / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_data_config() -> Dict[str, Any]:
    """Get data configuration."""
    return load_yaml_config("data.yaml")


def get_features_config() -> Dict[str, Any]:
    """Get feature engineering configuration."""
    return load_yaml_config("features.yaml")


def get_forecasting_config() -> Dict[str, Any]:
    """Get forecasting configuration."""
    return load_yaml_config("forecasting.yaml")


def get_model_config() -> Dict[str, Any]:
    """Get model configuration."""
    return load_yaml_config("model_config.yaml")
