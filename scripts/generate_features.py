#!/usr/bin/env python3
"""CLI script to run feature engineering pipeline."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR
from src.data import clean_telemetry_pipeline, load_raw_telemetry
from src.features import create_feature_pipeline


def main():
    parser = argparse.ArgumentParser(description="Generate temporal features for forecasting.")
    parser.add_argument("--output", type=str, default=str(DATA_DIR / "processed" / "featured_telemetry.csv"))
    args = parser.parse_args()

    print("📥 Ingesting raw telemetry...")
    raw_df = load_raw_telemetry(n_hours=8760)
    cleaned_df = clean_telemetry_pipeline(raw_df)

    print("⚙️ Constructing lags, rolling windows, and cyclical encodings...")
    feat_df = create_feature_pipeline(cleaned_df, drop_na=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(output_path, index=False)

    print(f"✅ Features generated successfully! Shape: {feat_df.shape}. Saved to {output_path}")


if __name__ == "__main__":
    main()
