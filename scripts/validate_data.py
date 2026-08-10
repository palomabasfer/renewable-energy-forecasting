#!/usr/bin/env python3
"""CLI script to validate renewable energy grid telemetry data."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_raw_telemetry, validate_time_series_data


def main():
    parser = argparse.ArgumentParser(description="Validate grid telemetry time series data.")
    parser.add_argument("--hours", type=int, default=8760, help="Number of synthetic hours if generating data.")
    args = parser.parse_args()

    print("🔍 Ingesting renewable telemetry data...")
    df = load_raw_telemetry(n_hours=args.hours)

    print("🛡️ Running temporal and physical validation checks...")
    report = validate_time_series_data(df)

    print("\n--- DATA VALIDATION REPORT ---")
    print(f"Is Valid: {report.is_valid}")
    print(f"Observations: {report.n_observations}")
    print(f"Start Timestamp: {report.start_timestamp}")
    print(f"End Timestamp: {report.end_timestamp}")
    print(f"Duplicate Timestamps: {report.duplicate_timestamps}")
    print(f"Missing Timestamps: {report.missing_timestamps}")
    print(f"Negative Power Values: {report.negative_power_values}")

    if report.issues:
        print("\n⚠️ Issues Detected:")
        for issue in report.issues:
            print(f" - {issue}")
    else:
        print("\n✅ All validation checks passed cleanly!")


if __name__ == "__main__":
    main()
