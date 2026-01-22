"""
Data drift detection for fish classification API.

This module uses Evidently AI to detect drift in image features and prediction
distributions over time. Since we're working with image data, we extract structured
features (brightness, contrast, etc.) that can be analyzed with standard drift
detection methods.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from evidently.metric_preset import (  # type: ignore[import-untyped]
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.report import Report  # type: ignore[import-untyped]
from evidently.test_preset import DataDriftTestPreset, DataQualityTestPreset  # type: ignore[import-untyped]
from evidently.test_suite import TestSuite  # type: ignore[import-untyped]


def load_reference_data(data_dir: str = "data/processed") -> pd.DataFrame:
    """
    Load reference data (training set features) for drift comparison.

    This function should extract features from the training images.
    For now, we'll create a placeholder that can be replaced with
    actual training data features.

    Args:
        data_dir: Path to processed data directory

    Returns:
        DataFrame with feature columns
    """
    # TODO: Extract features from training set images
    # For now, return empty DataFrame with correct columns
    columns = [
        "mean_brightness",
        "std_brightness",
        "contrast",
        "sharpness",
        "red_mean",
        "green_mean",
        "blue_mean",
        "red_std",
        "green_std",
        "blue_std",
        "aspect_ratio",
        "approx_size_kb",
        "prediction",
    ]
    return pd.DataFrame(columns=columns)


def load_current_data(database_path: str = "prediction_database.csv", n_latest: int | None = None) -> pd.DataFrame:
    """
    Load current production data from prediction database.

    Args:
        database_path: Path to prediction database CSV
        n_latest: If set, only load the latest N entries

    Returns:
        DataFrame with feature columns and predictions
    """
    if not Path(database_path).exists():
        raise FileNotFoundError(f"Prediction database not found: {database_path}")

    df = pd.read_csv(database_path)

    # Drop timestamp column (not used in drift detection)
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    if n_latest is not None:
        df = df.tail(n_latest)

    return df


def generate_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: str = "data_drift_report.html",
) -> Report:
    """
    Generate Evidently drift report with data drift, data quality, and target drift analysis.

    Args:
        reference_data: Training/reference dataset
        current_data: Production/current dataset
        output_path: Path to save HTML report

    Returns:
        Evidently Report object
    """
    report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            TargetDriftPreset(columns=["prediction"]),
        ]
    )

    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html(output_path)

    return report


def run_drift_tests(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> TestSuite:
    """
    Run Evidently test suite for programmatic drift detection.

    This returns pass/fail results that can be used in CI/CD or alerts.

    Args:
        reference_data: Training/reference dataset
        current_data: Production/current dataset

    Returns:
        Evidently TestSuite with results
    """
    test_suite = TestSuite(
        tests=[
            DataDriftTestPreset(),
            DataQualityTestPreset(),
        ]
    )

    test_suite.run(reference_data=reference_data, current_data=current_data)

    return test_suite


def filter_by_last_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Filter dataframe to last N entries.

    Args:
        df: Input dataframe
        n: Number of entries to keep

    Returns:
        Filtered dataframe
    """
    return df.tail(n)


def filter_by_hours(df: pd.DataFrame, hours: int, timestamp_column: str = "timestamp") -> pd.DataFrame:
    """
    Filter dataframe to entries from last N hours.

    Args:
        df: Input dataframe with timestamp column
        hours: Number of hours to look back
        timestamp_column: Name of timestamp column

    Returns:
        Filtered dataframe
    """
    if timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found")

    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    cutoff = pd.Timestamp.now("UTC") - pd.Timedelta(hours=hours)
    return df[df[timestamp_column] >= cutoff]


def main():
    """Run data drift detection on prediction database."""
    # Load data
    print("Loading reference and current data...")
    reference_data = load_reference_data()
    current_data = load_current_data()

    if reference_data.empty:
        print("Warning: Reference data is empty. Please extract features from training data first.")
        return

    if current_data.empty:
        print("Warning: No predictions logged yet. Make some API calls first.")
        return

    print(f"Reference data: {len(reference_data)} samples")
    print(f"Current data: {len(current_data)} samples")

    # Generate report
    print("\nGenerating drift report...")
    generate_drift_report(reference_data, current_data)
    print("Report saved to: data_drift_report.html")

    # Run tests
    print("\nRunning drift tests...")
    test_suite = run_drift_tests(reference_data, current_data)

    # Print test results
    results = test_suite.as_dict()
    print("\nTest Results:")
    for test in results.get("tests", []):
        status = "✅ PASS" if test.get("status") == "SUCCESS" else "❌ FAIL"
        print(f"{status}: {test.get('name', 'Unknown test')}")


if __name__ == "__main__":
    main()
