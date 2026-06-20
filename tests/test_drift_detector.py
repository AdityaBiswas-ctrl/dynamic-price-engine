"""
test_drift_detector.py
Tests for monitoring/drift_detector.py.

These tests generate their own reference and current datasets in-memory via
generate_synthetic_dataset(), so they don't depend on a pre-existing
reference_data.csv on disk (unlike running the module standalone, which
reads from REFERENCE_DATA_PATH).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "monitoring"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# pyrefly: ignore [missing-import]
import pytest
# pyrefly: ignore [missing-import]
from train_pipeline import generate_synthetic_dataset
# pyrefly: ignore [missing-import]
from drift_detector import run_drift_report


@pytest.fixture
def reference_data():
    return generate_synthetic_dataset(n_samples=1000)


class TestRunDriftReport:
    def test_identical_distribution_shows_no_drift(self, reference_data):
        # Current data drawn from the exact same generator/distribution as reference
        current = generate_synthetic_dataset(n_samples=300)
        result = run_drift_report(reference_data, current)
        assert result["dataset_drift"] is False

    def test_heavily_shifted_distribution_is_detected(self, reference_data):
        current = generate_synthetic_dataset(n_samples=300).copy()
        # Force a strong, consistent shift across most feature columns
        current["weather_factor"] = 1.0
        current["traffic_factor"] = 1.0
        current["demand_supply_ratio"] = current["demand_supply_ratio"] * 5
        current["peak_hour"] = 1

        result = run_drift_report(reference_data, current)
        assert result["dataset_drift"] is True
        assert result["number_of_drifted_columns"] >= 4

    def test_result_contains_expected_keys(self, reference_data):
        current = generate_synthetic_dataset(n_samples=300)
        result = run_drift_report(reference_data, current)
        expected_keys = {
            "dataset_drift", "number_of_columns", "number_of_drifted_columns",
            "share_of_drifted_columns", "drift_share_threshold",
        }
        assert expected_keys.issubset(result.keys())
        assert result["number_of_columns"] == 7  # len(FEATURE_COLUMNS)