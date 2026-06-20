"""
test_feature_engineering.py
Unit tests for feature_engineering.py.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# pyrefly: ignore [missing-import]
import pytest
# pyrefly: ignore [missing-import]
from feature_engineering import (
    extract_hour_from_timestamp,
    is_peak_hour,
    compute_demand_supply_ratio,
    compute_competitor_gap,
    compute_weather_factor,
    compute_traffic_factor,
    engineer_features,
)


class TestExtractHourFromTimestamp:
    def test_valid_timestamp(self):
        assert extract_hour_from_timestamp("2026-06-17T18:30:00+00:00") == 18

    def test_midnight(self):
        assert extract_hour_from_timestamp("2026-06-17T00:00:00+00:00") == 0

    def test_invalid_timestamp_defaults_to_noon(self):
        assert extract_hour_from_timestamp("not-a-timestamp") == 12

    def test_empty_string_defaults_to_noon(self):
        assert extract_hour_from_timestamp("") == 12


class TestIsPeakHour:
    @pytest.mark.parametrize("hour", [8, 9, 17, 18, 19])
    def test_peak_hours_return_one(self, hour):
        assert is_peak_hour(hour) == 1

    @pytest.mark.parametrize("hour", [0, 3, 12, 14, 23])
    def test_non_peak_hours_return_zero(self, hour):
        assert is_peak_hour(hour) == 0


class TestComputeDemandSupplyRatio:
    def test_basic_ratio(self):
        assert compute_demand_supply_ratio(40, 10) == 4.0

    def test_equal_demand_supply(self):
        assert compute_demand_supply_ratio(20, 20) == 1.0

    def test_zero_supply_does_not_divide_by_zero(self):
        # supply is clamped to at least 1, so this must not raise
        result = compute_demand_supply_ratio(10, 0)
        assert result == 10.0

    def test_zero_demand(self):
        assert compute_demand_supply_ratio(0, 10) == 0.0


class TestComputeCompetitorGap:
    def test_we_are_more_expensive(self):
        assert compute_competitor_gap(220.0, 190.0) == 30.0

    def test_we_are_cheaper(self):
        assert compute_competitor_gap(150.0, 200.0) == -50.0

    def test_equal_prices(self):
        assert compute_competitor_gap(100.0, 100.0) == 0.0


class TestComputeWeatherFactor:
    def test_known_conditions(self):
        assert compute_weather_factor("Clear") == 0.0
        assert compute_weather_factor("Storm") == 1.0
        assert compute_weather_factor("Rain") == 0.8

    def test_unknown_condition_defaults_to_zero(self):
        assert compute_weather_factor("Tornado") == 0.0


class TestComputeTrafficFactor:
    def test_known_levels(self):
        assert compute_traffic_factor("Low") == 0.0
        assert compute_traffic_factor("Severe") == 1.0

    def test_unknown_level_defaults_to_zero(self):
        assert compute_traffic_factor("Gridlock") == 0.0


class TestEngineerFeatures:
    def test_complete_record_produces_all_expected_keys(self):
        record = {
            "order": {
                "order_id": "test-123",
                "city": "Pune",
                "timestamp": "2026-06-17T18:30:00+00:00",
                "demand_units": 40,
                "supply_units": 10,
                "base_price": 220.0,
            },
            "weather": {"condition": "Rain", "temperature_c": 24.0},
            "traffic": {"traffic_level": "Severe", "avg_speed_kmph": 8.0},
            "competitor": {"competitor_price": 190.0},
        }
        features = engineer_features(record)

        expected_keys = {
            "city", "timestamp", "base_price", "time_of_day", "peak_hour",
            "demand_supply_ratio", "competitor_gap", "weather_factor", "traffic_factor",
        }
        assert expected_keys.issubset(features.keys())
        assert features["city"] == "Pune"
        assert features["peak_hour"] == 1
        assert features["demand_supply_ratio"] == 4.0
        assert features["competitor_gap"] == 30.0
        assert features["weather_factor"] == 0.8
        assert features["traffic_factor"] == 1.0

    def test_missing_fields_use_safe_defaults(self):
        # An incomplete record should not raise — missing pieces fall back to defaults
        record = {"order": {"city": "Mumbai", "timestamp": "2026-06-17T12:00:00+00:00"}}
        features = engineer_features(record)
        assert features["city"] == "Mumbai"
        assert features["demand_supply_ratio"] == 0.0  # demand defaults to 0