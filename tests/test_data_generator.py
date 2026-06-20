"""
test_data_generator.py
Unit tests for data_generator.py.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# pyrefly: ignore [missing-import]
from data_generator import (
    generate_order_event,
    generate_weather_event,
    generate_traffic_event,
    generate_competitor_price_event,
    generate_batch_event,
    CITIES,
    WEATHER_CONDITIONS,
    TRAFFIC_LEVELS,
)


class TestGenerateOrderEvent:
    def test_has_required_fields(self):
        order = generate_order_event()
        required = {"order_id", "city", "timestamp", "demand_units", "supply_units", "base_price"}
        assert required.issubset(order.keys())

    def test_city_is_valid(self):
        order = generate_order_event()
        assert order["city"] in CITIES

    def test_demand_and_supply_in_valid_range(self):
        for _ in range(20):
            order = generate_order_event()
            assert 1 <= order["demand_units"] <= 50
            assert 1 <= order["supply_units"] <= 50

    def test_base_price_is_positive(self):
        order = generate_order_event()
        assert order["base_price"] > 0


class TestGenerateWeatherEvent:
    def test_condition_is_valid(self):
        weather = generate_weather_event("Mumbai")
        assert weather["condition"] in WEATHER_CONDITIONS

    def test_city_passthrough(self):
        weather = generate_weather_event("Pune")
        assert weather["city"] == "Pune"


class TestGenerateTrafficEvent:
    def test_traffic_level_is_valid(self):
        traffic = generate_traffic_event("Delhi")
        assert traffic["traffic_level"] in TRAFFIC_LEVELS

    def test_city_passthrough(self):
        traffic = generate_traffic_event("Bangalore")
        assert traffic["city"] == "Bangalore"


class TestGenerateCompetitorPriceEvent:
    def test_price_is_positive(self):
        competitor = generate_competitor_price_event("Hyderabad")
        assert competitor["competitor_price"] > 0


class TestGenerateBatchEvent:
    def test_batch_has_all_four_components(self):
        batch = generate_batch_event()
        assert set(batch.keys()) == {"order", "weather", "traffic", "competitor"}

    def test_all_components_share_same_city(self):
        batch = generate_batch_event()
        city = batch["order"]["city"]
        assert batch["weather"]["city"] == city
        assert batch["traffic"]["city"] == city
        assert batch["competitor"]["city"] == city