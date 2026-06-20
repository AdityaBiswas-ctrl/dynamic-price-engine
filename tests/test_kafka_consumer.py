"""
test_kafka_consumer.py
Unit tests for the kafka_consumer.py buffering and record assembly logic.
Does NOT require a live Kafka broker — tests handle_message() directly
with fake message objects, the same way it was manually verified during development.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# pyrefly: ignore [missing-import]
import pytest
# pyrefly: ignore [missing-import]
from kafka_consumer import handle_message, city_buffer


class FakeMessage:
    """Mimics the attributes kafka-python's ConsumerRecord exposes that handle_message() reads."""
    def __init__(self, topic, key, value):
        self.topic = topic
        self.key = key
        self.value = value


@pytest.fixture(autouse=True)
def clear_buffer():
    """Ensure city_buffer starts empty for every test, since it's a module-level dict."""
    city_buffer.clear()
    yield
    city_buffer.clear()


class TestHandleMessage:
    def test_partial_record_does_not_emit(self):
        results = []
        handle_message(
            FakeMessage("orders_stream", "Pune", {"order_id": 1}),
            on_complete_record=results.append,
        )
        handle_message(
            FakeMessage("weather_stream", "Pune", {"condition": "Rain"}),
            on_complete_record=results.append,
        )
        assert len(results) == 0
        assert "order" in city_buffer["Pune"]
        assert "weather" in city_buffer["Pune"]

    def test_complete_record_emits_exactly_once(self):
        results = []
        handle_message(FakeMessage("orders_stream", "Pune", {"order_id": 1}), on_complete_record=results.append)
        handle_message(FakeMessage("weather_stream", "Pune", {"condition": "Rain"}), on_complete_record=results.append)
        handle_message(FakeMessage("traffic_stream", "Pune", {"traffic_level": "High"}), on_complete_record=results.append)
        handle_message(FakeMessage("competitor_price_stream", "Pune", {"competitor_price": 200}), on_complete_record=results.append)

        assert len(results) == 1
        assembled = results[0]
        assert assembled["order"] == {"order_id": 1}
        assert assembled["weather"] == {"condition": "Rain"}
        assert assembled["traffic"] == {"traffic_level": "High"}
        assert assembled["competitor"] == {"competitor_price": 200}

    def test_buffer_is_cleared_after_completion(self):
        results = []
        for topic, payload in [
            ("orders_stream", {"order_id": 1}),
            ("weather_stream", {"condition": "Rain"}),
            ("traffic_stream", {"traffic_level": "High"}),
            ("competitor_price_stream", {"competitor_price": 200}),
        ]:
            handle_message(FakeMessage(topic, "Pune", payload), on_complete_record=results.append)

        assert "Pune" not in city_buffer

    def test_different_cities_buffered_independently(self):
        results = []
        handle_message(FakeMessage("orders_stream", "Mumbai", {"order_id": 1}), on_complete_record=results.append)
        handle_message(FakeMessage("orders_stream", "Delhi", {"order_id": 2}), on_complete_record=results.append)

        assert "order" in city_buffer["Mumbai"]
        assert "order" in city_buffer["Delhi"]
        assert len(results) == 0

    def test_message_with_no_city_key_is_skipped(self):
        results = []
        handle_message(FakeMessage("orders_stream", None, {"order_id": 1}), on_complete_record=results.append)
        assert len(results) == 0
        assert len(city_buffer) == 0

    def test_unrecognized_topic_is_skipped(self):
        results = []
        handle_message(FakeMessage("unknown_topic", "Pune", {"x": 1}), on_complete_record=results.append)
        assert len(results) == 0
        assert "Pune" not in city_buffer