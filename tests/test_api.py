"""
test_api.py
Tests for the FastAPI /predict and /health endpoints in app/main.py.
Uses FastAPI's TestClient as a context manager so startup events
(model loading) actually fire -- see development notes: using TestClient
without the `with` block skips lifespan/startup events and model_loaded
will incorrectly read False.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# pyrefly: ignore [missing-import]
import pytest
from fastapi.testclient import TestClient
# pyrefly: ignore [missing-import]
from main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


VALID_PAYLOAD = {
    "order": {
        "order_id": "test-1",
        "city": "Mumbai",
        "timestamp": "2026-06-18T18:30:00+00:00",
        "demand_units": 40,
        "supply_units": 10,
        "base_price": 220.0,
    },
    "weather": {"condition": "Rain", "temperature_c": 24.0},
    "traffic": {"traffic_level": "Severe", "avg_speed_kmph": 8.0},
    "competitor": {"competitor_price": 190.0},
}


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_reports_model_loaded(self, client):
        # Assumes a trained model exists at pipeline/models/latest_model.json
        # (run train_pipeline.py before running this test suite).
        response = client.get("/health")
        assert response.json()["status"] == "ok"


class TestPredictEndpoint:
    def test_valid_request_returns_200(self, client):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_response_contains_expected_fields(self, client):
        response = client.post("/predict", json=VALID_PAYLOAD)
        body = response.json()
        assert "city" in body
        assert "predicted_price" in body
        assert "features_used" in body
        assert body["city"] == "Mumbai"

    def test_predicted_price_is_positive(self, client):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.json()["predicted_price"] > 0

    def test_missing_required_field_returns_422(self, client):
        bad_payload = {
            "order": {
                # city is intentionally missing
                "order_id": "x",
                "timestamp": "2026-06-18T10:00:00+00:00",
                "demand_units": 5,
                "supply_units": 5,
                "base_price": 100.0,
            },
            "weather": {"condition": "Clear"},
            "traffic": {"traffic_level": "Low"},
            "competitor": {"competitor_price": 100.0},
        }
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_negative_base_price_returns_422(self, client):
        bad_payload = dict(VALID_PAYLOAD)
        bad_payload["order"] = dict(VALID_PAYLOAD["order"])
        bad_payload["order"]["base_price"] = -50.0
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422