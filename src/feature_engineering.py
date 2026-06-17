"""
feature_engineering.py
Transforms raw order, weather, traffic, and competitor data into engineered features
suitable for model training and inference.

Features produced: peak_hour, demand_supply_ratio, competitor_gap,
weather_factor, traffic_factor, time_of_day.
"""

import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("feature_engineering")

# Mapping raw categorical conditions to numeric severity scores.
# Higher score = condition more likely to push price up.
WEATHER_SEVERITY = {
    "Clear": 0.0,
    "Cloudy": 0.2,
    "Fog": 0.5,
    "Rain": 0.8,
    "Storm": 1.0,
}

TRAFFIC_SEVERITY = {
    "Low": 0.0,
    "Moderate": 0.4,
    "High": 0.7,
    "Severe": 1.0,
}


def extract_hour_from_timestamp(timestamp_str: str) -> int:
    """Extracts hour (0-23) from an ISO 8601 timestamp string. Defaults to noon on failure."""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.hour
    except (ValueError, TypeError) as e:
        logger.warning("Failed to parse timestamp '%s': %s", timestamp_str, e)
        return 12


def is_peak_hour(hour: int) -> int:
    """Returns 1 if the hour falls in typical peak demand windows (8-10am, 5-8pm), else 0."""
    return 1 if hour in (8, 9, 17, 18, 19) else 0


def compute_demand_supply_ratio(demand: float, supply: float) -> float:
    """Ratio of demand to supply. Higher ratio = scarcity = upward price pressure."""
    safe_supply = max(supply, 1)  # avoid division by zero
    return round(demand / safe_supply, 3)


def compute_competitor_gap(base_price: float, competitor_price: float) -> float:
    """
    Difference between our base price and competitor's price.
    Positive = we are more expensive than competitor.
    Negative = we are cheaper than competitor.
    """
    return round(base_price - competitor_price, 2)


def compute_weather_factor(condition: str) -> float:
    """Maps a weather condition string to a numeric severity score (0.0 - 1.0)."""
    return WEATHER_SEVERITY.get(condition, 0.0)


def compute_traffic_factor(traffic_level: str) -> float:
    """Maps a traffic level string to a numeric severity score (0.0 - 1.0)."""
    return TRAFFIC_SEVERITY.get(traffic_level, 0.0)


def engineer_features(record: dict) -> dict:
    """
    Transforms one completed record (order + weather + traffic + competitor)
    into a flat dictionary of model-ready features.

    Args:
        record: dict with keys 'order', 'weather', 'traffic', 'competitor',
                 as assembled by kafka_consumer.handle_message().

    Returns:
        dict: engineered feature values, plus passthrough identifiers
              (city, timestamp, base_price) useful for logging/joining later.
    """
    order = record.get("order", {})
    weather = record.get("weather", {})
    traffic = record.get("traffic", {})
    competitor = record.get("competitor", {})

    # Raw values with safe defaults in case a field is missing
    demand = order.get("demand_units", 0)
    supply = order.get("supply_units", 1)
    base_price = order.get("base_price", 100)
    timestamp = order.get("timestamp", "")
    city = order.get("city", "unknown")

    competitor_price = competitor.get("competitor_price", base_price)
    weather_condition = weather.get("condition", "Clear")
    traffic_level = traffic.get("traffic_level", "Low")

    hour = extract_hour_from_timestamp(timestamp)

    features = {
        "city": city,
        "timestamp": timestamp,
        "base_price": base_price,
        "time_of_day": hour,
        "peak_hour": is_peak_hour(hour),
        "demand_supply_ratio": compute_demand_supply_ratio(demand, supply),
        "competitor_gap": compute_competitor_gap(base_price, competitor_price),
        "weather_factor": compute_weather_factor(weather_condition),
        "traffic_factor": compute_traffic_factor(traffic_level),
    }

    logger.info("Engineered features for city=%s: %s", city, features)
    return features


if __name__ == "__main__":
    # Quick standalone test with a fake completed record
    sample_record = {
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
    result = engineer_features(sample_record)
    print(result)