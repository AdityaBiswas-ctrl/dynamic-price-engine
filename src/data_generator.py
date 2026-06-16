"""
data_generator.py
Simulates real-time data: customer orders, weather, traffic, and competitor pricing.
This acts as the source feeding the Kafka producer.
"""

import random
import time
import uuid
from datetime import datetime, timezone

# ---------------------------
# Reference data pools
# ---------------------------
CITIES = ["Mumbai", "Delhi", "Bangalore", "Pune", "Hyderabad"]
WEATHER_CONDITIONS = ["Clear", "Rain", "Cloudy", "Storm", "Fog"]
TRAFFIC_LEVELS = ["Low", "Moderate", "High", "Severe"]


def generate_order_event():
    """Simulates a single customer order/ride request event."""
    return {
        "order_id": str(uuid.uuid4()),
        "city": random.choice(CITIES),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "demand_units": random.randint(1, 50),     # number of active requests
        "supply_units": random.randint(1, 50),      # number of available drivers/items
        "base_price": round(random.uniform(50, 500), 2),
    }


def generate_weather_event(city):
    """Simulates current weather conditions for a city."""
    return {
        "city": city,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "condition": random.choice(WEATHER_CONDITIONS),
        "temperature_c": round(random.uniform(15, 40), 1),
    }


def generate_traffic_event(city):
    """Simulates current traffic conditions for a city."""
    return {
        "city": city,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "traffic_level": random.choice(TRAFFIC_LEVELS),
        "avg_speed_kmph": round(random.uniform(5, 60), 1),
    }


def generate_competitor_price_event(city):
    """Simulates a competitor's current price for the same service/city."""
    return {
        "city": city,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "competitor_price": round(random.uniform(50, 500), 2),
    }


def generate_batch_event():
    """
    Generates one synchronized batch of events (order + weather + traffic + competitor)
    for the same city and timestamp — useful for feeding all topics together.
    """
    order = generate_order_event()
    city = order["city"]

    return {
        "order": order,
        "weather": generate_weather_event(city),
        "traffic": generate_traffic_event(city),
        "competitor": generate_competitor_price_event(city),
    }


if __name__ == "__main__":
    # Simple standalone test: print simulated events every second
    print("Starting data generator... Press Ctrl+C to stop.\n")
    try:
        while True:
            batch = generate_batch_event()
            print(batch)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nData generator stopped.")