"""
kafka_consumer.py
Consumes order, weather, traffic, and competitor events from Kafka topics.
Buffers related events by city and forwards completed records for feature engineering.
"""

import json
import logging
from collections import defaultdict

from kafka import KafkaConsumer
from kafka.errors import KafkaError

from config import (
    KAFKA_BROKER_URL,
    KAFKA_TOPIC_ORDERS,
    KAFKA_TOPIC_WEATHER,
    KAFKA_TOPIC_TRAFFIC,
    KAFKA_TOPIC_COMPETITOR,
    KAFKA_CONSUMER_GROUP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("kafka_consumer")

TOPICS = [
    KAFKA_TOPIC_ORDERS,
    KAFKA_TOPIC_WEATHER,
    KAFKA_TOPIC_TRAFFIC,
    KAFKA_TOPIC_COMPETITOR,
]

# In-memory buffer: groups incoming events by city until all 4 pieces have arrived
city_buffer = defaultdict(dict)


def create_consumer():
    """Creates and returns a configured KafkaConsumer instance subscribed to all topics."""
    consumer = KafkaConsumer(
        *TOPICS,
        bootstrap_servers=KAFKA_BROKER_URL,
        group_id=KAFKA_CONSUMER_GROUP,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        auto_offset_reset="latest",   # only read new messages by default
        enable_auto_commit=True,
    )
    return consumer


def handle_message(message, on_complete_record=None):
    """
    Processes a single Kafka message, buffers it by city, and once all four
    event types (order, weather, traffic, competitor) have arrived for that
    city, emits a completed record.
    """
    topic = message.topic
    city = message.key
    payload = message.value

    if city is None:
        logger.warning("Received message with no city key on topic=%s, skipping", topic)
        return

    topic_to_field = {
        KAFKA_TOPIC_ORDERS: "order",
        KAFKA_TOPIC_WEATHER: "weather",
        KAFKA_TOPIC_TRAFFIC: "traffic",
        KAFKA_TOPIC_COMPETITOR: "competitor",
    }
    field_name = topic_to_field.get(topic)
    if field_name is None:
        logger.warning("Unrecognized topic=%s, skipping", topic)
        return

    city_buffer[city][field_name] = payload
    logger.info("Buffered %s data for city=%s", field_name, city)

    # Once all four pieces have arrived for this city, the record is complete
    required_fields = {"order", "weather", "traffic", "competitor"}
    if required_fields.issubset(city_buffer[city].keys()):
        completed_record = city_buffer.pop(city)
        logger.info("Completed record assembled for city=%s", city)
        if on_complete_record:
            on_complete_record(completed_record)
        else:
            logger.info("Completed record: %s", completed_record)


def run_consumer_loop(on_complete_record=None):
    """
    Continuously consumes messages from all subscribed topics.
    Pass `on_complete_record` (a function) to forward completed records
    to feature engineering instead of just logging them.
    """
    consumer = create_consumer()
    logger.info("Kafka consumer started. Subscribed to topics: %s", TOPICS)

    try:
        for message in consumer:
            handle_message(message, on_complete_record=on_complete_record)
    except KeyboardInterrupt:
        logger.info("Consumer stopped by user.")
    except KafkaError as e:
        logger.error("Kafka error occurred: %s", e)
    finally:
        consumer.close()
        logger.info("Consumer connection closed.")


if __name__ == "__main__":
    run_consumer_loop()