"""
kafka_producer.py
Pushes simulated order, weather, traffic, and competitor events into Kafka topics.
Consumes data from data_generator.py and publishes it as a continuous stream.
"""

import json
import time
import logging

# pyrefly: ignore [missing-import]
from kafka import KafkaProducer
# pyrefly: ignore [missing-import]
from kafka.errors import KafkaError

from config import (
    KAFKA_BROKER_URL,
    KAFKA_TOPIC_ORDERS,
    KAFKA_TOPIC_WEATHER,
    KAFKA_TOPIC_TRAFFIC,
    KAFKA_TOPIC_COMPETITOR,
)
from data_generator import generate_batch_event

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("kafka_producer")


def create_producer():
    """Creates and returns a configured KafkaProducer instance."""
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER_URL,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        retries=5,
        acks="all",  # wait for full acknowledgment so events aren't silently dropped
    )
    return producer


def on_send_success(record_metadata):
    logger.info(
        "Delivered to topic=%s partition=%s offset=%s",
        record_metadata.topic,
        record_metadata.partition,
        record_metadata.offset,
    )


def on_send_error(excp):
    logger.error("Failed to deliver message", exc_info=excp)


def publish_batch(producer, batch):
    """Publishes one synchronized batch (order/weather/traffic/competitor) to their respective topics."""
    topic_map = {
        KAFKA_TOPIC_ORDERS: batch["order"],
        KAFKA_TOPIC_WEATHER: batch["weather"],
        KAFKA_TOPIC_TRAFFIC: batch["traffic"],
        KAFKA_TOPIC_COMPETITOR: batch["competitor"],
    }

    for topic, payload in topic_map.items():
        key = payload.get("city", "unknown")
        future = producer.send(topic, key=key, value=payload)
        future.add_callback(on_send_success)
        future.add_errback(on_send_error)


def run_producer_loop(interval_seconds: float = 1.0):
    """Continuously generates and publishes batch events until interrupted."""
    producer = create_producer()
    logger.info("Kafka producer started. Broker: %s", KAFKA_BROKER_URL)

    try:
        while True:
            batch = generate_batch_event()
            publish_batch(producer, batch)
            producer.flush()  # ensure messages are sent before sleeping
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        logger.info("Producer stopped by user.")
    except KafkaError as e:
        logger.error("Kafka error occurred: %s", e)
    finally:
        producer.close()
        logger.info("Producer connection closed.")


if __name__ == "__main__":
    run_producer_loop()