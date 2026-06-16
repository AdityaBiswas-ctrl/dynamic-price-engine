"""
config.py
Centralized configuration for the Dynamic Pricing Engine.
All environment-specific settings live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# General
# ---------------------------
PROJECT_NAME = "dynamic-pricing-engine"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # development | staging | production

# ---------------------------
# Kafka Settings
# ---------------------------
KAFKA_BROKER_URL = os.getenv("KAFKA_BROKER_URL", "localhost:9092")
KAFKA_TOPIC_ORDERS = "orders_stream"
KAFKA_TOPIC_WEATHER = "weather_stream"
KAFKA_TOPIC_TRAFFIC = "traffic_stream"
KAFKA_TOPIC_COMPETITOR = "competitor_price_stream"
KAFKA_CONSUMER_GROUP = "pricing-engine-group"

# ---------------------------
# Data Paths
# ---------------------------
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
FEATURE_STORE_DIR = "data/features"

# ---------------------------
# Feature Engineering
# ---------------------------
FEATURE_COLUMNS = [
    "peak_hour",
    "demand_supply_ratio",
    "competitor_gap",
    "weather_factor",
    "traffic_factor",
    "time_of_day",
]
TARGET_COLUMN = "optimal_price"

# ---------------------------
# Model Training
# ---------------------------
MODEL_DIR = "pipeline/models"
DEFAULT_MODEL_TYPE = os.getenv("DEFAULT_MODEL_TYPE", "xgboost")  # xgboost | random_forest | lightgbm
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------------------------
# MLflow Settings
# ---------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "dynamic-pricing-experiments"
MLFLOW_MODEL_REGISTRY_NAME = "dynamic-pricing-model"

# ---------------------------
# DVC Settings
# ---------------------------
DVC_REMOTE_NAME = os.getenv("DVC_REMOTE_NAME", "storage")
DVC_REMOTE_URL = os.getenv("DVC_REMOTE_URL", "")

# ---------------------------
# FastAPI Settings
# ---------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_TITLE = "Dynamic Pricing Engine API"
API_VERSION = "1.0.0"

# ---------------------------
# Drift Detection
# ---------------------------
DRIFT_CHECK_INTERVAL_MINUTES = int(os.getenv("DRIFT_CHECK_INTERVAL_MINUTES", 30))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", 0.1))
REFERENCE_DATA_PATH = "data/processed/reference_data.csv"

# ---------------------------
# Retraining
# ---------------------------
RETRAIN_TRIGGER_ON_DRIFT = True
MIN_IMPROVEMENT_THRESHOLD = 0.01  # new model must beat old by at least 1%

# ---------------------------
# Monitoring
# ---------------------------
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", 9000))
GRAFANA_PORT = int(os.getenv("GRAFANA_PORT", 3000))

# ---------------------------
# Logging
# ---------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = "logs"