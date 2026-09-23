"""
config.py
Centralized configuration for the Dynamic Pricing Engine.
All environment-specific settings live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# Project Root (anchors all relative paths below, regardless of
# which directory a script is launched from)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FEATURE_STORE_DIR = os.path.join(BASE_DIR, "data", "features")

# ---------------------------
# Feature Engineering
# ---------------------------
FEATURE_COLUMNS = [
    "base_price",
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
MODEL_DIR = os.path.join(BASE_DIR, "pipeline", "models")
DEFAULT_MODEL_TYPE = os.getenv("DEFAULT_MODEL_TYPE", "xgboost")  # xgboost | random_forest | lightgbm
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------------------------
# MLflow Settings
# ---------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
# For local development without a running MLflow server, set in your .env:
#   MLFLOW_TRACKING_URI=sqlite:///mlflow.db
# (Plain folder-based tracking like "./mlruns" is deprecated in current MLflow
# and will raise an MlflowException unless MLFLOW_ALLOW_FILE_STORE=true is set.)
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
# This is used as Evidently's `drift_share`: the fraction of feature columns
# that must individually show statistical drift before the whole dataset is
# flagged. Tested empirically with this project's 7 feature columns:
#   - 0.1 (an earlier default here) caused a ~40% FALSE POSITIVE rate when
#     comparing two batches drawn from the identical distribution (random
#     per-column significance tests cross threshold by chance with few columns).
#   - 0.5 (Evidently's own library default) showed 0 false positives across
#     15 trials under the same test. Don't lower this without re-testing
#     false-positive rate first -- see monitoring/drift_detector.py.
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", 0.5))
REFERENCE_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "reference_data.csv")
PREDICTIONS_DB_PATH = os.path.join(BASE_DIR, "predictions.db")

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
LOG_DIR = os.path.join(BASE_DIR, "logs")