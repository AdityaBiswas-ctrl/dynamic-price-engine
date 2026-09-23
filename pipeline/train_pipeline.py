"""
train_pipeline.py
Generates synthetic training data using an engineered price formula,
trains an XGBoost model, logs experiments to MLflow, and saves the model.

NOTE: The synthetic price formula here is a placeholder. Once real historical
pricing data is available, replace `generate_synthetic_dataset()` with a
function that loads your actual labeled dataset (e.g. from data/processed/).
"""

import logging
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import xgboost as xgb
# pyrefly: ignore [missing-import]
import mlflow
# pyrefly: ignore [missing-import]
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os
import sys

# Ensure root and src directories are on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    MODEL_DIR,
    REFERENCE_DATA_PATH,
)
# pyrefly: ignore [missing-import]
from feature_engineering import (
    is_peak_hour,
    compute_demand_supply_ratio,
    compute_competitor_gap,
    compute_weather_factor,
    compute_traffic_factor,
    WEATHER_SEVERITY,
    TRAFFIC_SEVERITY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_pipeline")

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def synthetic_price_formula(base_price, demand_supply_ratio, competitor_gap,
                             weather_factor, traffic_factor, peak_hour):
    """
    Placeholder rule used ONLY to generate a synthetic target price for development.
    Real deployments must replace this with actual historical price labels.
    """
    price = base_price
    price += demand_supply_ratio * 15          # scarcity pushes price up
    price -= competitor_gap * 0.3              # if we're already pricier than competitor, ease up
    price += weather_factor * 25               # bad weather pushes price up
    price += traffic_factor * 20               # heavy traffic pushes price up
    price += peak_hour * 10                    # peak hour surcharge
    noise = np.random.normal(0, 5)              # small random noise for realism
    return round(max(price + noise, 0), 2)


def generate_synthetic_dataset(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generates a synthetic labeled dataset shaped like what feature_engineering.py
    produces, with a target price computed by synthetic_price_formula().
    """
    rows = []
    base_time = datetime.now(timezone.utc)

    for i in range(n_samples):
        demand = random.randint(1, 50)
        supply = random.randint(1, 50)
        base_price = round(random.uniform(50, 500), 2)
        competitor_price = round(random.uniform(50, 500), 2)
        weather_condition = random.choice(list(WEATHER_SEVERITY.keys()))
        traffic_level = random.choice(list(TRAFFIC_SEVERITY.keys()))
        timestamp = base_time - timedelta(minutes=random.randint(0, 100000))
        hour = timestamp.hour

        demand_supply_ratio = compute_demand_supply_ratio(demand, supply)
        competitor_gap = compute_competitor_gap(base_price, competitor_price)
        weather_factor = compute_weather_factor(weather_condition)
        traffic_factor = compute_traffic_factor(traffic_level)
        peak_hour = is_peak_hour(hour)

        target_price = synthetic_price_formula(
            base_price, demand_supply_ratio, competitor_gap,
            weather_factor, traffic_factor, peak_hour,
        )

        rows.append({
            "base_price": base_price,
            "time_of_day": hour,
            "peak_hour": peak_hour,
            "demand_supply_ratio": demand_supply_ratio,
            "competitor_gap": competitor_gap,
            "weather_factor": weather_factor,
            "traffic_factor": traffic_factor,
            TARGET_COLUMN: target_price,
        })

    df = pd.DataFrame(rows)
    logger.info("Generated synthetic dataset with %d rows", len(df))
    return df


def train_model(df: pd.DataFrame):
    """Trains an XGBoost regressor on the engineered features and logs to MLflow."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # NOTE: MLFLOW_TRACKING_URI here is the value imported from config at
    # module load time. If you need to point at a different tracking server
    # at runtime (e.g. in tests), set the MLFLOW_TRACKING_URI environment
    # variable BEFORE this module is imported, or monkeypatch this module's
    # MLFLOW_TRACKING_URI attribute directly -- calling mlflow.set_tracking_uri()
    # separately will get silently overridden by this line.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        params = {
            "objective": "reg:squarederror",
            "n_estimators": 150,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": RANDOM_STATE,
        }
        mlflow.log_params(params)

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.xgboost.log_model(model, artifact_path="model")

        logger.info("Run ID: %s", run.info.run_id)
        logger.info("MAE=%.3f RMSE=%.3f R2=%.3f", mae, rmse, r2)

        return model, {"mae": mae, "rmse": rmse, "r2": r2, "run_id": run.info.run_id}


def save_model_locally(model, path: str = None):
    """
    Saves the trained model to disk as a backup alongside the MLflow artifact.
    `path` defaults to f"{MODEL_DIR}/latest_model.json", resolved at CALL time
    (not import time) so that tests or other callers can monkeypatch the
    module-level MODEL_DIR and have it actually take effect.
    """
    import os
    if path is None:
        path = os.path.join(MODEL_DIR, "latest_model.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_model(path)
    logger.info("Model saved locally at %s", path)


def save_reference_data(df: pd.DataFrame, path: str = None):
    """
    Saves the training feature data as the 'reference' dataset for drift detection.
    monitoring/drift_detector.py compares future production data against this
    snapshot to detect when incoming data starts looking statistically different
    from what the currently deployed model was trained on.

    `path` defaults to REFERENCE_DATA_PATH, resolved at CALL time (not import
    time) so that tests or other callers can monkeypatch the module-level
    REFERENCE_DATA_PATH and have it actually take effect.
    """
    import os
    if path is None:
        path = REFERENCE_DATA_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df[FEATURE_COLUMNS].to_csv(path, index=False)
    logger.info("Reference data for drift detection saved at %s (%d rows)", path, len(df))


def run_training_pipeline():
    logger.info("Starting training pipeline...")
    df = generate_synthetic_dataset(n_samples=5000)
    model, metrics = train_model(df)
    save_model_locally(model)
    save_reference_data(df)
    logger.info("Training pipeline complete. Metrics: %s", metrics)
    return model, metrics


if __name__ == "__main__":
    run_training_pipeline()