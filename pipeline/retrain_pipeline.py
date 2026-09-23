"""
retrain_pipeline.py
Auto-retraining orchestrator: checks for drift, and if detected (or forced),
trains a new "challenger" model, evaluates it against the currently deployed
"champion" model on the same held-out test data, and only replaces the
deployed model if the challenger genuinely improves on it by at least
MIN_IMPROVEMENT_THRESHOLD.

This implements the blueprint's flow:
  drift detected -> trigger retraining -> compare -> deploy better model
"""

import logging
import os
import shutil
from datetime import datetime, timezone

import numpy as np
# pyrefly: ignore [missing-import]
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

import sys

# Ensure root, src, and monitoring directories are on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "monitoring")))

from config import (
    MODEL_DIR,
    RETRAIN_TRIGGER_ON_DRIFT,
    MIN_IMPROVEMENT_THRESHOLD,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
)
from train_pipeline import generate_synthetic_dataset, train_model, save_model_locally, save_reference_data
# pyrefly: ignore [missing-import]
from drift_detector import check_drift, load_reference_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("retrain_pipeline")

CHAMPION_MODEL_PATH = os.path.join(MODEL_DIR, "latest_model.json")


def evaluate_model(model_path: str, X_test, y_test) -> float:
    """
    Loads an XGBoost model from disk and returns its MAE on the given test set.
    Used to evaluate the currently deployed "champion" model on the SAME
    test split as the newly trained "challenger", so the comparison is fair
    (different random splits would make MAE values not directly comparable).
    """
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    predictions = model.predict(X_test)
    return mean_absolute_error(y_test, predictions)


def backup_current_model(path: str = None) -> str:
    """
    Backs up the currently deployed model before it's potentially overwritten.
    `path` defaults to CHAMPION_MODEL_PATH, resolved at CALL time (not import
    time) so that monkeypatching CHAMPION_MODEL_PATH actually takes effect --
    this previously caused real backup files to leak into the production
    pipeline/models/ directory during isolated testing.
    """
    if path is None:
        path = CHAMPION_MODEL_PATH
    if not os.path.exists(path):
        return ""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = path.replace(".json", f"_backup_{timestamp}.json")
    shutil.copy2(path, backup_path)
    logger.info("Backed up current model to %s", backup_path)
    return backup_path


def run_retraining_pipeline(force: bool = False, current_data=None) -> dict:
    """
    Main orchestrator. Returns a dict summarizing what happened:
      {"retrained": bool, "deployed": bool, "reason": str, ...metrics if applicable}

    Args:
        force: skip the drift check and retrain unconditionally (e.g. for
               scheduled periodic retraining regardless of drift).
        current_data: DataFrame of recent production features to check for
               drift against. If None, a fresh synthetic batch is generated
               as a stand-in (since we don't have real production data yet --
               see train_pipeline.py's note on synthetic_price_formula).
    """
    # Step 1: Check drift (unless forced)
    if not force:
        try:
            drift_summary = check_drift(current_data)
        except FileNotFoundError as e:
            logger.warning("No reference data yet, cannot check drift: %s", e)
            return {"retrained": False, "deployed": False, "reason": "no_reference_data"}

        if not drift_summary["dataset_drift"]:
            logger.info("No drift detected. Skipping retraining.")
            return {"retrained": False, "deployed": False, "reason": "no_drift", **drift_summary}

        if not RETRAIN_TRIGGER_ON_DRIFT:
            logger.info("Drift detected but RETRAIN_TRIGGER_ON_DRIFT is False. Skipping retraining.")
            return {"retrained": False, "deployed": False, "reason": "retrain_disabled", **drift_summary}

        logger.info("Drift detected. Proceeding with retraining.")
    else:
        logger.info("Forced retraining requested (skipping drift check).")

    # Step 2: Train the challenger model on fresh data
    training_df = generate_synthetic_dataset(n_samples=5000)
    challenger_model, challenger_metrics = train_model(training_df)
    logger.info("Challenger trained: MAE=%.3f", challenger_metrics["mae"])

    # Step 3: Evaluate the current champion (if one exists) on the SAME test
    # split used to evaluate the challenger, for a fair comparison.
    from sklearn.model_selection import train_test_split
    X = training_df[FEATURE_COLUMNS]
    y = training_df[TARGET_COLUMN]
    _, X_test, _, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    if not os.path.exists(CHAMPION_MODEL_PATH):
        logger.info("No existing champion model found. Deploying challenger unconditionally.")
        save_model_locally(challenger_model)
        save_reference_data(training_df)
        return {
            "retrained": True, "deployed": True, "reason": "no_existing_champion",
            "challenger_mae": challenger_metrics["mae"],
        }

    champion_mae = evaluate_model(CHAMPION_MODEL_PATH, X_test, y_test)
    challenger_mae = challenger_metrics["mae"]

    # Improvement = relative reduction in MAE (lower MAE is better)
    improvement = (champion_mae - challenger_mae) / champion_mae if champion_mae > 0 else 0

    logger.info(
        "Champion MAE=%.3f | Challenger MAE=%.3f | Improvement=%.2f%% (threshold=%.2f%%)",
        champion_mae, challenger_mae, improvement * 100, MIN_IMPROVEMENT_THRESHOLD * 100,
    )

    # Step 4: Deploy only if the challenger genuinely improves enough
    if improvement >= MIN_IMPROVEMENT_THRESHOLD:
        backup_current_model()
        save_model_locally(challenger_model)
        save_reference_data(training_df)
        logger.info("Challenger deployed as new champion.")
        return {
            "retrained": True, "deployed": True, "reason": "challenger_improved",
            "champion_mae": champion_mae, "challenger_mae": challenger_mae,
            "improvement": improvement,
        }
    else:
        logger.info("Challenger did not improve enough. Keeping existing champion model.")
        return {
            "retrained": True, "deployed": False, "reason": "insufficient_improvement",
            "champion_mae": champion_mae, "challenger_mae": challenger_mae,
            "improvement": improvement,
        }


if __name__ == "__main__":
    result = run_retraining_pipeline()
    print(result)