"""
drift_detector.py
Detects feature distribution drift between the data the current model was
trained on (reference) and recent production data (current), using Evidently.

Reference data is written by pipeline/train_pipeline.py to REFERENCE_DATA_PATH
every time the model is (re)trained. Current data is whatever recent feature
records you pass in -- e.g. a buffer of features computed by
feature_engineering.engineer_features() in production, or a CSV export of
recent predictions.
"""

import logging
import os

import pandas as pd
# pyrefly: ignore [missing-import]
from evidently.report import Report
# pyrefly: ignore [missing-import]
from evidently.metric_preset import DataDriftPreset

from config import FEATURE_COLUMNS, REFERENCE_DATA_PATH, DRIFT_THRESHOLD

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drift_detector")


def load_reference_data() -> pd.DataFrame:
    """
    Loads the reference dataset (the data the currently deployed model
    was trained on) from REFERENCE_DATA_PATH.
    """
    if not os.path.exists(REFERENCE_DATA_PATH):
        raise FileNotFoundError(
            f"No reference data found at {REFERENCE_DATA_PATH}. "
            "Run the training pipeline first -- it writes this file automatically."
        )
    return pd.read_csv(REFERENCE_DATA_PATH)


def run_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    """
    Runs an Evidently DataDriftPreset report comparing reference vs current
    feature distributions, restricted to FEATURE_COLUMNS (the actual model inputs).

    NOTE on DRIFT_THRESHOLD: Evidently's `drift_share` is the fraction of
    columns that must individually show statistical drift before the overall
    dataset is flagged as drifted. With this project's 7 feature columns,
    a low threshold (e.g. 0.1) was empirically tested and caused a ~40% false
    positive rate when comparing two batches from the IDENTICAL distribution --
    per-column significance tests cross threshold by chance with so few columns.
    DRIFT_THRESHOLD now defaults to 0.5 (Evidently's own library default),
    which showed 0 false positives across 15 trials under the same test.
    Don't lower this in .env without re-running that false-positive check.
    """
    reference = reference_df[FEATURE_COLUMNS]
    current = current_df[FEATURE_COLUMNS]

    report = Report(metrics=[DataDriftPreset(drift_share=DRIFT_THRESHOLD)])
    report.run(reference_data=reference, current_data=current)
    result = report.as_dict()["metrics"][0]["result"]

    summary = {
        "dataset_drift": result["dataset_drift"],
        "number_of_columns": result["number_of_columns"],
        "number_of_drifted_columns": result["number_of_drifted_columns"],
        "share_of_drifted_columns": result["share_of_drifted_columns"],
        "drift_share_threshold": result["drift_share"],
    }
    return summary


def check_drift(current_df: pd.DataFrame) -> dict:
    """
    Main entry point: loads the reference dataset and checks the given
    current data for drift against it. Returns a summary dict; callers
    (e.g. an auto-retraining job) should check summary["dataset_drift"].
    """
    reference_df = load_reference_data()

    if len(current_df) < 30:
        logger.warning(
            "Only %d current samples provided; drift statistics are unreliable "
            "below ~30 samples. Consider buffering more data before checking.",
            len(current_df),
        )

    summary = run_drift_report(reference_df, current_df)

    if summary["dataset_drift"]:
        logger.warning(
            "DRIFT DETECTED: %d/%d columns drifted (%.1f%%, threshold=%.1f%%)",
            summary["number_of_drifted_columns"],
            summary["number_of_columns"],
            summary["share_of_drifted_columns"] * 100,
            summary["drift_share_threshold"] * 100,
        )
    else:
        logger.info(
            "No significant drift: %d/%d columns drifted (%.1f%%, threshold=%.1f%%)",
            summary["number_of_drifted_columns"],
            summary["number_of_columns"],
            summary["share_of_drifted_columns"] * 100,
            summary["drift_share_threshold"] * 100,
        )

    return summary


if __name__ == "__main__":
    # Standalone smoke test: compare the reference data against a freshly
    # generated synthetic batch (acting as a stand-in for "current production data").
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
    # pyrefly: ignore [missing-import]
    from train_pipeline import generate_synthetic_dataset

    current = generate_synthetic_dataset(n_samples=300)
    result = check_drift(current)
    print(result)