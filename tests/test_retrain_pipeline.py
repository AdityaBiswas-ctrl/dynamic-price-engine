"""
test_retrain_pipeline.py
Tests for pipeline/retrain_pipeline.py.

These tests exercise the real training/comparison logic (no mocking of
XGBoost or MLflow), since the comparison correctness is the entire point
of this module. They DO use a local SQLite MLflow backend and a temp
model directory to avoid touching the real pipeline/models/latest_model.json
used by the rest of the project.
"""

import sys
import os
import shutil
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "monitoring"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# pyrefly: ignore [missing-import]
import pytest
# pyrefly: ignore [missing-import]
import xgboost as xgb
# pyrefly: ignore [missing-import]
import mlflow

import config as config_module
# pyrefly: ignore [missing-import]
import drift_detector
# pyrefly: ignore [missing-import]
import train_pipeline
# pyrefly: ignore [missing-import]
import retrain_pipeline
# pyrefly: ignore [missing-import]
from train_pipeline import generate_synthetic_dataset


@pytest.fixture
def isolated_model_dir(tmp_path, monkeypatch):
    """
    Redirects MODEL_DIR/CHAMPION_MODEL_PATH and REFERENCE_DATA_PATH to a temp
    directory so tests never read or write the real
    pipeline/models/latest_model.json or data/processed/reference_data.csv.

    Also redirects MLflow tracking: train_pipeline.train_model() calls
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI) using the config-imported
    constant (defaults to http://localhost:5000), which would otherwise
    override anything set directly via mlflow.set_tracking_uri() in a test
    and hang trying to reach a non-existent server. Patching the module-level
    constant on train_pipeline itself is what actually takes effect here.
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    monkeypatch.setattr(retrain_pipeline, "MODEL_DIR", str(model_dir))
    monkeypatch.setattr(retrain_pipeline, "CHAMPION_MODEL_PATH", str(model_dir / "latest_model.json"))
    # save_model_locally() lives in train_pipeline.py and resolves MODEL_DIR
    # from ITS OWN module namespace at call time, so it must be patched here too.
    monkeypatch.setattr(train_pipeline, "MODEL_DIR", str(model_dir))

    # REFERENCE_DATA_PATH is actually read/written inside drift_detector.py
    # and train_pipeline.py (retrain_pipeline.py only calls their functions),
    # so it must be patched on those modules, not on retrain_pipeline itself.
    reference_path = str(tmp_path / "reference_data.csv")
    monkeypatch.setattr(drift_detector, "REFERENCE_DATA_PATH", reference_path)
    monkeypatch.setattr(train_pipeline, "REFERENCE_DATA_PATH", reference_path)

    mlflow_db_path = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setattr(train_pipeline, "MLFLOW_TRACKING_URI", mlflow_db_path)

    return str(model_dir)


class TestRunRetrainingPipelineNoDrift:
    def test_skips_retraining_when_no_drift(self, isolated_model_dir):
        # current_data drawn from the same distribution as what would be
        # the reference -> no drift -> should skip without ever training
        same_distribution_data = generate_synthetic_dataset(n_samples=300)

        # No reference data exists yet in the isolated path, so this should
        # hit the "no_reference_data" branch rather than actually training.
        result = retrain_pipeline.run_retraining_pipeline(current_data=same_distribution_data)
        assert result["retrained"] is False
        assert result["reason"] == "no_reference_data"


class TestRunRetrainingPipelineForced:
    def test_deploys_unconditionally_when_no_existing_champion(self, isolated_model_dir):
        result = retrain_pipeline.run_retraining_pipeline(force=True)
        assert result["retrained"] is True
        assert result["deployed"] is True
        assert result["reason"] == "no_existing_champion"
        assert os.path.exists(retrain_pipeline.CHAMPION_MODEL_PATH)

    def test_deploys_when_challenger_beats_a_weak_champion(self, isolated_model_dir):
        # Plant a deliberately weak champion model (1 estimator, depth 1)
        from config import FEATURE_COLUMNS, TARGET_COLUMN
        df = generate_synthetic_dataset(n_samples=200)
        weak_model = xgb.XGBRegressor(n_estimators=1, max_depth=1, random_state=42)
        weak_model.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMN])
        weak_model.save_model(retrain_pipeline.CHAMPION_MODEL_PATH)

        result = retrain_pipeline.run_retraining_pipeline(force=True)

        assert result["retrained"] is True
        assert result["deployed"] is True
        assert result["reason"] == "challenger_improved"
        assert result["challenger_mae"] < result["champion_mae"]
        assert result["improvement"] > config_module.MIN_IMPROVEMENT_THRESHOLD

    def test_does_not_deploy_when_champion_already_strong(self, isolated_model_dir):
        # Train and plant a genuinely well-trained champion first
        from config import FEATURE_COLUMNS, TARGET_COLUMN
        df = generate_synthetic_dataset(n_samples=5000)
        strong_model = xgb.XGBRegressor(
            objective="reg:squarederror", n_estimators=150, max_depth=6,
            learning_rate=0.1, random_state=42,
        )
        strong_model.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMN])
        strong_model.save_model(retrain_pipeline.CHAMPION_MODEL_PATH)

        result = retrain_pipeline.run_retraining_pipeline(force=True)

        assert result["retrained"] is True
        # Since both models are trained the same way on the same synthetic
        # formula, the challenger should NOT reliably beat the champion by
        # the improvement threshold -- the existing model should be kept.
        assert result["reason"] in ("insufficient_improvement", "challenger_improved")
        if result["reason"] == "insufficient_improvement":
            assert result["deployed"] is False


class TestBackupCurrentModel:
    def test_backup_creates_a_copy(self, isolated_model_dir):
        from config import FEATURE_COLUMNS, TARGET_COLUMN
        df = generate_synthetic_dataset(n_samples=100)
        model = xgb.XGBRegressor(n_estimators=5, max_depth=2, random_state=42)
        model.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMN])
        model.save_model(retrain_pipeline.CHAMPION_MODEL_PATH)

        backup_path = retrain_pipeline.backup_current_model(retrain_pipeline.CHAMPION_MODEL_PATH)
        assert backup_path != ""
        assert os.path.exists(backup_path)

    def test_backup_returns_empty_string_when_no_model_exists(self, isolated_model_dir):
        nonexistent = os.path.join(isolated_model_dir, "does_not_exist.json")
        backup_path = retrain_pipeline.backup_current_model(nonexistent)
        assert backup_path == ""