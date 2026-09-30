# Dynamic Pricing Engine — Production-Grade Real-Time MLOps System

A real-time dynamic pricing engine that predicts optimal prices based on demand, supply,
traffic, weather, time, and competitor pricing — the kind of system used by ride-sharing
and e-commerce platforms to automatically adjust prices based on live conditions.

> **Honest note on training data:** There is no real historical pricing dataset wired in
> yet. `pipeline/train_pipeline.py` generates synthetic data using a documented placeholder
> formula (`synthetic_price_formula()`) so the full pipeline runs end-to-end. The ML brain
> of the project — training, serving, drift detection, and retraining — is fully functional.
> The infrastructure layer (Kafka, Docker, Kubernetes) is correctly
> built and verified for structural correctness. See Known Limitations at the bottom for
> an honest breakdown of what has and hasn't been tested against live infrastructure.

---

## What This Is

Most pricing models are trained once and left to go stale. This system is designed to be
self-sustaining:

- Live data streams in through **Kafka** (orders, weather, traffic, competitor prices)
- **Feature engineering** transforms raw events into meaningful signals
- **XGBoost** predicts the optimal price in real time via a **FastAPI** `/predict` endpoint
- **MLflow** tracks every training run's metrics and parameters
- **Evidently** watches for incoming data drifting away from what the model was trained on
- When drift is detected, the system automatically retrains a challenger model, compares
  it fairly against the currently deployed one on the same held-out test split, and only
  promotes it if it's genuinely better (by at least 1% MAE improvement)
- **GitHub Actions** automates the full pipeline on every push: test → train → build → deploy
- **DVC** versions datasets and model artifacts for reproducibility

---

## Architecture

```
Customer Activity → Data Generator → Kafka → Feature Engineering
    → Training Pipeline → MLflow → FastAPI /predict
    → Docker → Kubernetes
    → Drift Detection → Auto-Retraining
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Serving | FastAPI, Uvicorn |
| ML Model | XGBoost |
| Experiment Tracking | MLflow (SQLite backend locally) |
| Streaming | Apache Kafka |
| Feature Engineering | Pandas, NumPy |
| Drift Detection | Evidently AI |
| Containerization | Docker |
| Orchestration | Kubernetes |
| CI/CD | GitHub Actions |
| Data Versioning | DVC |
| Testing | Pytest (60 tests) |

---

## Project Structure

```
dynamic-pricing-engine/
├── config.py                        # central settings: paths, Kafka, MLflow, thresholds
├── requirements.txt                 # all dependencies with version pins
├── .env                             # local secrets / overrides (not committed)
├── Dockerfile                       # containerizes the FastAPI API
├── docker-compose.yml               # Zookeeper + Kafka + API
├── pytest.ini                       # test discovery config
├── dvc.yaml / dvc.lock              # DVC pipeline stage + reproducibility lock
├── .github/workflows/ci-cd.yaml     # GitHub Actions: test → train → build → deploy
│
├── src/
│   ├── data_generator.py            # simulates orders, weather, traffic, competitor prices
│   ├── kafka_producer.py            # publishes simulated events to Kafka topics
│   ├── kafka_consumer.py            # buffers and assembles complete records from 4 topics
│   └── feature_engineering.py       # raw data → model features
│
├── pipeline/
│   ├── train_pipeline.py            # trains XGBoost, logs to MLflow, saves reference data
│   ├── retrain_pipeline.py          # drift check → retrain → compare → deploy if better
│   └── models/                      # trained model artifacts (generated, not committed)
│
├── app/
│   └── main.py                      # FastAPI: /predict, /health
│
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   └── secret.yaml
│
├── monitoring/
│   └── drift_detector.py            # Evidently-based drift detection
│
├── data/processed/                  # reference dataset for drift comparison (generated)
│
└── tests/                           # 60 tests across every module
    ├── test_data_generator.py        # 11 tests
    ├── test_feature_engineering.py   # 27 tests
    ├── test_kafka_consumer.py        # 6 tests
    ├── test_api.py                   # 7 tests
    ├── test_drift_detector.py        # 3 tests
    └── test_retrain_pipeline.py      # 6 tests
```

---

## Quickstart — Run The Core Pipeline

No Docker needed for this. The ML pipeline works standalone.

```powershell
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate          # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set PYTHONPATH (required — config.py is at root, src/ modules import from it)
# Windows PowerShell:
$env:PYTHONPATH = "$PWD;$PWD/src"
# Linux/macOS:
export PYTHONPATH="$(pwd):$(pwd)/src"

# 4. Train the model
# Generates synthetic data, trains XGBoost, saves:
#   pipeline/models/latest_model.json
#   data/processed/reference_data.csv
#   mlflow.db (MLflow experiment tracking via SQLite)
python pipeline/train_pipeline.py

# 5. Run the API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 6. Open Swagger UI in browser
# http://localhost:8000/docs
```

### Test a prediction

**JSON Payload (for Swagger UI / Postman):**
```json
{
  "order": {
    "order_id": "test-001",
    "city": "Mumbai",
    "timestamp": "2026-06-27T18:30:00+00:00",
    "demand_units": 40,
    "supply_units": 10,
    "base_price": 220.0
  },
  "weather": {
    "condition": "Rain",
    "temperature_c": 28.0
  },
  "traffic": {
    "traffic_level": "Severe",
    "avg_speed_kmph": 10.0
  },
  "competitor": {
    "competitor_price": 190.0
  }
}
```

**cURL (Linux/macOS/Git Bash):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "order": {"order_id": "test-001", "city": "Mumbai", "timestamp": "2026-06-27T18:30:00+00:00", "demand_units": 40, "supply_units": 10, "base_price": 220.0},
    "weather": {"condition": "Rain", "temperature_c": 28.0},
    "traffic": {"traffic_level": "Severe", "avg_speed_kmph": 10.0},
    "competitor": {"competitor_price": 190.0}
  }'
```

**PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body '{
  "order": {"order_id": "test-001", "city": "Mumbai", "timestamp": "2026-06-27T18:30:00+00:00", "demand_units": 40, "supply_units": 10, "base_price": 220.0},
  "weather": {"condition": "Rain", "temperature_c": 28.0},
  "traffic": {"traffic_level": "Severe", "avg_speed_kmph": 10.0},
  "competitor": {"competitor_price": 190.0}
}'
```

High demand + rain + severe traffic → predicted price will be well above the 220 base price.
Change `demand_units` to 5, `supply_units` to 40, `condition` to `"Clear"` and the price
drops back toward base — that's the dynamic pricing logic working.

---

## Running Tests

```powershell
pytest tests/ -v
```

60 tests across every module. All pass. Tests use real XGBoost training — no mocks for
the core ML logic — so the retraining suite takes ~20 seconds to run.

---

## Full Stack — Docker + Kafka

Requires Docker Desktop running.

```powershell
# Build the API image first (model must exist — run train_pipeline.py first)
docker build -t dynamic-pricing-engine:latest .

# Start everything: Zookeeper, Kafka, API
docker-compose up
```

| Service | URL | Notes |
|---|---|---|
| Pricing API | http://localhost:8000/docs | Swagger UI |

---

## Kafka Streaming

With `docker-compose up` running:

```powershell
# Publish simulated events (orders, weather, traffic, competitor prices)
python src/kafka_producer.py

# Consume, buffer by city, assemble complete records
python src/kafka_consumer.py
```

The consumer buffers incoming messages by city and only emits a complete record once all
four event types (order + weather + traffic + competitor) have arrived for that city.

---

## MLflow Experiment Tracking

MLflow logs to a local SQLite database (`mlflow.db`) — no separate server needed.
Set in `.env`:

```
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

To view the MLflow UI:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

Every training run logs: `n_estimators`, `max_depth`, `learning_rate`, `random_state`,
`test_size` as parameters, and `mae`, `rmse`, `r2` as metrics.

## Model and Monitoring Metrics

The training pipeline generates 5,000 synthetic examples and evaluates each model on a
held-out 20% test split. MLflow records these regression metrics for every run:

| Metric | Meaning | How to read it |
|---|---|---|
| MAE | Mean absolute difference between predicted and target prices | Lower is better; expressed in the same price units as the target |
| RMSE | Root mean squared prediction error | Lower is better; penalizes large errors more than MAE |
| R² | Share of target-price variation explained by the model | Closer to 1 is better; can be negative when predictions are worse than a mean baseline |

Retraining compares champion and challenger MAE on the same test split. A challenger is
promoted only when its MAE improves by at least 1%. Drift monitoring reports whether the
feature dataset drifted, how many of the seven input features drifted, and their share; a
current batch with fewer than 100 samples is skipped as too small for reliable detection.

These are pipeline metrics, not a claim of production pricing accuracy: the target labels
are generated by `synthetic_price_formula()`. Exact MAE, RMSE, and R² vary by training run
and are available in that run's MLflow record. The API currently exposes `/health` and
`/predict`; it does not yet provide a Prometheus `/metrics` endpoint.

---

## Drift Detection and Auto-Retraining

```powershell
# Standalone drift check — compares reference_data.csv against current data
python monitoring/drift_detector.py

# Full retraining pipeline — check drift, retrain, compare, deploy if better
python pipeline/retrain_pipeline.py
```

`DRIFT_THRESHOLD` in `config.py` controls what share of feature columns must individually
drift before the dataset is flagged. Defaults to `0.5` (Evidently's own default) — tuned
empirically to avoid false positives on this project's 7-feature set. At `0.1`, false
positive rate was 40% on identical distributions; at `0.5` it drops to zero.

Retraining only deploys a challenger model if it beats the current champion's MAE by at
least `MIN_IMPROVEMENT_THRESHOLD` (default 1%), evaluated on the same held-out test split
for a fair comparison. The previous model is backed up with a timestamp before replacement.

---

## Kubernetes Deployment

```powershell
# Start local cluster
minikube start

# Load image into minikube
minikube image load dynamic-pricing-engine:latest

# Apply all manifests
kubectl apply -f k8s/

# Check pods
kubectl get pods

# Access the API
kubectl port-forward service/pricing-api 8000:80
```

---

## Data Versioning (DVC)

See `DVC_SETUP.md` for full setup. Important: `pathspec==0.11.2` must stay pinned or
`dvc init` fails on the pinned DVC version (3.49.0).

```powershell
dvc repro    # run training pipeline stage, skips if nothing changed
dvc push     # upload tracked data/model to remote
dvc pull     # download tracked data/model on a fresh clone
```

---

## CI/CD

`.github/workflows/ci-cd.yaml` triggers on push/PR to `main`:

```
tests → train → build Docker image → deploy to Kubernetes
```

Build and deploy steps require registry and kubeconfig secrets configured in GitHub repo
settings. Test and train steps work without any secrets.

---

## Model Selection

XGBoost was selected as the primary model because it is the industry standard for tabular
regression problems. Pricing data is tabular (structured features like demand, supply,
weather) where tree-based models consistently outperform neural networks. The architecture
supports swapping in any model — the auto-retraining pipeline compares challenger vs
champion by MAE, so a LightGBM or RandomForest challenger that performs better would be
deployed automatically.

---

## What the Features Mean

| Feature | What It Represents |
|---|---|
| `demand_supply_ratio` | Demand divided by supply — high ratio means scarcity, price goes up |
| `peak_hour` | 1 if rush hour (8-9 AM, 5-7 PM), else 0 |
| `weather_factor` | Numeric severity: Clear=0.0, Rain=0.8, Storm=1.0 |
| `traffic_factor` | Numeric severity: Low=0.0, Severe=1.0 |
| `competitor_gap` | Your base price minus competitor price |
| `time_of_day` | Hour extracted from timestamp (0-23) |
| `base_price` | Starting price before dynamic adjustment |

---

## Known Limitations

- **Synthetic training data** — no real historical pricing dataset is wired in yet.
  `synthetic_price_formula()` in `train_pipeline.py` is a documented placeholder.

- **Drift detection uses synthetic reference data** — `reference_data.csv` is generated
  by the same synthetic formula as the training data, so drift detection currently compares
  synthetic vs synthetic. Adding prediction logging to a database would give drift detection
  real incoming data to compare against, making it genuinely meaningful.


- **Infrastructure not tested against live systems** — Kafka, Docker, Kubernetes were not available in the build environment. All are verified for
  structural correctness (valid configs, correct cross-references, tested import paths)
  but exercising them against live infrastructure is left to your environment.

- **DVC cloud remote untested** — only a local filesystem remote was verified. Cloud
  backends (S3/GCS/Azure) exercise the same DVC code path but cloud-specific auth and
  networking are untested.

- **MLflow model comparison** — only XGBoost is trained currently. MLflow experiment
  tracking becomes more useful with multi-model comparison (XGBoost vs LightGBM vs
  RandomForest) or hyperparameter tuning across multiple runs.