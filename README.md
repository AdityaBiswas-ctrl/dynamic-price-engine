# Dynamic Pricing Engine — Real-Time MLOps System

A real-time dynamic pricing engine that predicts optimal prices based on
demand, supply, traffic, weather, time, and competitor pricing — built as a
full MLOps pipeline with streaming ingestion, automated training, model
serving, monitoring, drift detection, auto-retraining, and data versioning.

## What this is

Most pricing models are trained once and left to go stale. This system is
designed to be self-sustaining: live data streams in continuously, a FastAPI
service serves predictions in real time, Prometheus/Grafana watch system
health, Evidently watches for the incoming data drifting away from what the
model was trained on, and when it does, the system automatically retrains a
challenger model, compares it fairly against the currently deployed one, and
only promotes it if it's genuinely better.

> **Note on training data:** there's no real historical pricing dataset yet.
> `pipeline/train_pipeline.py` generates synthetic data using a documented
> placeholder formula (see `synthetic_price_formula()`) so the full pipeline
> is runnable end-to-end. Swap this for real historical data when available —
> everything downstream (training, serving, drift detection, retraining)
> works the same way regardless of where the data comes from.

## Architecture

```
Customer Activity → Data Generator → Kafka → Feature Engineering
    → Training Pipeline → MLflow → Model Registry → FastAPI
    → Docker → Kubernetes → Monitoring → Drift Detection → Auto-Retraining
```

## Project structure

```
dynamic-pricing-engine/
├── config.py                  # central settings: paths, Kafka, MLflow, drift threshold, etc.
├── requirements.txt
├── .env                       # local secrets / overrides (not committed)
├── Dockerfile
├── pytest.ini
├── dvc.yaml / dvc.lock        # DVC pipeline stage + reproducibility lock
├── .github/workflows/ci-cd.yaml
├── src/
│   ├── data_generator.py      # simulates orders, weather, traffic, competitor prices
│   ├── kafka_producer.py      # publishes simulated events to Kafka topics
│   ├── kafka_consumer.py      # buffers and assembles complete records from 4 topics
│   └── feature_engineering.py # raw data → model features
├── pipeline/
│   ├── train_pipeline.py      # trains XGBoost, logs to MLflow, saves reference data
│   ├── retrain_pipeline.py    # drift check → retrain → compare → deploy
│   └── models/                # trained model artifacts (generated, not committed)
├── app/
│   └── main.py                # FastAPI: /predict, /health, /metrics
├── k8s/                       # Deployment, Service, ConfigMap, Secret
├── monitoring/
│   ├── drift_detector.py      # Evidently-based drift detection
│   ├── prometheus.yml
│   ├── docker-compose.monitoring.yml
│   └── grafana/                # provisioned datasource + dashboard
├── data/processed/             # reference dataset for drift comparison (generated)
└── tests/                      # 60 tests covering every module above
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in environment variables
cp .env.example .env   # or use the .env already provided as a starting point

# 3. Train the model (generates synthetic data, trains XGBoost, saves
#    pipeline/models/latest_model.json and data/processed/reference_data.csv)
python pipeline/train_pipeline.py

# 4. Run the API
uvicorn app.main:app --host 0.0.0.0 --port 8000
# or: python app/main.py

# 5. Try it
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "order": {"order_id": "1", "city": "Mumbai", "timestamp": "2026-06-20T18:30:00+00:00",
               "demand_units": 40, "supply_units": 10, "base_price": 220.0},
    "weather": {"condition": "Rain", "temperature_c": 24.0},
    "traffic": {"traffic_level": "Severe", "avg_speed_kmph": 8.0},
    "competitor": {"competitor_price": 190.0}
  }'
```

**Note:** if you run scripts directly (not via `pytest`, which handles this
automatically), make sure the project root and `src/` are on your Python
path, since `config.py` lives at the root and most modules import from it:

```bash
export PYTHONPATH="$(pwd):$(pwd)/src:$(pwd)/app"
```

## Running tests

```bash
pytest tests/ -v
```

60 tests covering data generation, Kafka buffering logic, feature
engineering, the FastAPI endpoints, drift detection, and the full
auto-retraining decision logic (using real XGBoost training, not mocks).

## Streaming (Kafka)

Requires a running Kafka broker (set `KAFKA_BROKER_URL` in `.env`).

```bash
python src/kafka_producer.py   # publishes simulated events
python src/kafka_consumer.py   # consumes, buffers, and assembles complete records
```

## Monitoring (Prometheus + Grafana)

```bash
docker compose -f monitoring/docker-compose.monitoring.yml up
```
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin) — dashboard auto-provisioned

The API exposes `/metrics` with request latency, request counts, prediction
volume by city, and prediction error counts. CPU/memory come from cAdvisor
via Kubernetes, not from the app itself — see `monitoring/prometheus.yml`.

## Drift detection & auto-retraining

```bash
python monitoring/drift_detector.py     # standalone drift check against reference data
python pipeline/retrain_pipeline.py     # check drift, retrain + deploy if it improves
```

`DRIFT_THRESHOLD` (in `config.py`) controls how large a share of feature
columns must individually drift before the whole dataset is flagged —
defaults to `0.5`, tuned empirically to avoid false positives on this
project's 7-feature set (see comments in `config.py` and
`monitoring/drift_detector.py` for the reasoning).

Retraining only deploys a new model if it beats the current one's MAE by at
least `MIN_IMPROVEMENT_THRESHOLD` (default 1%), evaluated on the same
held-out test split for a fair comparison. The previous model is backed up
before being replaced.

## Containers & deployment

```bash
docker build -t dynamic-pricing-engine:latest .
kubectl apply -f k8s/
```

## Data versioning (DVC)

See `DVC_SETUP.md` for full setup, including a required dependency pin
(`pathspec==0.11.2`) without which `dvc init` fails on the pinned DVC
version. Quick reference:

```bash
dvc repro    # run the training pipeline stage, skips if nothing changed
dvc push     # upload tracked data/model to remote storage
dvc pull     # download tracked data/model (e.g. on a fresh clone)
```

## CI/CD

`.github/workflows/ci-cd.yaml` runs on push/PR to `main`: tests → train →
build Docker image → deploy to Kubernetes. Build and deploy steps need
registry and kubeconfig secrets configured in your GitHub repo settings
before they'll do anything beyond test and train.

## Known limitations

- **Synthetic training data**: no real historical pricing dataset is wired
  in yet (see note at the top).
- **No live Kafka/Kubernetes/Docker daemon was available** while building
  this, so the streaming, container, and orchestration pieces are verified
  for correctness (config structure, cross-references, logic) but not
  exercised against live infrastructure — test these in your actual
  environment before depending on them in production.
- **Cloud DVC remote** (S3/GCS/Azure) is untested — only a local filesystem
  remote was verified, which exercises the same DVC code path but not
  cloud-specific auth/networking.