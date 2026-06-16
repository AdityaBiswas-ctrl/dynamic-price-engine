# Dynamic Pricing Engine

A production-ready system for dynamic price optimization. This project leverages real-time data streaming, feature stores, machine learning, and continuous monitoring to adjust prices dynamically based on market signals, user sentiment, and system health.

## Project Structure

This repository is structured as follows:

*   **`data/`**: Stores raw datasets, processed features, and local data files. Versioned using DVC (Data Version Control) to ensure reproducibility.
*   **`src/`**: Core business logic, including:
    *   Data producers and consumers (Kafka).
    *   Feature engineering scripts.
    *   Shared utilities and helper functions.
*   **`app/`**: FastAPI application serving the price prediction API endpoints.
*   **`pipeline/`**: ML training pipeline:
    *   Model training scripts.
    *   Experiment tracking (MLflow).
    *   Model validation.
    *   Retraining logic triggered by drift detection.
*   **`monitoring/`**: Prometheus & Grafana configurations, alert rules, and performance dashboards for system health and model accuracy.
*   **`k8s/`**: Kubernetes manifests for deployments, services, and configurations for running FastAPI, Kafka, and the monitoring stack.
*   **`tests/`**: Unit, integration, and end-to-end (E2E) tests.
*   **`workflows/`**: GitHub Actions CI/CD workflows automating testing, training, container building, and deployment.

---

## Root Configuration Files

*   `Dockerfile`: Containerization setup for the FastAPI app and pipeline.
*   `requirements.txt`: Python package dependencies.
*   `dvc.yaml`: Data Version Control pipeline configuration.
*   `README.md`: Project documentation and guidelines.
