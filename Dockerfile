# ---------------------------
# Dynamic Pricing Engine — API Container
# Serves the FastAPI /predict endpoint
# ---------------------------
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr (so logs show up immediately)
ENV PYTHONUNBUFFERED=1
# Ensure imports resolve correctly regardless of where the process is launched from:
# /app is the project root (where config.py lives), /app/src holds business logic.
ENV PYTHONPATH=/app:/app/src

WORKDIR /app

# Install system dependencies needed by some ML libraries (xgboost/lightgbm need libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (separate layer so this is cached
# unless requirements.txt actually changes — speeds up rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY config.py .
COPY src/ src/
COPY app/ app/
COPY pipeline/ pipeline/

# Trained model must already exist at pipeline/models/latest_model.json
# (run train_pipeline.py before building, or mount the model directory as a volume)

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]