"""
main.py
FastAPI serving layer for the Dynamic Pricing Engine.

Exposes POST /predict — accepts raw order/weather/traffic/competitor data,
runs it through feature engineering, and returns a predicted price using
the locally saved XGBoost model.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

# pyrefly: ignore [missing-import]
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

import sys
import sqlite3

# Ensure project root and src directory are on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config import API_TITLE, API_VERSION, MODEL_DIR, FEATURE_COLUMNS, PREDICTIONS_DB_PATH
# pyrefly: ignore [missing-import]
from feature_engineering import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api")

MODEL_PATH = os.path.join(MODEL_DIR, "latest_model.json")
_model = None  # loaded on startup via the lifespan handler below


def load_model():
    """Loads the trained XGBoost model from disk. Raises if the model file doesn't exist."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run the training pipeline first."
        )
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
    return model


def init_db():
    """Initializes predictions.db schema."""
    conn = sqlite3.connect(PREDICTIONS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            city TEXT,
            predicted_price REAL,
            demand_supply_ratio REAL,
            peak_hour INTEGER,
            weather_factor REAL,
            traffic_factor REAL,
            base_price REAL,
            competitor_gap REAL,
            time_of_day INTEGER
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Predictions database initialized at %s", PREDICTIONS_DB_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    init_db()
    try:
        _model = load_model()
    except FileNotFoundError as e:
        # Don't crash the API on startup if no model exists yet — fail clearly at request time instead.
        logger.warning("Model not loaded at startup: %s", e)
        _model = None
    yield
    # No teardown logic needed currently; placeholder for future cleanup (e.g. closing connections).


app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)


# ---------------------------
# Request / Response Schemas
# ---------------------------
class OrderInput(BaseModel):
    order_id: str = Field(..., description="Unique order identifier")
    city: str = Field(..., description="City where the order originated")
    timestamp: str = Field(..., description="ISO 8601 timestamp of the order")
    demand_units: int = Field(..., ge=0, description="Active demand units in the city")
    supply_units: int = Field(..., ge=0, description="Available supply units in the city")
    base_price: float = Field(..., gt=0, description="Standard price before dynamic adjustment")


class WeatherInput(BaseModel):
    condition: str = Field(..., description="Weather condition, e.g. Clear, Rain, Storm")
    temperature_c: float = Field(default=20.0, description="Temperature in Celsius")


class TrafficInput(BaseModel):
    traffic_level: str = Field(..., description="Traffic level, e.g. Low, Moderate, High, Severe")
    avg_speed_kmph: float = Field(default=30.0, description="Average traffic speed in km/h")


class CompetitorInput(BaseModel):
    competitor_price: float = Field(..., gt=0, description="Competitor's current price for the same service")


class PredictRequest(BaseModel):
    order: OrderInput
    weather: WeatherInput
    traffic: TrafficInput
    competitor: CompetitorInput


class PredictResponse(BaseModel):
    city: str
    predicted_price: float
    features_used: dict


# ---------------------------
# Routes
# ---------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train the model first via the training pipeline.",
        )

    record = {
        "order": request.order.model_dump(),
        "weather": request.weather.model_dump(),
        "traffic": request.traffic.model_dump(),
        "competitor": request.competitor.model_dump(),
    }

    try:
        features = engineer_features(record)
    except Exception as e:
        logger.error("Feature engineering failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Failed to engineer features: {e}")

    # Build the feature vector in the exact column order the model was trained on
    try:
        feature_vector = [[features[col] for col in FEATURE_COLUMNS]]
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Missing expected feature column {e}. Check FEATURE_COLUMNS vs engineer_features().",
        )

    predicted_price = float(_model.predict(feature_vector)[0])

    try:
        conn = sqlite3.connect(PREDICTIONS_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (
                timestamp, city, predicted_price, demand_supply_ratio,
                peak_hour, weather_factor, traffic_factor, base_price,
                competitor_gap, time_of_day
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            features["timestamp"],
            features["city"],
            round(predicted_price, 2),
            features["demand_supply_ratio"],
            features["peak_hour"],
            features["weather_factor"],
            features["traffic_factor"],
            features["base_price"],
            features["competitor_gap"],
            features["time_of_day"]
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error("Failed to log prediction to database: %s", e)

    return PredictResponse(
        city=features["city"],
        predicted_price=round(predicted_price, 2),
        features_used={col: features[col] for col in FEATURE_COLUMNS},
    )


if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    uvicorn.run(app, host=API_HOST, port=API_PORT)