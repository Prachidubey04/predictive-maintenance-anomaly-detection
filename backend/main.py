"""
main.py

FastAPI application entry point. Loads ML artifacts once at startup, exposes
health/predict/history endpoints, and persists predictions to SQLite.
"""

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend import database
from backend import ml_service
from backend.schemas import (
    SensorReading,
    PredictionResponse,
    PredictionHistoryItem,
    HealthResponse,
)

app = FastAPI(
    title="Vehicle Telemetry Anomaly Detection API",
    description=(
        "Serves supervised machine-failure predictions and unsupervised "
        "anomaly detection over equipment sensor telemetry."
    ),
    version="1.0.0",
)

# Allow the React dev server (typically localhost:3000 / 5173) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # kept permissive for local dev; tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    """Load ML artifacts and initialize the SQLite schema once, at boot."""
    ml_service.load_artifacts()
    database.init_db()


@app.get("/health", response_model=HealthResponse)
def health():
    """Basic liveness/readiness check, including whether models are loaded."""
    loaded = ml_service.artifacts_loaded()
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        anomaly_model_loaded=loaded,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    """
    Run failure prediction + anomaly detection on one sensor reading,
    persist the result to SQLite, and return it to the caller.
    """
    try:
        result = ml_service.predict(reading)
    except RuntimeError as exc:
        # Models not loaded (shouldn't happen after startup, but guard anyway).
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        # Any unexpected inference error -> 500 with a clear message.
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    timestamp = datetime.now(timezone.utc)

    record = {
        "equipment_id": reading.equipment_id,
        "air_temperature": reading.air_temperature,
        "process_temperature": reading.process_temperature,
        "rotational_speed": reading.rotational_speed,
        "torque": reading.torque,
        "tool_wear": reading.tool_wear,
        "type": reading.type,
        "failure_prediction": result["failure_prediction"],
        "failure_probability": result["failure_probability"],
        "anomaly_detected": int(result["anomaly_detected"]),
        "anomaly_score": result["anomaly_score"],
        "maintenance_status": result["maintenance_status"],
        "timestamp": timestamp.isoformat(),
    }
    database.insert_prediction(record)

    return PredictionResponse(
        equipment_id=reading.equipment_id,
        failure_prediction=result["failure_prediction"],
        failure_probability=result["failure_probability"],
        anomaly_detected=result["anomaly_detected"],
        anomaly_score=result["anomaly_score"],
        maintenance_status=result["maintenance_status"],
        timestamp=timestamp,
    )


def _row_to_history_item(row) -> PredictionHistoryItem:
    """Convert a sqlite3.Row from the predictions table into the response schema."""
    return PredictionHistoryItem(
        id=row["id"],
        equipment_id=row["equipment_id"],
        air_temperature=row["air_temperature"],
        process_temperature=row["process_temperature"],
        rotational_speed=row["rotational_speed"],
        torque=row["torque"],
        tool_wear=row["tool_wear"],
        type=row["type"],
        failure_prediction=row["failure_prediction"],
        failure_probability=row["failure_probability"],
        anomaly_detected=bool(row["anomaly_detected"]),
        anomaly_score=row["anomaly_score"],
        maintenance_status=row["maintenance_status"],
        timestamp=row["timestamp"],
    )


@app.get("/predictions", response_model=list[PredictionHistoryItem])
def get_predictions(limit: int = Query(50, ge=1, le=500, description="Max rows to return")):
    """Return the most recent prediction history, newest first."""
    rows = database.fetch_recent_predictions(limit=limit)
    return [_row_to_history_item(r) for r in rows]


@app.get("/predictions/{equipment_id}", response_model=list[PredictionHistoryItem])
def get_predictions_for_equipment(
    equipment_id: str,
    limit: int = Query(50, ge=1, le=500, description="Max rows to return"),
):
    """Return prediction history for a single equipment_id, newest first."""
    rows = database.fetch_predictions_by_equipment(equipment_id, limit=limit)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction history found for equipment_id '{equipment_id}'",
        )
    return [_row_to_history_item(r) for r in rows]
