"""
schemas.py

Pydantic request/response models for the FastAPI backend.
"""

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """Input payload for POST /predict.

    Mirrors the raw sensor columns from the AI4I 2020 dataset that the
    trained model was built on (see ml/preprocess.py RAW_SENSOR_COLUMNS).
    """

    equipment_id: str = Field(..., min_length=1, description="Identifier for the machine/equipment")
    air_temperature: float = Field(..., gt=0, description="Air temperature in Kelvin")
    process_temperature: float = Field(..., gt=0, description="Process temperature in Kelvin")
    rotational_speed: float = Field(..., gt=0, description="Rotational speed in rpm")
    torque: float = Field(..., ge=0, description="Torque in Nm")
    tool_wear: float = Field(..., ge=0, description="Tool wear in minutes")
    type: Literal["L", "M", "H"] = Field(..., description="Product quality variant: L, M, or H")


class PredictionResponse(BaseModel):
    """Response returned by POST /predict and used in history listings."""

    equipment_id: str
    failure_prediction: int  # 0 = no failure predicted, 1 = failure predicted
    failure_probability: float  # supervised model's probability of failure (0-1)
    anomaly_detected: bool  # unsupervised Isolation Forest flag - NOT the same as failure
    anomaly_score: float  # raw Isolation Forest decision score (lower = more anomalous)
    maintenance_status: str  # human-readable status derived from the two signals above
    timestamp: datetime


class PredictionHistoryItem(PredictionResponse):
    """A stored prediction row, including its DB id and the sensor inputs."""

    id: int
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float
    type: str


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    model_loaded: bool
    anomaly_model_loaded: bool
