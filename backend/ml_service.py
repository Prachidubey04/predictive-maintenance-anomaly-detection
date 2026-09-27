"""
ml_service.py

Loads the pre-trained ML artifacts ONCE at API startup and exposes a single
`predict` function used by the /predict endpoint. Models are never retrained
here -- this module only does inference.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Artifacts were produced by ml/train.py and live in the project-level models/ dir.
MODELS_DIR = Path(__file__).parent.parent / "models"

_failure_model = None
_scaler = None
_anomaly_model = None
_feature_names = None


def load_artifacts() -> None:
    """Load all joblib artifacts into module-level variables. Call once at startup."""
    global _failure_model, _scaler, _anomaly_model, _feature_names

    _failure_model = joblib.load(MODELS_DIR / "failure_model.joblib")
    _scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    _anomaly_model = joblib.load(MODELS_DIR / "anomaly_model.joblib")
    _feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")


def artifacts_loaded() -> bool:
    """Used by /health to report whether models are ready to serve predictions."""
    return all(
        x is not None for x in (_failure_model, _scaler, _anomaly_model, _feature_names)
    )


def _build_feature_row(reading) -> pd.DataFrame:
    """
    Reconstruct the exact same engineered feature row used during training
    (see ml/preprocess.py), from a single SensorReading input.
    """
    # Raw sensor values, engineered features (must match preprocess.py exactly).
    temp_diff = reading.process_temperature - reading.air_temperature
    power = reading.torque * reading.rotational_speed * (2 * np.pi / 60)
    torque_wear_interaction = reading.torque * reading.tool_wear

    row = {
        "Air temperature [K]": reading.air_temperature,
        "Process temperature [K]": reading.process_temperature,
        "Rotational speed [rpm]": reading.rotational_speed,
        "Torque [Nm]": reading.torque,
        "Tool wear [min]": reading.tool_wear,
        "temp_diff": temp_diff,
        "power": power,
        "torque_wear_interaction": torque_wear_interaction,
        # One-hot encode Type manually to match training-time columns.
        "Type_H": 1 if reading.type == "H" else 0,
        "Type_L": 1 if reading.type == "L" else 0,
        "Type_M": 1 if reading.type == "M" else 0,
    }

    # Reindex to the exact column order the model was trained on.
    df = pd.DataFrame([row])[_feature_names]
    return df


def predict(reading) -> dict:
    """
    Run both the supervised failure model and the unsupervised anomaly
    detector on a single sensor reading, and derive a maintenance status.
    """
    if not artifacts_loaded():
        raise RuntimeError("ML artifacts are not loaded. Call load_artifacts() at startup.")

    X = _build_feature_row(reading)
    X_scaled = _scaler.transform(X)

    # Supervised prediction: probability of Machine failure = 1.
    failure_proba = float(_failure_model.predict_proba(X_scaled)[0, 1])
    failure_pred = int(_failure_model.predict(X_scaled)[0])

    # Unsupervised anomaly detection: -1 means anomalous, 1 means normal.
    anomaly_raw = int(_anomaly_model.predict(X_scaled)[0])
    anomaly_detected = anomaly_raw == -1
    anomaly_score = float(_anomaly_model.decision_function(X_scaled)[0])

    # Derive a simple, explainable maintenance status from both signals.
    # This is a business-rule layer on top of the two ML outputs, not a
    # separate model.
    if failure_pred == 1:
        maintenance_status = "URGENT: Failure predicted - schedule maintenance"
    elif anomaly_detected:
        maintenance_status = "WATCH: Unusual sensor readings - inspect equipment"
    else:
        maintenance_status = "OK: Normal operation"

    return {
        "failure_prediction": failure_pred,
        "failure_probability": failure_proba,
        "anomaly_detected": anomaly_detected,
        "anomaly_score": anomaly_score,
        "maintenance_status": maintenance_status,
    }
