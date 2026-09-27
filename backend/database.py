"""
database.py

Minimal SQLite persistence layer for prediction history.

We use the plain sqlite3 module (no ORM) to keep things simple and
interview-defensible: it's easy to explain exactly what SQL is being run.
"""

import sqlite3
from pathlib import Path
from typing import Optional

# Database file lives next to the backend code.
DB_PATH = Path(__file__).parent / "predictions.db"


def get_connection() -> sqlite3.Connection:
    """Open a new SQLite connection with row access by column name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the predictions table if it doesn't already exist."""
    conn = get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                equipment_id TEXT NOT NULL,
                air_temperature REAL NOT NULL,
                process_temperature REAL NOT NULL,
                rotational_speed REAL NOT NULL,
                torque REAL NOT NULL,
                tool_wear REAL NOT NULL,
                type TEXT NOT NULL,
                failure_prediction INTEGER NOT NULL,
                failure_probability REAL NOT NULL,
                anomaly_detected INTEGER NOT NULL,
                anomaly_score REAL NOT NULL,
                maintenance_status TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def insert_prediction(record: dict) -> int:
    """Insert one prediction record and return its new row id."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO predictions (
                equipment_id, air_temperature, process_temperature,
                rotational_speed, torque, tool_wear, type,
                failure_prediction, failure_probability,
                anomaly_detected, anomaly_score, maintenance_status, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["equipment_id"],
                record["air_temperature"],
                record["process_temperature"],
                record["rotational_speed"],
                record["torque"],
                record["tool_wear"],
                record["type"],
                record["failure_prediction"],
                record["failure_probability"],
                record["anomaly_detected"],
                record["anomaly_score"],
                record["maintenance_status"],
                record["timestamp"],
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def fetch_recent_predictions(limit: int = 50) -> list[sqlite3.Row]:
    """Return the most recent `limit` predictions, newest first."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return rows
    finally:
        conn.close()


def fetch_predictions_by_equipment(equipment_id: str, limit: int = 50) -> list[sqlite3.Row]:
    """Return recent predictions for a single equipment_id, newest first."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM predictions
            WHERE equipment_id = ?
            ORDER BY id DESC LIMIT ?
            """,
            (equipment_id, limit),
        ).fetchall()
        return rows
    finally:
        conn.close()
