"""
preprocess.py

Loads the AI4I 2020 dataset, validates it, and builds the feature matrix (X)
and target vector (y) used for supervised machine-failure prediction.

IMPORTANT: The dataset contains five failure-mode sub-labels (TWF, HDF, PWF,
OSF, RNF) that are effectively post-hoc annotations describing *why* a machine
failed. Machine failure = 1 whenever any of these flags is 1, so including them
as model inputs would leak the answer directly into the features. They are
therefore always dropped from X.
"""

import numpy as np
import pandas as pd

# Columns that are identifiers only, not predictive sensor signals.
ID_COLUMNS = ["UDI", "Product ID"]

# Failure-mode sub-labels. These leak the target and must never be used as features.
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

TARGET_COLUMN = "Machine failure"

# Original sensor columns we keep as-is.
RAW_SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

CATEGORICAL_COLUMNS = ["Type"]


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV and run basic sanity checks."""
    df = pd.read_csv(csv_path)

    # Basic validation: make sure the columns we depend on actually exist.
    required_columns = (
        ID_COLUMNS + CATEGORICAL_COLUMNS + RAW_SENSOR_COLUMNS + [TARGET_COLUMN] + LEAKAGE_COLUMNS
    )
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    # Fail loudly if there are unexpected nulls (the raw AI4I dataset has none,
    # but future data drops could break this assumption).
    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values; please clean before training.")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-motivated features derived from the raw sensor columns."""
    df = df.copy()

    # Thermal stress: gap between process and air temperature. A small/negative
    # gap is associated with heat-dissipation failures (HDF) in the AI4I data
    # generation rules.
    df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]

    # Mechanical power delivered by the tool, in Watts.
    # Power = Torque (Nm) * Angular speed (rad/s); rpm -> rad/s via (2*pi/60).
    df["power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)

    # Torque-wear interaction: captures accumulated mechanical stress, related
    # to overstrain failures (OSF) which trigger past a torque*wear threshold.
    df["torque_wear_interaction"] = df["Torque [Nm]"] * df["Tool wear [min]"]

    return df


def get_feature_target_split(df: pd.DataFrame):
    """
    Build the final feature matrix X (with engineered features and one-hot
    encoded Type) and target vector y. Drops IDs and leakage columns.
    """
    df = engineer_features(df)

    feature_columns = (
        RAW_SENSOR_COLUMNS
        + ["temp_diff", "power", "torque_wear_interaction"]
        + CATEGORICAL_COLUMNS
    )

    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    # One-hot encode the categorical "Type" column (L/M/H product quality).
    X = pd.get_dummies(X, columns=CATEGORICAL_COLUMNS, drop_first=False)

    return X, y


if __name__ == "__main__":
    # Quick manual check when run directly.
    data = load_dataset("data/ai4i2020.csv")
    X, y = get_feature_target_split(data)
    print("Feature columns:", X.columns.tolist())
    print("X shape:", X.shape, "y shape:", y.shape)
    print("Class balance:\n", y.value_counts(normalize=True))
