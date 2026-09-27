"""
train.py

Trains and evaluates supervised models for machine-failure prediction, plus
an unsupervised Isolation Forest for general anomaly detection.

Supervised task:   predict `Machine failure` (0/1) from sensor readings.
Unsupervised task: flag statistically unusual sensor readings (anomalies).

These two are DIFFERENT things:
- A supervised "failure" prediction means the model was trained on labeled
  failure/non-failure examples and predicts that specific outcome.
- An "anomaly" from Isolation Forest only means the reading looks statistically
  unusual compared to normal operation. It does NOT mean the machine has
  failed or will fail -- it is a flag for "this looks different, worth a
  closer look."
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from preprocess import load_dataset, get_feature_target_split

DATA_PATH = "data/ai4i2020.csv"
MODELS_DIR = Path("models")
RANDOM_STATE = 42


def evaluate_model(name: str, model, X_test, y_test):
    """Print standard classification metrics for a fitted model."""
    y_pred = model.predict(X_test)

    # predict_proba gives us the probability of the positive class (failure=1),
    # which we need for ROC-AUC and for a risk score in the API later.
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n=== {name} ===")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["No Failure", "Failure"]))


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    # 1. Load and preprocess data.
    df = load_dataset(DATA_PATH)
    X, y = get_feature_target_split(df)
    feature_names = X.columns.tolist()

    # 2. Train/test split. Stratify on y to preserve the ~3.4% failure rate
    # in both splits, since the dataset is imbalanced.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 3. Scale features. Logistic Regression needs scaling to converge well;
    # tree-based models (Random Forest) don't strictly need it, but we reuse
    # the same scaled matrix for consistency and simplicity.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Baseline model: Logistic Regression with class_weight='balanced' to
    # counter the 96.6% / 3.4% class imbalance.
    baseline = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    )
    baseline.fit(X_train_scaled, y_train)
    evaluate_model("Baseline: Logistic Regression", baseline, X_test_scaled, y_test)

    # 5. Stronger model: Random Forest, also balanced for the minority class.
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)
    evaluate_model("Stronger model: Random Forest", rf, X_test_scaled, y_test)

    # 6. Unsupervised anomaly detection with Isolation Forest.
    # Trained WITHOUT using the failure labels -- it only learns what "normal"
    # sensor behaviour looks like and flags statistical outliers.
    # contamination is set close to the true failure rate purely as a rough
    # guide for how many points to flag; it does not use the labels directly.
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=0.034,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso_forest.fit(X_train_scaled)

    # Sanity-check: how well do IsolationForest anomaly flags line up with
    # actual failures? (Informational only -- anomaly != failure.)
    anomaly_pred = iso_forest.predict(X_test_scaled)  # -1 = anomaly, 1 = normal
    anomaly_flag = (anomaly_pred == -1).astype(int)
    overlap_recall = recall_score(y_test, anomaly_flag)
    print("\n=== Isolation Forest (unsupervised anomaly detector) ===")
    print("NOTE: This model does NOT predict machine failure. It flags sensor")
    print("readings that are statistically unusual vs. normal operation.")
    print(f"Fraction of test rows flagged as anomalies: {anomaly_flag.mean():.4f}")
    print(f"Of actual failures, fraction that were ALSO flagged as anomalies: {overlap_recall:.4f}")
    print("(This overlap is informational only, not a claim of predictive accuracy.)")

    # 7. Persist the best supervised model, scaler, and anomaly detector.
    # Random Forest is our production model based on stronger F1/ROC-AUC below.
    joblib.dump(rf, MODELS_DIR / "failure_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(iso_forest, MODELS_DIR / "anomaly_model.joblib")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.joblib")

    print(f"\nSaved model artifacts to: {MODELS_DIR.resolve()}")


if __name__ == "__main__":
    main()
