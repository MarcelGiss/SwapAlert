"""Swap Detection Model
=========================

This module implements a supervised learning pipeline that predicts whether a
patient's historical measurement data contains a *swap* – i.e. data from a
different patient has been mixed into the history.

The original dataset (``data/preprocessed_auftrag.csv``) is assumed to be clean
and contain no swaps.  To train a model we generate synthetic training examples:

* **No‑swap** examples – the raw data for a single patient.
* **Swap** examples – the raw data for a patient with the data of a second
  patient concatenated to it.  The label is ``1`` (swap) regardless of where the
  foreign data appears.

Feature engineering aggregates the raw rows of a patient history into a fixed‑
size numeric vector that can be consumed by standard classifiers.  The chosen
features are deliberately simple but capture the kind of inconsistencies a
swap would introduce:

* Number of rows (measurements)
* Number of distinct orders (``auftragsid``)
* Number of distinct analyt types
* Statistics of the measurement values (mean, std, min, max)
* Statistics of the time gaps between consecutive measurements (mean, std,
  min, max)

The pipeline is built with **pandas**, **scikit‑learn**, and **joblib** for model
serialization.  No custom model implementation is required – a
``RandomForestClassifier`` provides a good balance of performance and
interpretability for this tabular problem.

Usage
-----
::

    from src.swap_detection_model import train_and_save_model, predict_swap

    # Train the model and persist it to ``swap_model.pkl``
    train_and_save_model("data/preprocessed_auftrag.csv", "swap_model.pkl")

    # Later, load the model and predict for a new patient history (DataFrame)
    is_swap = predict_swap("swap_model.pkl", patient_history_df)

The module does **not** execute any code on import – all heavy work is inside the
functions so that it can be imported safely in other scripts or notebooks.
"""

from __future__ import annotations

import pathlib
import random
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_data(csv_path: pathlib.Path) -> pd.DataFrame:
    """Load the pre‑processed CSV.

    Parameters
    ----------
    csv_path:
        Path to ``preprocessed_auftrag.csv``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with appropriate dtypes. ``messtimestamp`` is parsed as a
        ``datetime64[ns]`` column for time‑gap calculations.
    """
    return pd.read_csv(csv_path, parse_dates=["messtimestamp"])


def _aggregate_patient(df: pd.DataFrame) -> pd.Series:
    """Create a fixed‑size feature vector for a single patient.

    The function expects *all* rows belonging to the same ``patientid``.
    It returns a ``pandas.Series`` where the index are feature names.
    """
    # Basic counts
    n_rows = len(df)
    n_orders = df["auftragsid"].nunique()
    n_analyt = df["analyt"].nunique()

    # Measurement value statistics (``messwert`` may contain NaNs – ignore them)
    messwert = df["messwert"].astype(float)
    mess_stats = messwert.agg(["mean", "std", "min", "max"]).fillna(0)

    # Time‑gap statistics – sort by timestamp first
    timestamps = df["messtimestamp"].sort_values()
    if len(timestamps) > 1:
        gaps = timestamps.diff().dt.total_seconds().iloc[1:]
        time_stats = gaps.agg(["mean", "std", "min", "max"]).fillna(0)
    else:
        time_stats = pd.Series([0, 0, 0, 0], index=["mean", "std", "min", "max"])

    # Assemble feature vector
    features = {
        "n_rows": n_rows,
        "n_orders": n_orders,
        "n_analyt": n_analyt,
        "messwert_mean": mess_stats["mean"],
        "messwert_std": mess_stats["std"],
        "messwert_min": mess_stats["min"],
        "messwert_max": mess_stats["max"],
        "gap_mean": time_stats["mean"],
        "gap_std": time_stats["std"],
        "gap_min": time_stats["min"],
        "gap_max": time_stats["max"],
    }
    return pd.Series(features)


def _generate_examples(
    df: pd.DataFrame, n_swap: int = 5, random_state: int | None = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic training examples.

    Parameters
    ----------
    df:
        Full dataset containing many patients.
    n_swap:
        Number of synthetic *swap* examples to create per patient.  The total
        number of *no‑swap* examples equals the number of distinct patients.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    X, y
        Feature matrix (DataFrame) and label vector (Series).  ``y = 0`` denotes
        a clean history, ``y = 1`` denotes a history containing a swap.
    """
    rng = random.Random(random_state)
    patient_ids = df["patientid"].unique()
    feature_rows = []
    labels = []

    # No‑swap examples – one per patient
    for pid in patient_ids:
        patient_df = df[df["patientid"] == pid]
        feature_rows.append(_aggregate_patient(patient_df))
        labels.append(0)

    # Swap examples – combine two different patients
    for _ in range(n_swap * len(patient_ids)):
        pid_a, pid_b = rng.sample(list(patient_ids), 2)
        df_a = df[df["patientid"] == pid_a]
        df_b = df[df["patientid"] == pid_b]
        # Concatenate the rows; keep original patientid of the first patient
        combined = pd.concat([df_a, df_b], ignore_index=True)
        # The combined history is labelled as a swap (1)
        feature_rows.append(_aggregate_patient(combined))
        labels.append(1)

    X = pd.DataFrame(feature_rows)
    y = pd.Series(labels, name="swap")
    return X, y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_and_save_model(
    csv_path: str | pathlib.Path,
    model_path: str | pathlib.Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Train a ``RandomForestClassifier`` and persist it to ``model_path``.

    A lightweight status report is printed to ``stdout`` so the user can follow
    the progress of each major step (loading, generating examples, training,
    evaluating, and saving).
    """
    print("[1/5] Loading data from", csv_path)
    data = _load_data(pathlib.Path(csv_path))

    print("[2/5] Generating synthetic training examples")
    X, y = _generate_examples(data, n_swap=5, random_state=random_state)

    print("[3/5] Splitting data into train / test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print("[4/5] Training RandomForestClassifier")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    print("[5/5] Evaluating model and saving")
    y_pred = clf.predict(X_test)
    print("=== Swap detection model evaluation ===")
    print(classification_report(y_test, y_pred, target_names=["no_swap", "swap"]))

    # Persist the model
    joblib.dump(clf, pathlib.Path(model_path))
    print(f"Model saved to {model_path}")


def load_model(model_path: str | pathlib.Path) -> RandomForestClassifier:
    """Load a previously saved model.

    Parameters
    ----------
    model_path:
        Path to the ``.pkl`` file created by :func:`train_and_save_model`.
    """
    return joblib.load(pathlib.Path(model_path))


def predict_swap(
    model_path: str | pathlib.Path, patient_history: pd.DataFrame
) -> bool:
    """Predict whether *patient_history* contains a swap.

    The function loads the model from ``model_path`` (cached on first call) and
    returns ``True`` if a swap is predicted, otherwise ``False``.
    """
    model = load_model(model_path)
    features = _aggregate_patient(patient_history).to_frame().T
    pred = model.predict(features)[0]
    return bool(pred)


if __name__ == "__main__":
    # Simple CLI entry point for quick experimentation
    import argparse

    parser = argparse.ArgumentParser(description="Train or use the swap detection model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--csv", required=False, help="Path to preprocessed_auftrag.csv", default="./data/preprocessed_auftrag.csv")
    train_parser.add_argument("--out", required=False, help="Path to store the trained model", default="./models/out")

    pred_parser = subparsers.add_parser("predict", help="Predict swap for a patient CSV")
    pred_parser.add_argument("--model", required=True, help="Path to the trained model file")
    pred_parser.add_argument("--patient-csv", required=True, help="CSV containing a single patient's rows")

    args = parser.parse_args()

    if args.command == "train":
        train_and_save_model(args.csv, args.out)
    elif args.command == "predict":
        patient_df = pd.read_csv(args.patient_csv, parse_dates=["messtimestamp"])
        result = predict_swap(args.model, patient_df)
        print("Swap detected:" if result else "No swap detected.")
