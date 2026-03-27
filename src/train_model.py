"""Training script for the swap detection model.

The goal of the model is to predict whether a patient's historical order
data contains a *swap* – i.e. at least one of the ``history_length`` recent
orders originates from a different patient.  The :class:`~src.data_loader.DataLoader`
already provides a convenient way to generate synthetic samples with the
``add_synthetic_swap`` flag.

This script builds a binary classifier using scikit‑learn.  The pipeline is:

1. Generate a balanced dataset of samples with and without a synthetic swap.
2. Flatten each sample (a ``DataFrame`` of shape ``n_analytes × history_length``)
   into a 1‑D feature vector.
3. Impute missing values (``NaN``) with the median of each feature.
4. Scale the features with ``StandardScaler``.
5. Train a ``RandomForestClassifier`` – a robust choice for tabular data with
   many missing values.
6. Evaluate on a hold‑out test split and print a short classification report.
7. Persist the trained model (pipeline) to ``models/swap_detector.pkl`` for
   later inference.

The script can be executed directly with the project's Python interpreter:

```
~/.pyenv/versions/3.14.0/envs/hackathon_wue26/bin/python -m src.train_model
```

The path to the CSV data can be overridden via the ``--data`` CLI argument.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Local imports
from data_loader import DataLoader


def _flatten_sample(df: pd.DataFrame) -> np.ndarray:
    """Flatten a ``DataLoader`` sample to a 1‑D ``numpy`` array.

    ``df`` has analytes as rows and ``history_length`` columns.  Missing values
    are kept as ``np.nan`` – they will be handled later by the imputer.
    """
    # ``values`` returns a 2‑D ``ndarray``; ``ravel`` flattens in row‑major order.
    return df.values.ravel()


def _generate_dataset(
    loader: DataLoader,
    n_samples: int,
    swap_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate feature matrix ``X`` and label vector ``y`` with console progress.

    Parameters
    ----------
    loader:
        Instance of :class:`DataLoader` pointing to the pre‑processed CSV.
    n_samples:
        Total number of samples to generate.
    swap_ratio:
        Fraction of samples that contain a synthetic swap (label ``1``).
    """
    n_swap = int(n_samples * swap_ratio)
    n_clean = n_samples - n_swap

    # Helper to generate a single sample and label.
    def _gen_one(swap: bool) -> Tuple[np.ndarray, int]:
        sample = loader.get_sample(add_synthetic_swap=swap)
        return _flatten_sample(sample), int(swap)

    # Use joblib for simple parallelism if many samples are requested.
    from joblib import Parallel, delayed

    total = n_samples
    generated = 0
    def _log_progress(step: int = 1):
        nonlocal generated
        generated += step
        percent = (generated / total) * 100
        print(f"\rGenerating samples: {generated}/{total} ({percent:.1f}%)", end="", flush=True)

    # Generate clean and swapped samples in parallel batches.
    # Determine number of workers (use all cores but limit to avoid oversubscription).
    import os
    n_jobs = max(1, os.cpu_count() - 1)

    # Clean samples
    clean_results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_gen_one)(False) for _ in range(n_clean)
    )
    for _ in range(n_clean):
        _log_progress()

    # Swapped samples
    swap_results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_gen_one)(True) for _ in range(n_swap)
    )
    for _ in range(n_swap):
        _log_progress()

    print()  # newline after progress bar
    # Combine results
    X = [feat for feat, _ in clean_results] + [feat for feat, _ in swap_results]
    y = [label for _, label in clean_results] + [label for _, label in swap_results]
    return np.stack(X, axis=0), np.array(y, dtype=int)


from typing import Optional


def _run_training(loader: Optional[DataLoader], args: argparse.Namespace) -> None:
    """Execute the full training pipeline.

    This function consolidates the dataset preparation, model training,
    evaluation and persistence steps that were previously located in ``main``.
    Keeping this logic in a dedicated function makes the script easier to
    test and reuse programmatically.
    """
    # ---------------------------------------------------------------------
    # Dataset generation (with progress reporting)
    # ---------------------------------------------------------------------
    test_dataset_path = pathlib.Path(args.test_dataset_path)
    if test_dataset_path.is_file():
        print(f"Loading previously generated test dataset from {test_dataset_path}")
        loaded = np.load(test_dataset_path)
        X, y = loaded["X"], loaded["y"]
    else:
        # If no cached dataset exists we must generate it. Ensure we have a DataLoader.
        if loader is None:
            raise RuntimeError(
                "DataLoader is required to generate a new dataset but none was provided."
            )
        print("Generating dataset …")
        X, y = _generate_dataset(loader, n_samples=args.samples, swap_ratio=0.5)
        # Save for fast subsequent runs
        print(f"Saving generated dataset to {test_dataset_path}")
        np.savez_compressed(test_dataset_path, X=X, y=y)

    # Split into train / test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build a scikit‑learn pipeline.
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training model …")
    pipeline.fit(X_train, y_train)

    print("Evaluating on test set …")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["no_swap", "swap"]))

    # Persist the pipeline.
    model_path = pathlib.Path("models/swap_detector.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train swap detection model")
    parser.add_argument(
        "--data",
        type=str,
        default="../data/preprocessed_auftrag.csv",
        help="Path to the pre‑processed CSV file",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of training samples to generate",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=5,
        help="History length used by DataLoader",
    )
    parser.add_argument(
        "--test-dataset-path",
        type=str,
        default="../data/test_dataset.npz",
        help="Path where the generated test dataset is cached",
    )
    parser.add_argument(
        "--required-auftraege-per-patient",
        type=int,
        default=5,
        help="Minimum number of distinct orders a patient must have to be included",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for dataset generation (default: all CPU cores - 1)",
    )
    args = parser.parse_args()

    data_path = pathlib.Path(args.data)
    test_dataset_path = pathlib.Path(args.test_dataset_path)

    if test_dataset_path.is_file():
        print(
            "Cached dataset detected; skipping CSV load. "
            "If you need to regenerate the dataset, delete the test dataset file."
        )
        loader: Optional[DataLoader] = None
    else:
        if not data_path.is_file():
            sys.exit(f"Data file not found: {data_path}")
        loader = DataLoader(
            path=str(data_path),
            history_length=args.history_length,
            required_auftraege_per_patient=args.required_auftraege_per_patient,
        )

    # Delegate the remaining workflow to the dedicated helper.
    _run_training(loader, args)


if __name__ == "__main__":
    main()
