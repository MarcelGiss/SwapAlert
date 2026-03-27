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

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# Additional supervised learning models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _flatten_sample(df: pd.DataFrame) -> np.ndarray:
    """Flatten a ``DataLoader`` sample to a 1‑D ``numpy`` array.

    ``df`` has analytes as rows and ``history_length`` columns.  Missing values
    are kept as ``np.nan`` – they will be handled later by the imputer.
    """
    # ``values`` returns a 2‑D ``ndarray``; ``ravel`` flattens in row‑major order.
    return df.values.ravel()




from typing import Optional


def _run_training(args: argparse.Namespace) -> None:
    """Execute the full training pipeline.

    This function consolidates the dataset preparation, model training,
    evaluation and persistence steps that were previously located in ``main``.
    Keeping this logic in a dedicated function makes the script easier to
    test and reuse programmatically.
    """
    # ---------------------------------------------------------------------
    # Dataset generation (with progress reporting)
    # ---------------------------------------------------------------------
    train_dataset_path = pathlib.Path(args.train_dataset_path)
    if train_dataset_path.is_file():
        print(f"Loading previously generated test dataset from {train_dataset_path}")
        import pandas as pd

        df = pd.read_csv(train_dataset_path)
        if "label" not in df.columns:
            raise ValueError("CSV dataset must contain a 'label' column")
        y = df["label"].values
        X = df.drop(columns=["label"]).values
    else:
        raise ValueError("No train dataset found")

    # Split into train / test.
    # Determine a safe test split size.  For very small synthetic datasets the
    # default 20% split can result in a test set with fewer than two classes,
    # causing ``train_test_split`` to raise a ValueError.  We fallback to a 50%
    # split when the total number of samples is less than 5.
    split_ratio = 0.2 if X.shape[0] >= 5 else 0.5
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=42, stratify=y
    )

    # Choose classifier based on requested model type.
    if args.model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif args.model_type == "gradient_boosting":
        clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            random_state=42,
            verbose=args.verbose,
        )
    elif args.model_type == "hist_gradient_boosting":
        # Using HistGradientBoostingClassifier for faster training and early stopping support
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            max_depth=None,
            early_stopping=True,
            l2_regularization=0.0,
            random_state=42,
            verbose=args.verbose,
        )
    elif args.model_type == "logistic_regression":
        # Logistic Regression with L2 regularization. Use the 'saga' solver which
        # supports parallelism (n_jobs) and verbose output, making it faster on
        # larger datasets. The ``verbose`` flag is tied to the CLI ``--verbose``
        # argument so users can enable per‑iteration progress information.
        clf = LogisticRegression(
            max_iter=10,
            C=1.0,
            solver="saga",
            n_jobs=-1,
            random_state=42,
            verbose=args.verbose,
        )
    elif args.model_type == "svm":
        # Linear Support Vector Machine – much faster than the RBF kernel on
        # high‑dimensional tabular data. ``LinearSVC`` supports a ``verbose``
        # flag for progress reporting and can leverage all CPU cores.
        from sklearn.svm import LinearSVC

        clf = LinearSVC(
            C=1.0,
            random_state=42,
            max_iter=10,
            verbose=args.verbose,
        )
    elif args.model_type == "mlp":
        # Multi-layer Perceptron with a single hidden layer. Increase ``max_iter``
        # modestly for better convergence while keeping training time reasonable.
        # ``verbose`` already provides per‑epoch progress when enabled.
        clf = MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=10,
            learning_rate_init=0.1,
            random_state=42,
            verbose=args.verbose,
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # Build a scikit‑learn pipeline using the selected classifier.
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )

    print("Training model …")
    pipeline.fit(X_train, y_train)

    print("Evaluating on test set …")
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["no_swap", "swap"])
    print(report)

    # ---------------------------------------------------------------------
    # Determine next model version based on existing files in the models dir.
    # Files follow the pattern ``swap_detector_<N>.pkl`` where <N> is an integer.
    # ---------------------------------------------------------------------
    models_dir = pathlib.Path("../models")
    models_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(models_dir.glob("swap_detector_*.pkl"))
    if existing:
        # Extract numeric suffixes and compute the next version.
        versions = [int(p.stem.split("_")[-1]) for p in existing if p.stem.split("_")[-1].isdigit()]
        next_version = max(versions) + 1 if versions else 1
    else:
        next_version = 1

    model_path = models_dir / f"swap_detector_{next_version}.pkl"
    # Persist the pipeline. Use ``pickle`` to avoid a hard dependency on the
    # external ``joblib`` package, which may not be installed in all
    # environments. The file extension remains ``.pkl`` for compatibility.
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {model_path}")

    # ---------------------------------------------------------------------
    # Persist the evaluation report in the ``trains`` directory using a
    # naming scheme that mirrors existing files, e.g. ``random_forest_2.txt``.
    # ---------------------------------------------------------------------
    trains_dir = pathlib.Path("../trainings")
    trains_dir.mkdir(parents=True, exist_ok=True)
    report_path = trains_dir / f"{args.model_type}_{next_version}.txt"
    report_path.write_text(report)
    print(f"Evaluation report saved to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train swap detection model")
    parser.add_argument(
        "--train-dataset-path",
        type=str,
        default="../data/train_dataset_samp100k_hist20_min5_50swap.csv",
        help="Path where the generated test dataset is cached (CSV format)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="logistic_regression",
        choices=[
            "random_forest",
            "gradient_boosting",
            "hist_gradient_boosting",
            "logistic_regression",
            "svm",
            "mlp",
        ],
        help="Select which supervised model to train. Added options: logistic_regression, svm, mlp.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output during model training (shows progress for gradient boosting).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for dataset generation (default: all CPU cores - 1)",
    )
    args = parser.parse_args()

    # Delegate the remaining workflow to the dedicated helper.
    _run_training(args)


if __name__ == "__main__":
    main()
