"""Utility module for generating the test dataset used by the training script.

The original implementation of dataset generation lived inside ``src/train_model.py``.
For better separation of concerns we now expose a dedicated ``generate_dataset``
function here.  It mirrors the previous behaviour:

* Uses a :class:`~src.data_loader.DataLoader` instance to create synthetic samples.
* Supports an optional ``swap_ratio`` to control the proportion of swapped samples.
* Optionally saves the generated ``X`` and ``y`` arrays to a ``.npz`` file for
  fast subsequent runs.

The training script imports :func:`generate_dataset` and calls it only when the
cached dataset file does not already exist.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Tuple

import numpy as np
from joblib import Parallel, delayed

# Local imports
from data_loader import DataLoader


def _flatten_sample(df: "pandas.DataFrame") -> np.ndarray:
    """Flatten a ``DataLoader`` sample to a 1‑D ``numpy`` array.

    The ``DataLoader`` returns a ``DataFrame`` where rows represent analytes and
    columns represent the ``history_length`` recent orders.  ``values`` yields a
    2‑D ``ndarray`` which we flatten in row‑major order.
    """
    return df.values.ravel()


def _gen_one(loader: DataLoader, swap: bool) -> Tuple[np.ndarray, int]:
    """Generate a single sample and its label.

    Parameters
    ----------
    loader:
        The ``DataLoader`` instance.
    swap:
        ``True`` to request a synthetic swap (label ``1``); ``False`` for a clean
        sample (label ``0``).
    """
    sample = loader.get_sample(add_synthetic_swap=swap)
    return _flatten_sample(sample), int(swap)


def generate_dataset(
    loader: DataLoader,
    n_samples: int,
    swap_ratio: float = 0.5,
    save_path: pathlib.Path | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a balanced feature matrix ``X`` and label vector ``y``.

    Parameters
    ----------
    loader:
        Instance of :class:`DataLoader` pointing to the pre‑processed CSV.
    n_samples:
        Total number of samples to generate.
    swap_ratio:
        Fraction of samples that contain a synthetic swap (label ``1``).
    save_path:
        Optional path where the generated dataset is saved as a CSV file.
        The CSV will contain all feature columns and a ``label`` column.
        If provided, the file will be overwritten if it already exists.
    """
    n_swap = int(n_samples * swap_ratio)
    n_clean = n_samples - n_swap

    import os
    n_jobs = max(1, (os.cpu_count() or 1) - 1)

    clean_results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_gen_one)(loader, False) for _ in range(n_clean)
    )
    swap_results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_gen_one)(loader, True) for _ in range(n_swap)
    )

    X = [feat for feat, _ in clean_results] + [feat for feat, _ in swap_results]
    y = [label for _, label in clean_results] + [label for _, label in swap_results]

    X_arr = np.stack(X, axis=0)
    y_arr = np.array(y, dtype=int)

    if save_path is not None:
        # Ensure parent directory exists.
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Save as CSV with a ``label`` column for compatibility with training script.
        import pandas as pd

        df = pd.DataFrame(X_arr)
        df["label"] = y_arr
        df.to_csv(save_path, index=False)

    return X_arr, y_arr


def _parse_args() -> argparse.Namespace:
    """Parse command‑line arguments for standalone dataset generation."""
    parser = argparse.ArgumentParser(description="Generate train dataset for SwapAlert")
    parser.add_argument(
        "--data",
        type=str,
        required=False,
        default="../data/preprocessed_auftrag.csv",
        help="Path to the pre‑processed CSV file used by DataLoader",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100000,
        help="Total number of samples to generate",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=20,
        help="History length used by DataLoader",
    )
    parser.add_argument(
        "--required-auftraege-per-patient",
        type=int,
        default=5,
        help="Minimum number of distinct orders a patient must have to be included",
    )
    parser.add_argument(
        "--swap-ratio",
        type=float,
        default=0.5,
        help="Fraction of samples that contain a synthetic swap",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default="../data/train_dataset_samp100_hist20_min5.csv",
        help="File path where the generated dataset will be saved (CSV format)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_path = pathlib.Path(args.data)
    if not data_path.is_file():
        sys.exit(f"Data file not found: {data_path}")
    loader = DataLoader(
        path=str(data_path),
        history_length=args.history_length,
        required_auftraege_per_patient=args.required_auftraege_per_patient,
    )
    output_path = pathlib.Path(args.output)
    print("Generating dataset …")
    generate_dataset(
        loader,
        n_samples=args.samples,
        swap_ratio=args.swap_ratio,
        save_path=output_path,
    )
    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    main()

