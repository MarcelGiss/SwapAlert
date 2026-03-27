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
import logging
from typing import Tuple

import numpy as np
# Parallel generation was replaced with a simple loop to allow a progress bar.
# from joblib import Parallel, delayed

# Local imports
from data_loader import DataLoader

# Set up module‑level logger. If the application configures logging elsewhere this
# call is a no‑op; otherwise it ensures that INFO‑level messages are printed.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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

    This implementation provides a tqdm progress bar to monitor the generation
    process. Logging is performed **before** the heavy work starts and **after**
    the dataset is optionally saved, but not for each individual sample.

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
    # Log the intent before heavy computation.
    logger.info(
        "Generating %d samples (swap_ratio=%.2f)", n_samples, swap_ratio
    )

    # Ensure we do not generate more samples than there are distinct patients.
    # The DataLoader stores a list of unique patient IDs in ``_patient_ids``.
    # If the requested ``n_samples`` exceeds this count we cap it to the number
    # of patients and log the adjustment. This guarantees that each generated
    # sample can correspond to a unique patient when possible.
    try:
        patient_count = len(loader._patient_ids)
    except Exception:
        # Fallback: if the attribute is unavailable, proceed without capping.
        patient_count = None

    if patient_count is not None and n_samples > patient_count:
        logger.info(
            "Requested %d samples exceeds number of unique patients (%d); "
            "capping to %d.",
            n_samples,
            patient_count,
            patient_count,
        )
        n_samples = patient_count

    n_swap = int(n_samples * swap_ratio)
    n_clean = n_samples - n_swap

    X: list[np.ndarray] = []
    y: list[int] = []

    # Use tqdm to display a single progress bar for the whole generation.
    from tqdm import tqdm

    with tqdm(total=n_samples, desc="Generating dataset", unit="sample") as pbar:
        for _ in range(n_clean):
            feat, label = _gen_one(loader, False)
            X.append(feat)
            y.append(label)
            pbar.update(1)
        for _ in range(n_swap):
            feat, label = _gen_one(loader, True)
            X.append(feat)
            y.append(label)
            pbar.update(1)

    X_arr = np.stack(X, axis=0)
    y_arr = np.array(y, dtype=int)

    if save_path is not None:
        # Ensure parent directory exists.
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Build column names in the format ``{ANALYTID}_{POSITION_IN_HISTORY}``.
        # ``loader.all_analyte`` holds the analyte identifiers in the order used
        # when flattening each sample. ``loader.history_length`` is the number of
        # historical columns per sample.
        col_names = [
            f"{analyt}_{pos}"
            for analyt in loader.all_analyte
            for pos in range(loader.history_length)
        ]
        import pandas as pd

        df = pd.DataFrame(X_arr, columns=col_names)
        df["label"] = y_arr
        df.to_csv(save_path, index=False)
        logger.info("Dataset saved to %s", save_path)

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
        default="../data/train_dataset_samp100k_hist20_min5_50swap.csv",
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

