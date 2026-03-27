"""Generate a performance bar chart for model training results.

This script scans the ``trainings`` directory for files that contain a
classification report (the typical ``sklearn`` output). For each distinct
model type it selects the best run – defined as the run with the highest
combined precision and recall for *both* classes (``swap`` and ``no_swap``).

It then creates a grouped bar chart showing precision and recall for the two
classes across all selected models. The resulting figure is saved as
``trainings_performance.png`` in the repository root.

Run the script from the repository root using the project's dedicated Python
interpreter (as specified in ``AGENTS.md``):

    ~/.pyenv/versions/3.14.0/envs/hackathon_wue26/bin/python src/generate_graphics.py

The script requires ``matplotlib`` which can be installed via ``pip`` if not
already present.
"""

from __future__ import annotations

import pathlib
import re
from collections import defaultdict
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np


def parse_report(file_path: pathlib.Path) -> Tuple[float, float, float, float]:
    """Extract precision and recall for ``no_swap`` and ``swap``.

    Returns a tuple ``(no_swap_precision, no_swap_recall, swap_precision,
    swap_recall)``. Raises ``ValueError`` if the expected lines cannot be
    located.
    """
    text = file_path.read_text()
    # Match lines that start with the class name followed by precision and recall
    pattern = re.compile(r"^\s*(no_swap|swap)\s+([0-9.]+)\s+([0-9.]+)", re.MULTILINE)
    matches = {m.group(1): (float(m.group(2)), float(m.group(3))) for m in pattern.finditer(text)}
    if "no_swap" not in matches or "swap" not in matches:
        raise ValueError(f"Could not find both class lines in {file_path.name}")
    no_swap_prec, no_swap_rec = matches["no_swap"]
    swap_prec, swap_rec = matches["swap"]
    return no_swap_prec, no_swap_rec, swap_prec, swap_rec


def model_name_from_file(file_name: str) -> str:
    """Derive a stable model identifier from a filename.

    Filenames follow ``<model>_<run_id>.txt`` with a numeric ``run_id``. This
    function strips the numeric suffix and the extension, returning the pure
    model name, e.g. ``random_forest``.
    """
    base = pathlib.Path(file_name).stem  # removes .txt
    return re.sub(r"_\d+$", "", base)


def select_best_runs(metrics: Dict[str, List[Tuple[Tuple[float, float, float, float], pathlib.Path]]]) -> Dict[str, Tuple[float, float, float, float]]:
    """Select the best run for each model based on summed metrics.

    The best run is the one with the highest sum of all four scores (precision
    and recall for both classes). This simple heuristic prefers overall
    balanced performance.
    """
    best: Dict[str, Tuple[float, float, float, float]] = {}
    for model, runs in metrics.items():
        best_run = max(runs, key=lambda item: sum(item[0]))
        best[model] = best_run[0]
    return best


def plot_metrics(best_metrics: Dict[str, Tuple[float, float, float, float]]) -> None:
    """Create and save a grouped bar chart of the metrics."""
    models = sorted(best_metrics.keys())
    no_swap_prec = [best_metrics[m][0] for m in models]
    no_swap_rec = [best_metrics[m][1] for m in models]
    swap_prec = [best_metrics[m][2] for m in models]
    swap_rec = [best_metrics[m][3] for m in models]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5 * width, no_swap_prec, width, label="no_swap precision", color="#4c72b0")
    ax.bar(x - 0.5 * width, no_swap_rec, width, label="no_swap recall", color="#55a868")
    ax.bar(x + 0.5 * width, swap_prec, width, label="swap precision", color="#c44e52")
    ax.bar(x + 1.5 * width, swap_rec, width, label="swap recall", color="#8172b2")

    ax.set_ylabel("Score")
    ax.set_title("Model performance (precision & recall)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    out_path = pathlib.Path("trainings_performance.png")
    fig.savefig(out_path)
    print(f"Saved performance chart to {out_path.resolve()}")


def main() -> None:
    trainings_dir = pathlib.Path("trainings")
    if not trainings_dir.is_dir():
        raise FileNotFoundError("The 'trainings' directory does not exist.")

    collected: Dict[str, List[Tuple[Tuple[float, float, float, float], pathlib.Path]]] = defaultdict(list)
    for file_path in trainings_dir.glob("*.txt"):
        model = model_name_from_file(file_path.name)
        try:
            metrics = parse_report(file_path)
        except ValueError as exc:
            print(f"Skipping {file_path.name}: {exc}")
            continue
        collected[model].append((metrics, file_path))

    best = select_best_runs(collected)
    plot_metrics(best)


if __name__ == "__main__":
    # The repository's AGENTS.md specifies that we must use the dedicated
    # Python interpreter for all commands.
    main()
