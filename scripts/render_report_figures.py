#!/usr/bin/env python3
"""Regenerate selected public confusion-matrix figures from prediction CSVs."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.evaluate import write_confusion_matrix  # noqa: E402


FIGURES = (
    (
        "results/strict_predictions_kermany_grouped_test_densenet121_seed42.csv",
        "figures/strict_confusion_kermany_grouped_test_densenet121_seed42.png",
    ),
    (
        "results/strict_predictions_rsna_test_densenet121_seed42.csv",
        "figures/strict_confusion_rsna_test_densenet121_seed42.png",
    ),
    (
        "results/mixed_simple_predictions_rsna_test_densenet121_seed42.csv",
        "figures/mixed_simple_confusion_rsna_test_densenet121_seed42.png",
    ),
)


def render(predictions: Path, output: Path) -> None:
    with predictions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = ("NORMAL", "PNEUMONIA")
    matrix = confusion_matrix(
        [row["true_label"] for row in rows],
        [row["predicted_label"] for row in rows],
        labels=list(labels),
    )
    write_confusion_matrix(output, matrix.tolist(), labels)


def main() -> int:
    for source_name, output_name in FIGURES:
        source = PROJECT_ROOT / source_name
        output = PROJECT_ROOT / output_name
        render(source, output)
        print(f"Wrote {output.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
