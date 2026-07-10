#!/usr/bin/env python3
"""Plot class-conditional pneumonia-score distributions from prediction CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Pneumonia probability distribution")
    args = parser.parse_args()

    scores: dict[str, list[float]] = {"NORMAL": [], "PNEUMONIA": []}
    with args.predictions.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = row["true_label"]
            if label in scores:
                value = row.get("prob_PNEUMONIA", row.get("prob_pneumonia"))
                if value is None:
                    raise ValueError("Prediction CSV has no pneumonia probability column")
                scores[label].append(float(value))

    if not all(scores.values()):
        raise ValueError("Both NORMAL and PNEUMONIA samples are required")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 4.8))
    bins = [index / 20 for index in range(21)]
    axis.hist(scores["NORMAL"], bins=bins, alpha=0.65, label="True NORMAL")
    axis.hist(scores["PNEUMONIA"], bins=bins, alpha=0.65, label="True PNEUMONIA")
    axis.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="Threshold 0.5")
    axis.set(xlabel="Predicted probability of PNEUMONIA", ylabel="Sample count", title=args.title)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=220)
    plt.close(fig)
    print(f"Wrote probability distribution: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
