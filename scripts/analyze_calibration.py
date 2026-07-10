#!/usr/bin/env python3
"""Compute Brier score, ECE, and reliability diagrams from prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xray_pneumonia.calibration import (  # noqa: E402
    calibration_summary,
    fit_temperature_grid,
    load_binary_predictions,
)


def resolve_project_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_bins_csv(path: Path, bins: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["bin_index", "lower", "upper", "count", "accuracy", "confidence", "gap"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bins)


def write_reliability_figure(path: Path, bins: list[dict[str, object]], title: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    centers = [(float(row["lower"]) + float(row["upper"])) / 2 for row in bins]
    accuracies = [float(row["accuracy"]) for row in bins]
    confidences = [float(row["confidence"]) for row in bins]
    counts = [int(row["count"]) for row in bins]

    fig, axes = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1, label="Perfect calibration")
    axes[0].bar(centers, accuracies, width=0.08, color="#4C78A8", alpha=0.75, label="Accuracy")
    axes[0].scatter(centers, confidences, color="#F58518", s=24, label="Confidence")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy / confidence")
    axes[0].set_title(title)
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].bar(centers, counts, width=0.08, color="#54A24B", alpha=0.8)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Confidence bin")
    axes[1].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze probability calibration.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--calibration-predictions",
        type=Path,
        default=None,
        help="Optional validation/calibration prediction CSV used to fit temperature.",
    )
    parser.add_argument("--positive-class", type=str, default="PNEUMONIA")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--bins-output", type=Path, default=None)
    parser.add_argument("--figure-output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions_path = resolve_project_path(args.predictions)
    records = load_binary_predictions(predictions_path, positive_class=args.positive_class)

    temperature_source = "fixed"
    temperature = 1.0 if args.temperature is None else float(args.temperature)
    fit_report: dict[str, object] | None = None
    if args.calibration_predictions is not None:
        calibration_path = resolve_project_path(args.calibration_predictions)
        calibration_records = load_binary_predictions(
            calibration_path,
            positive_class=args.positive_class,
        )
        fit_report = fit_temperature_grid(
            calibration_records,
            positive_class=args.positive_class,
        )
        temperature = float(fit_report["temperature"])
        fit_report["calibration_predictions"] = calibration_path.as_posix()
        temperature_source = "calibration_predictions"

    before = calibration_summary(
        records,
        n_bins=args.bins,
        threshold=args.threshold,
        temperature=1.0,
        positive_class=args.positive_class,
    )
    after = calibration_summary(
        records,
        n_bins=args.bins,
        threshold=args.threshold,
        temperature=temperature,
        positive_class=args.positive_class,
    )
    payload: dict[str, object] = {
        "ok": True,
        "predictions": predictions_path.as_posix(),
        "positive_class": args.positive_class,
        "temperature_source": temperature_source,
        "temperature_fit": fit_report,
        "uncalibrated": before,
        "calibrated": after,
        "note": (
            "Temperature must be fit on validation/calibration predictions. "
            "Do not fit it on the same test predictions used for final reporting."
        ),
    }

    json_output = resolve_project_path(args.json_output)
    write_json(json_output, payload)
    if args.bins_output is not None:
        write_bins_csv(resolve_project_path(args.bins_output), after["bins"])  # type: ignore[arg-type]
    if args.figure_output is not None:
        write_reliability_figure(
            resolve_project_path(args.figure_output),
            after["bins"],  # type: ignore[arg-type]
            title=f"Reliability Diagram ({predictions_path.stem})",
        )
    print(
        "Calibration "
        f"ECE={after['ece']:.4f} "
        f"Brier={after['brier_score']:.4f} "
        f"T={temperature:.3f}"
    )
    print(f"Wrote calibration report: {json_output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
