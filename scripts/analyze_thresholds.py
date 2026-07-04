#!/usr/bin/env python3
"""Analyze binary decision thresholds from saved prediction probabilities."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xray_pneumonia.thresholds import (  # noqa: E402
    ThresholdMetrics,
    binary_roc_auc,
    compute_threshold_metrics,
    load_prediction_csv,
    make_threshold_grid,
    select_best,
    select_best_with_recall_floor,
    threshold_curve,
)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def resolve_project_path(path: Path | str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep positive-class probability thresholds for a prediction CSV."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Prediction CSV from scripts/evaluate.py. Defaults to results/evaluation_latest.json.",
    )
    parser.add_argument(
        "--evaluation-summary",
        type=Path,
        default=Path("results/evaluation_latest.json"),
        help="Evaluation JSON used when --predictions is omitted.",
    )
    parser.add_argument("--positive-class", type=str, default="PNEUMONIA")
    parser.add_argument("--score-column", type=str, default=None)
    parser.add_argument("--threshold-step", type=float, default=0.001)
    parser.add_argument(
        "--recall-floors",
        type=float,
        nargs="*",
        default=[0.995, 0.99, 0.98, 0.95],
        help="Recall floors for high-sensitivity operating points.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--curve-output", type=Path, default=None)
    parser.add_argument("--figure-output", type=Path, default=None)
    parser.add_argument("--no-figure", action="store_true")
    return parser.parse_args()


def predictions_from_latest_summary(path: Path) -> Path:
    summary = load_json(path)
    predictions_output = summary.get("predictions_output")
    if not predictions_output:
        raise ValueError(f"Evaluation summary has no predictions_output: {path}")
    return Path(predictions_output)


def output_suffix(predictions_path: Path) -> str:
    stem = predictions_path.stem
    prefix = "predictions_"
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


def metric_row(name: str, metrics: ThresholdMetrics | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    row = metrics.to_dict()
    row["name"] = name
    return row


def write_curve_csv(path: Path, curve: list[ThresholdMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(curve[0].to_dict().keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(item.to_dict() for item in curve)


def write_threshold_plot(
    path: Path,
    curve: list[ThresholdMetrics],
    selected: ThresholdMetrics,
    positive_class: str,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = [item.threshold for item in curve]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, color in [
        ("accuracy", "#1f77b4"),
        ("precision", "#ff7f0e"),
        ("recall", "#2ca02c"),
        ("specificity", "#d62728"),
        ("f1", "#9467bd"),
    ]:
        ax.plot(
            thresholds,
            [getattr(item, name) for item in curve],
            label=name,
            linewidth=1.7,
            color=color,
        )
    ax.axvline(
        selected.threshold,
        color="#333333",
        linestyle="--",
        linewidth=1.2,
        label="best balanced",
    )
    ax.set_xlabel(f"{positive_class} probability threshold")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    predictions_path = (
        resolve_project_path(args.predictions)
        if args.predictions
        else predictions_from_latest_summary(resolve_project_path(args.evaluation_summary))
    )
    predictions_path = resolve_project_path(predictions_path)
    suffix = output_suffix(predictions_path)
    json_output = resolve_project_path(args.json_output or f"results/threshold_analysis_{suffix}.json")
    curve_output = resolve_project_path(args.curve_output or f"results/threshold_curve_{suffix}.csv")
    figure_output = None
    if not args.no_figure:
        figure_output = resolve_project_path(args.figure_output or f"figures/threshold_curve_{suffix}.png")

    records = load_prediction_csv(
        predictions_path,
        positive_class=args.positive_class,
        score_column=args.score_column,
    )
    thresholds = make_threshold_grid(args.threshold_step)
    curve = threshold_curve(records, thresholds, positive_class=args.positive_class)
    best_balanced = select_best(curve, "balanced_accuracy")
    best_accuracy = select_best(curve, "accuracy")
    best_f1 = select_best(curve, "f1")
    default_threshold = compute_threshold_metrics(records, 0.5, positive_class=args.positive_class)
    recall_floor_rows = {
        f"{floor:.3f}": metric_row(
            f"recall>={floor:.3f}",
            select_best_with_recall_floor(curve, floor),
        )
        for floor in args.recall_floors
    }

    write_curve_csv(curve_output, curve)
    if figure_output is not None:
        write_threshold_plot(
            figure_output,
            curve,
            selected=best_balanced,
            positive_class=args.positive_class,
        )

    report = {
        "ok": True,
        "created_at": timestamp(),
        "predictions": predictions_path.as_posix(),
        "positive_class": args.positive_class,
        "score_column": args.score_column or f"prob_{args.positive_class}",
        "sample_count": len(records),
        "positive_count": sum(1 for record in records if record.true_label == args.positive_class),
        "negative_count": sum(1 for record in records if record.true_label != args.positive_class),
        "threshold_step": args.threshold_step,
        "roc_auc": binary_roc_auc(records, positive_class=args.positive_class),
        "default_threshold": metric_row("default_0.500", default_threshold),
        "best_balanced_accuracy": metric_row("best_balanced_accuracy", best_balanced),
        "best_accuracy": metric_row("best_accuracy", best_accuracy),
        "best_f1": metric_row("best_f1", best_f1),
        "recall_floor_operating_points": recall_floor_rows,
        "curve_output": curve_output.as_posix(),
        "figure_output": figure_output.as_posix() if figure_output else None,
        "note": (
            "Thresholds computed on an evaluation split describe trade-offs for that split; "
            "use a held-out calibration split for unbiased threshold selection when possible."
        ),
    }
    write_json(json_output, report)
    report["json_output"] = json_output.as_posix()
    return report


def print_summary(report: dict[str, Any]) -> None:
    def fmt(row: dict[str, Any]) -> str:
        return (
            f"thr={row['threshold']:.3f} acc={row['accuracy']:.4f} "
            f"prec={row['precision']:.4f} rec={row['recall']:.4f} "
            f"spec={row['specificity']:.4f} f1={row['f1']:.4f} "
            f"cm=[[{row['tn']},{row['fp']}],[{row['fn']},{row['tp']}]]"
        )

    print(f"Predictions: {report['predictions']}")
    auc = report["roc_auc"]
    auc_text = f"{auc:.4f}" if auc is not None else "NA"
    print(f"Samples: {report['sample_count']}  ROC-AUC: {auc_text}")
    print(f"Default: {fmt(report['default_threshold'])}")
    print(f"Best balanced: {fmt(report['best_balanced_accuracy'])}")
    for name, row in report["recall_floor_operating_points"].items():
        if row is not None:
            print(f"Recall floor {name}: {fmt(row)}")
    print(f"JSON: {report['json_output']}")
    print(f"Curve: {report['curve_output']}")
    if report.get("figure_output"):
        print(f"Figure: {report['figure_output']}")


def main() -> int:
    args = parse_args()
    try:
        report = analyze(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
