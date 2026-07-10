#!/usr/bin/env python3
"""Collect evaluation and calibration JSON files into report-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(metrics: dict[str, Any], name: str) -> Any:
    if name in metrics:
        return metrics[name]
    if name == "specificity":
        matrix = metrics.get("confusion_matrix")
        if matrix and len(matrix) >= 2:
            tn, fp = matrix[0]
            return tn / (tn + fp) if (tn + fp) else 0.0
    if name == "balanced_accuracy":
        recall = metric_value(metrics, "recall")
        specificity = metric_value(metrics, "specificity")
        return (float(recall) + float(specificity)) / 2
    return None


def parse_label(path: Path, payload: dict[str, Any]) -> tuple[str, str, str]:
    stem = path.stem
    parts = stem.split("_")
    known_datasets = {"kermany", "nih", "rsna"}
    known_splits = {"train", "val", "test"}
    split = str(payload.get("split", ""))
    if len(parts) > 1 and parts[1] in known_datasets:
        dataset = parts[1]
        if len(parts) > 2 and parts[2] in known_splits:
            split = parts[2]
            model_start = 3
        else:
            model_start = 2
        model = "_".join(parts[model_start:]) if len(parts) > model_start else stem
    elif len(parts) > 1 and parts[1] == "test":
        dataset = "kermany"
        model = "_".join(parts[3:]) if len(parts) > 3 else stem
    else:
        dataset = parts[1] if len(parts) > 1 else ""
        model = "_".join(parts[3:]) if len(parts) > 3 else stem
    return dataset, split, model


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment result JSON files.")
    parser.add_argument("--eval-json", type=Path, nargs="*", default=[])
    parser.add_argument("--calibration-json", type=Path, nargs="*", default=[])
    parser.add_argument("--eval-output", type=Path, default=Path("results/experiment_comparison.csv"))
    parser.add_argument("--calibration-output", type=Path, default=Path("results/calibration_comparison.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_rows: list[dict[str, Any]] = []
    for item in args.eval_json:
        path = resolve_project_path(item)
        payload = read_json(path)
        dataset, split, model = parse_label(path, payload)
        metrics = payload.get("metrics", {})
        eval_rows.append(
            {
                "file": path.as_posix(),
                "dataset": dataset,
                "split": split or payload.get("split", ""),
                "model": payload.get("model", model),
                "sample_count": payload.get("sample_count", ""),
                "accuracy": metric_value(metrics, "accuracy"),
                "precision": metric_value(metrics, "precision"),
                "recall": metric_value(metrics, "recall"),
                "specificity": metric_value(metrics, "specificity"),
                "f1": metric_value(metrics, "f1"),
                "roc_auc": metric_value(metrics, "roc_auc"),
                "balanced_accuracy": metric_value(metrics, "balanced_accuracy"),
                "false_positive_rate": metric_value(metrics, "false_positive_rate"),
                "false_negative_rate": metric_value(metrics, "false_negative_rate"),
                "brier_score": metric_value(metrics, "brier_score"),
                "ece": metric_value(metrics, "ece"),
            }
        )
    calibration_rows: list[dict[str, Any]] = []
    for item in args.calibration_json:
        path = resolve_project_path(item)
        payload = read_json(path)
        summary = payload.get("calibrated", {})
        calibration_rows.append(
            {
                "file": path.as_posix(),
                "predictions": payload.get("predictions", ""),
                "temperature_source": payload.get("temperature_source", ""),
                "temperature": summary.get("temperature", ""),
                "ece": summary.get("ece", ""),
                "mce": summary.get("mce", ""),
                "brier_score": summary.get("brier_score", ""),
                "accuracy_at_threshold": summary.get("accuracy_at_threshold", ""),
                "negative_log_likelihood": summary.get("negative_log_likelihood", ""),
            }
        )
    if eval_rows:
        write_csv(
            resolve_project_path(args.eval_output),
            eval_rows,
            list(eval_rows[0].keys()),
        )
    if calibration_rows:
        write_csv(
            resolve_project_path(args.calibration_output),
            calibration_rows,
            list(calibration_rows[0].keys()),
        )
    print(f"Collected {len(eval_rows)} evaluation rows and {len(calibration_rows)} calibration rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
