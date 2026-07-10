#!/usr/bin/env python3
"""Aggregate strict multi-seed results and patient-bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def subject_from_path(path: str, dataset: str) -> str:
    stem = Path(path).stem.lower()
    if dataset == "kermany_grouped":
        if stem.startswith("person"):
            return stem.split("_", 1)[0]
        parts = stem.split("-")
        if stem.startswith("normal") and len(parts) >= 3:
            return "-".join(parts[:3])
        if stem.startswith("im-") and len(parts) >= 2:
            return "-".join(parts[:2])
    return stem


def metrics(y: list[int], score: list[float]) -> dict[str, float]:
    pred = [int(value >= 0.5) for value in score]
    negatives = [i for i, value in enumerate(y) if value == 0]
    specificity = sum(pred[i] == 0 for i in negatives) / len(negatives) if negatives else 0.0
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "brier_score": float(brier_score_loss(y, score)),
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_ci(y: list[int], score: list[float], subjects: list[str], iterations: int, seed: int) -> dict[str, list[float]]:
    by_subject: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(subjects):
        by_subject[item].append(index)
    unique = sorted(by_subject)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        selected = [rng.choice(unique) for _ in unique]
        indices = [index for item in selected for index in by_subject[item]]
        y_sample = [y[index] for index in indices]
        if len(set(y_sample)) < 2:
            continue
        current = metrics(y_sample, [score[index] for index in indices])
        for key, value in current.items():
            samples[key].append(value)
    return {key: [percentile(values, 0.025), percentile(values, 0.975)] for key, values in samples.items()}


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-json", type=Path, default=Path("results/strict_summary.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/strict_summary.csv"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = PROJECT_ROOT / args.results_dir
    models = ("densenet121", "convnext_tiny", "vit_b16")
    datasets = ("kermany_grouped", "rsna")
    rows_out: list[dict[str, Any]] = []
    payload: dict[str, Any] = {"bootstrap_iterations": args.bootstrap, "groups": []}
    for dataset in datasets:
        for model in models:
            seed_rows = []
            aligned: dict[str, list[float]] = defaultdict(list)
            labels: dict[str, int] = {}
            subjects: dict[str, str] = {}
            for seed in (42, 43, 44):
                name = f"strict_predictions_{'kermany_grouped_test' if dataset == 'kermany_grouped' else 'rsna_test'}_{model}_seed{seed}.csv"
                path = results / name
                if not path.is_file():
                    continue
                data = read_predictions(path)
                y = [int(row["true_label"] == "PNEUMONIA") for row in data]
                score = [float(row["prob_PNEUMONIA"]) for row in data]
                seed_rows.append({"seed": seed, **metrics(y, score)})
                for row in data:
                    key = row["path"]
                    aligned[key].append(float(row["prob_PNEUMONIA"]))
                    labels[key] = int(row["true_label"] == "PNEUMONIA")
                    subjects[key] = subject_from_path(key, dataset)
            if len(seed_rows) != 3:
                continue
            keys = sorted(aligned)
            y = [labels[key] for key in keys]
            ensemble_score = [statistics.mean(aligned[key]) for key in keys]
            ensemble = metrics(y, ensemble_score)
            ci = bootstrap_ci(y, ensemble_score, [subjects[key] for key in keys], args.bootstrap, 20260710)
            group = {"dataset": dataset, "model": model, "seed_results": seed_rows, "ensemble": ensemble, "ensemble_patient_bootstrap_95ci": ci}
            payload["groups"].append(group)
            for metric in ensemble:
                values = [row[metric] for row in seed_rows]
                rows_out.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "metric": metric,
                        "seed_mean": statistics.mean(values),
                        "seed_std": statistics.stdev(values),
                        "ensemble": ensemble[metric],
                        "ci_low": ci[metric][0],
                        "ci_high": ci[metric][1],
                    }
                )
    output_json = PROJECT_ROOT / args.output_json
    output_csv = PROJECT_ROOT / args.output_csv
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Wrote {len(payload['groups'])} result groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
