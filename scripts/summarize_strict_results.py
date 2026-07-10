#!/usr/bin/env python3
"""Aggregate strict results with explicit ensemble and grouped-bootstrap estimands."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from xray_pneumonia.protocol import Identity, artifact_name  # noqa: E402
ENSEMBLE_SEEDS = (42, 43, 44)
DECISION_THRESHOLD = 0.5
BOOTSTRAP_SEED = 20260710

PREDICTION_PATTERNS = {
    "strict": "strict_predictions_{dataset}_{model}_seed{seed}.csv",
    "robust": "robust_predictions_{dataset}_{model}_seed{seed}.csv",
    "mixed_simple": "mixed_simple_predictions_{dataset}_{model}_seed{seed}.csv",
    "mixed_domain_balanced": (
        "mixed_domain_balanced_predictions_{dataset}_{model}_seed{seed}.csv"
    ),
}
RECIPE_IDS = {"strict": "ERM", "robust": "ERM-Reg", "mixed_simple": "JT", "mixed_domain_balanced": "JT-DBS"}
MODEL_NAMES = {"densenet121": "DenseNet121", "convnext_tiny": "ConvNeXt-Tiny", "vit_b16": "ViT-B/16"}
DATASET_IDS = {"kermany_grouped": "Kermany-FG", "rsna": "RSNA-1707"}


def filename_group_from_path(path: str, dataset: str) -> str:
    """Return the documented cluster key used for grouped resampling."""
    stem = Path(path).stem.lower()
    if dataset == "kermany_grouped":
        person = re.match(r"(person\d+)_", stem)
        if person:
            # Use the same conservative unit as split construction and audit.
            # The public metadata do not prove equal bacterial/viral counters
            # are different patients, so bootstrap must keep them together.
            return f"kermany_filename_cluster:{person.group(1)}"
        normal = re.match(r"((?:normal\d+-)?im-\d+)(?:-|$)", stem)
        token = normal.group(1) if normal else stem
        return f"kermany_filename_cluster:normal:{token}"
    if dataset == "rsna":
        return f"rsna_patient_id:{stem}"
    raise ValueError(f"Unsupported dataset for grouped bootstrap: {dataset}")


def subject_from_path(path: str, dataset: str) -> str:
    """Compatibility alias; the return value is a grouping key, not a claim."""
    return filename_group_from_path(path, dataset)


def metrics(
    y: list[int],
    score: list[float],
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, float]:
    y_array = np.asarray(y, dtype=np.int8)
    score_array = np.asarray(score, dtype=float)
    pred = score_array >= threshold
    positive = y_array == 1
    negative = ~positive
    tp = int(np.sum(pred & positive))
    fp = int(np.sum(pred & negative))
    tn = int(np.sum(~pred & negative))
    fn = int(np.sum(~pred & positive))
    n_positive = tp + fn
    n_negative = tn + fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_positive if n_positive else 0.0
    specificity = tn / n_negative if n_negative else 0.0

    # Mann-Whitney form of AUROC with exact 0.5 credit for tied scores.
    ascending = np.argsort(score_array, kind="stable")
    sorted_score = score_array[ascending]
    sorted_y = y_array[ascending]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_score)) + 1]
    group_size = np.diff(np.r_[starts, len(sorted_score)])
    group_positive = np.add.reduceat(sorted_y, starts)
    group_negative = group_size - group_positive
    negative_before = np.cumsum(group_negative) - group_negative
    auc_numerator = np.sum(
        group_positive * (negative_before + 0.5 * group_negative)
    )
    roc_auc = float(auc_numerator / (n_positive * n_negative))

    # Average precision at the end of every tied-score block (sklearn's
    # non-interpolated definition).
    descending = np.argsort(-score_array, kind="stable")
    desc_score = score_array[descending]
    desc_y = y_array[descending]
    desc_starts = np.r_[0, np.flatnonzero(np.diff(desc_score)) + 1]
    desc_size = np.diff(np.r_[desc_starts, len(desc_score)])
    desc_positive = np.add.reduceat(desc_y, desc_starts)
    cumulative_positive = np.cumsum(desc_positive)
    cumulative_total = np.cumsum(desc_size)
    precision_at_threshold = cumulative_positive / cumulative_total
    auprc = float(np.sum((desc_positive / n_positive) * precision_at_threshold))

    return {
        "accuracy": float((tp + tn) / len(y_array)),
        "balanced_accuracy": float((recall + specificity) / 2),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(2 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0,
        "roc_auc": roc_auc,
        "auprc": auprc,
        "brier_score": float(np.mean((score_array - y_array) ** 2)),
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile from no bootstrap samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _group_index(groups: list[str]) -> dict[str, list[int]]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[group].append(index)
    if not by_group:
        raise ValueError("Grouped bootstrap requires at least one group")
    return by_group


def bootstrap_ci(
    y: list[int],
    score: list[float],
    groups: list[str],
    iterations: int,
    seed: int,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, list[float]]:
    """Percentile CI from resampling complete filename/patient groups."""
    by_group = _group_index(groups)
    unique = sorted(by_group)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        selected = [rng.choice(unique) for _ in unique]
        indices = [index for group in selected for index in by_group[group]]
        y_sample = [y[index] for index in indices]
        if len(set(y_sample)) < 2:
            continue
        current = metrics(
            y_sample,
            [score[index] for index in indices],
            threshold=threshold,
        )
        for key, value in current.items():
            samples[key].append(value)
    return {
        key: [percentile(values, 0.025), percentile(values, 0.975)]
        for key, values in samples.items()
    }


def paired_bootstrap_ci(
    y: list[int],
    baseline_score: list[float],
    candidate_score: list[float],
    groups: list[str],
    iterations: int,
    seed: int,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, list[float]]:
    """Paired percentile CI for candidate minus baseline metric differences."""
    by_group = _group_index(groups)
    unique = sorted(by_group)
    rng = random.Random(seed)
    differences: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        selected = [rng.choice(unique) for _ in unique]
        indices = [index for group in selected for index in by_group[group]]
        y_sample = [y[index] for index in indices]
        if len(set(y_sample)) < 2:
            continue
        baseline = metrics(
            y_sample,
            [baseline_score[index] for index in indices],
            threshold=threshold,
        )
        candidate = metrics(
            y_sample,
            [candidate_score[index] for index in indices],
            threshold=threshold,
        )
        for key in baseline:
            differences[key].append(candidate[key] - baseline[key])
    return {
        key: [percentile(values, 0.025), percentile(values, 0.975)]
        for key, values in differences.items()
    }


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _dataset_token(dataset: str) -> str:
    return "kermany_grouped_test" if dataset == "kermany_grouped" else "rsna_test"


def load_probability_ensemble(
    results_dir: Path,
    family: str,
    dataset: str,
    model: str,
) -> dict[str, Any]:
    """Load three aligned prediction files and average probabilities per image."""
    pattern = PREDICTION_PATTERNS[family]
    scores_by_path: dict[str, dict[int, float]] = defaultdict(dict)
    labels: dict[str, int] = {}
    seed_results: list[dict[str, Any]] = []
    files: list[str] = []
    for seed in ENSEMBLE_SEEDS:
        name = pattern.format(dataset=_dataset_token(dataset), model=model, seed=seed)
        path = results_dir / name
        canonical = results_dir / artifact_name(
            Identity(MODEL_NAMES[model], RECIPE_IDS[family], seed),
            DATASET_IDS[dataset], "test", "predictions", "csv",
        )
        if canonical.is_file():
            path = canonical
        if not path.is_file():
            raise FileNotFoundError(path)
        data = read_predictions(path)
        seen: set[str] = set()
        y_seed: list[int] = []
        score_seed: list[float] = []
        for row in data:
            key = row["path"]
            if key in seen:
                raise ValueError(f"Duplicate prediction path in {path}: {key}")
            seen.add(key)
            label = int(row["true_label"] == "PNEUMONIA")
            if key in labels and labels[key] != label:
                raise ValueError(f"Conflicting labels across seeds for {key}")
            labels[key] = label
            score = float(row["prob_PNEUMONIA"])
            scores_by_path[key][seed] = score
            y_seed.append(label)
            score_seed.append(score)
        seed_results.append({"seed": seed, **metrics(y_seed, score_seed)})
        files.append(path.relative_to(PROJECT_ROOT).as_posix())

    incomplete = {
        key: sorted(values)
        for key, values in scores_by_path.items()
        if set(values) != set(ENSEMBLE_SEEDS)
    }
    if incomplete:
        preview = list(incomplete.items())[:3]
        raise ValueError(f"Prediction paths are not aligned across seeds: {preview}")
    keys = sorted(scores_by_path)
    y = [labels[key] for key in keys]
    ensemble_score = [
        statistics.mean(scores_by_path[key][seed] for seed in ENSEMBLE_SEEDS)
        for key in keys
    ]
    return {
        "family": family,
        "dataset": dataset,
        "model": model,
        "paths": keys,
        "labels": y,
        "ensemble_score": ensemble_score,
        "seed_results": seed_results,
        "prediction_files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-json", type=Path, default=Path("results/strict_summary.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/strict_summary.csv"))
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless every published model, dataset, seed, and comparison family is present.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = args.results_dir if args.results_dir.is_absolute() else PROJECT_ROOT / args.results_dir
    models = ("densenet121", "convnext_tiny", "vit_b16")
    datasets = ("kermany_grouped", "rsna")
    rows_out: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "schema_version": 2,
        "analysis_specification": {
            "estimand": (
                "Per-image arithmetic mean of P(PNEUMONIA) from seeds 42, 43, and 44; "
                "metrics are computed once on the resulting probability ensemble."
            ),
            "ensemble_seeds": list(ENSEMBLE_SEEDS),
            "decision_threshold": DECISION_THRESHOLD,
            "bootstrap": {
                "method": "nonparametric percentile grouped bootstrap",
                "iterations": args.bootstrap,
                "confidence_level": 0.95,
                "quantile_interpolation": "linear",
                "seed": args.bootstrap_seed,
                "paired_comparisons": "candidate minus strict baseline on identical resampled groups",
            },
            "grouping_keys": {
                "kermany_grouped": (
                    "conservative subtype-agnostic filename cluster: equal personN "
                    "counters in bacterial and viral filenames remain grouped; NORMAL "
                    "uses the exam-like filename prefix"
                ),
                "rsna": "RSNA patientId, equal to the image filename stem",
            },
        },
        "groups": [],
        "paired_comparisons": [],
    }

    strict_ensembles: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in datasets:
        for model in models:
            try:
                loaded = load_probability_ensemble(results, "strict", dataset, model)
            except FileNotFoundError:
                if args.require_complete:
                    raise
                continue
            groups = [filename_group_from_path(path, dataset) for path in loaded["paths"]]
            ensemble = metrics(loaded["labels"], loaded["ensemble_score"])
            ci = bootstrap_ci(
                loaded["labels"],
                loaded["ensemble_score"],
                groups,
                args.bootstrap,
                args.bootstrap_seed,
            )
            group = {
                "dataset": DATASET_IDS[dataset],
                "model": MODEL_NAMES[model],
                "recipe": "ERM",
                "prediction_files": loaded["prediction_files"],
                "sample_count": len(loaded["paths"]),
                "bootstrap_group_count": len(set(groups)),
                "seed_results": loaded["seed_results"],
                "ensemble": ensemble,
                "ensemble_group_bootstrap_95ci": ci,
            }
            payload["groups"].append(group)
            strict_ensembles[(dataset, model)] = loaded
            for metric in ensemble:
                values = [row[metric] for row in loaded["seed_results"]]
                rows_out.append(
                    {
                        "dataset": DATASET_IDS[dataset],
                        "model": MODEL_NAMES[model],
                        "metric": metric,
                        "seed_mean": statistics.mean(values),
                        "seed_std": statistics.stdev(values),
                        "ensemble": ensemble[metric],
                        "ci_low": ci[metric][0],
                        "ci_high": ci[metric][1],
                    }
                )

    comparison_families = ("robust", "mixed_simple", "mixed_domain_balanced")
    for family in comparison_families:
        for dataset in datasets:
            baseline = strict_ensembles.get((dataset, "densenet121"))
            if baseline is None:
                continue
            try:
                candidate = load_probability_ensemble(
                    results, family, dataset, "densenet121"
                )
            except FileNotFoundError:
                if args.require_complete:
                    raise
                continue
            if candidate["paths"] != baseline["paths"]:
                raise ValueError(
                    f"{family}/{dataset} and strict baseline do not contain identical paths"
                )
            if candidate["labels"] != baseline["labels"]:
                raise ValueError(
                    f"{family}/{dataset} and strict baseline labels do not match"
                )
            groups = [
                filename_group_from_path(path, dataset) for path in baseline["paths"]
            ]
            baseline_metrics = metrics(baseline["labels"], baseline["ensemble_score"])
            candidate_metrics = metrics(candidate["labels"], candidate["ensemble_score"])
            differences = {
                key: candidate_metrics[key] - baseline_metrics[key]
                for key in baseline_metrics
            }
            difference_ci = paired_bootstrap_ci(
                baseline["labels"],
                baseline["ensemble_score"],
                candidate["ensemble_score"],
                groups,
                args.bootstrap,
                args.bootstrap_seed,
            )
            payload["paired_comparisons"].append(
                {
                    "candidate_recipe": RECIPE_IDS[family],
                    "baseline_recipe": "ERM",
                    "dataset": DATASET_IDS[dataset],
                    "model": "DenseNet121",
                    "sample_count": len(groups),
                    "bootstrap_group_count": len(set(groups)),
                    "baseline_ensemble": baseline_metrics,
                    "candidate_ensemble": candidate_metrics,
                    "candidate_minus_baseline": differences,
                    "paired_group_bootstrap_95ci": difference_ci,
                    "candidate_prediction_files": candidate["prediction_files"],
                }
            )

    if not rows_out:
        raise RuntimeError("No complete strict three-seed result groups were found")
    if args.require_complete:
        expected_groups = len(models) * len(datasets)
        expected_comparisons = len(comparison_families) * len(datasets)
        if len(payload["groups"]) != expected_groups:
            raise RuntimeError(
                f"Incomplete strict matrix: {len(payload['groups'])}/{expected_groups} groups"
            )
        if len(payload["paired_comparisons"]) != expected_comparisons:
            raise RuntimeError(
                "Incomplete comparison matrix: "
                f"{len(payload['paired_comparisons'])}/{expected_comparisons} comparisons"
            )
    output_json = args.output_json if args.output_json.is_absolute() else PROJECT_ROOT / args.output_json
    output_csv = args.output_csv if args.output_csv.is_absolute() else PROJECT_ROOT / args.output_csv
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows_out[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows_out)
    print(
        f"Wrote {len(payload['groups'])} strict groups and "
        f"{len(payload['paired_comparisons'])} paired comparisons"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
