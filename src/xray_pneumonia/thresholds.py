"""Threshold analysis helpers for binary chest X-ray predictions."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class PredictionRecord:
    """One scored binary-classification prediction."""

    path: str
    true_label: str
    score: float


@dataclass(frozen=True)
class ThresholdMetrics:
    """Metrics for one positive-class decision threshold."""

    threshold: float
    sample_count: int
    positive_count: int
    negative_count: int
    tn: int
    fp: int
    fn: int
    tp: int
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def load_prediction_csv(
    path: Path | str,
    positive_class: str = "PNEUMONIA",
    score_column: str | None = None,
) -> list[PredictionRecord]:
    """Load predictions written by ``scripts/evaluate.py``."""
    csv_path = Path(path)
    score_name = score_column or f"prob_{positive_class}"
    records: list[PredictionRecord] = []

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "true_label", score_name}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required prediction columns in {csv_path}: {sorted(missing)}")

        for row in reader:
            records.append(
                PredictionRecord(
                    path=str(row["path"]),
                    true_label=str(row["true_label"]),
                    score=float(row[score_name]),
                )
            )

    if not records:
        raise ValueError(f"No prediction rows found in {csv_path}")
    return records


def compute_threshold_metrics(
    records: Sequence[PredictionRecord],
    threshold: float,
    positive_class: str = "PNEUMONIA",
) -> ThresholdMetrics:
    """Compute confusion-matrix and scalar metrics for one threshold."""
    tp = tn = fp = fn = 0
    for record in records:
        is_positive = record.true_label == positive_class
        predicted_positive = record.score >= threshold
        if is_positive and predicted_positive:
            tp += 1
        elif is_positive:
            fn += 1
        elif predicted_positive:
            fp += 1
        else:
            tn += 1

    positive_count = tp + fn
    negative_count = tn + fp
    sample_count = positive_count + negative_count
    accuracy = safe_divide(tp + tn, sample_count)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, positive_count)
    specificity = safe_divide(tn, negative_count)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    balanced_accuracy = (recall + specificity) / 2

    return ThresholdMetrics(
        threshold=float(threshold),
        sample_count=sample_count,
        positive_count=positive_count,
        negative_count=negative_count,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
    )


def make_threshold_grid(step: float = 0.001) -> list[float]:
    """Return thresholds from 0.0 to 1.0 inclusive."""
    if step <= 0 or step > 1:
        raise ValueError("threshold step must be in the interval (0, 1]")
    count = int(round(1.0 / step))
    thresholds = {0.0, 1.0}
    for index in range(count + 1):
        thresholds.add(round(min(index * step, 1.0), 6))
    return sorted(thresholds)


def threshold_curve(
    records: Sequence[PredictionRecord],
    thresholds: Iterable[float],
    positive_class: str = "PNEUMONIA",
) -> list[ThresholdMetrics]:
    return [
        compute_threshold_metrics(records, threshold, positive_class=positive_class)
        for threshold in thresholds
    ]


def select_best(curve: Sequence[ThresholdMetrics], metric_name: str) -> ThresholdMetrics:
    """Select the best row for a metric, preferring recall and specificity on ties."""
    if not curve:
        raise ValueError("cannot select a threshold from an empty curve")
    return max(
        curve,
        key=lambda item: (
            float(getattr(item, metric_name)),
            item.recall,
            item.specificity,
            item.accuracy,
            item.threshold,
        ),
    )


def select_best_with_recall_floor(
    curve: Sequence[ThresholdMetrics],
    recall_floor: float,
) -> ThresholdMetrics | None:
    """Select the highest-specificity threshold that preserves a recall floor."""
    candidates = [item for item in curve if item.recall >= recall_floor]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item.specificity,
            item.accuracy,
            item.f1,
            item.threshold,
        ),
    )


def binary_roc_auc(
    records: Sequence[PredictionRecord],
    positive_class: str = "PNEUMONIA",
) -> float | None:
    """Compute binary ROC-AUC from scores using average ranks for ties."""
    labeled_scores = sorted(
        (record.score, record.true_label == positive_class) for record in records
    )
    positive_count = sum(1 for _, is_positive in labeled_scores if is_positive)
    negative_count = len(labeled_scores) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None

    rank_sum = 0.0
    index = 0
    rank = 1
    while index < len(labeled_scores):
        next_index = index + 1
        score = labeled_scores[index][0]
        while next_index < len(labeled_scores) and labeled_scores[next_index][0] == score:
            next_index += 1

        group_size = next_index - index
        average_rank = (rank + rank + group_size - 1) / 2
        for group_index in range(index, next_index):
            if labeled_scores[group_index][1]:
                rank_sum += average_rank

        rank += group_size
        index = next_index

    return safe_divide(
        rank_sum - positive_count * (positive_count + 1) / 2,
        positive_count * negative_count,
    )
