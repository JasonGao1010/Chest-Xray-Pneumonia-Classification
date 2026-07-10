"""Calibration helpers for binary chest X-ray prediction files."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CalibratedPrediction:
    path: str
    true_label: str
    predicted_label: str
    score: float


@dataclass(frozen=True)
class CalibrationBin:
    bin_index: int
    lower: float
    upper: float
    count: int
    accuracy: float
    confidence: float
    gap: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def clip_probability(value: float, eps: float = 1e-7) -> float:
    return min(max(float(value), eps), 1.0 - eps)


def load_binary_predictions(
    path: Path | str,
    positive_class: str = "PNEUMONIA",
) -> list[CalibratedPrediction]:
    csv_path = Path(path)
    score_column = f"prob_{positive_class}"
    rows: list[CalibratedPrediction] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "true_label", "predicted_label", score_column}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required prediction columns in {csv_path}: {sorted(missing)}")
        for row in reader:
            rows.append(
                CalibratedPrediction(
                    path=str(row["path"]),
                    true_label=str(row["true_label"]),
                    predicted_label=str(row["predicted_label"]),
                    score=clip_probability(float(row[score_column])),
                )
            )
    if not rows:
        raise ValueError(f"No prediction rows found in {csv_path}")
    return rows


def apply_temperature(score: float, temperature: float) -> float:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probability = clip_probability(score)
    logit = math.log(probability / (1.0 - probability))
    scaled_logit = logit / temperature
    if scaled_logit >= 0:
        z = math.exp(-scaled_logit)
        return 1.0 / (1.0 + z)
    z = math.exp(scaled_logit)
    return z / (1.0 + z)


def negative_log_likelihood(
    records: Sequence[CalibratedPrediction],
    temperature: float = 1.0,
    positive_class: str = "PNEUMONIA",
) -> float:
    total = 0.0
    for record in records:
        score = apply_temperature(record.score, temperature)
        label = 1.0 if record.true_label == positive_class else 0.0
        score = clip_probability(score)
        total += -(label * math.log(score) + (1.0 - label) * math.log(1.0 - score))
    return safe_divide(total, len(records))


def fit_temperature_grid(
    records: Sequence[CalibratedPrediction],
    positive_class: str = "PNEUMONIA",
    minimum: float = 0.5,
    maximum: float = 5.0,
    step: float = 0.01,
) -> dict[str, float]:
    if minimum <= 0 or maximum < minimum or step <= 0:
        raise ValueError("invalid temperature search range")
    best_temperature = minimum
    best_nll = float("inf")
    steps = int(round((maximum - minimum) / step))
    for index in range(steps + 1):
        temperature = round(minimum + index * step, 6)
        nll = negative_log_likelihood(
            records,
            temperature=temperature,
            positive_class=positive_class,
        )
        if nll < best_nll:
            best_temperature = temperature
            best_nll = nll
    return {"temperature": best_temperature, "nll": best_nll}


def calibration_bins(
    records: Sequence[CalibratedPrediction],
    *,
    n_bins: int = 10,
    threshold: float = 0.5,
    temperature: float = 1.0,
    positive_class: str = "PNEUMONIA",
) -> list[CalibrationBin]:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for record in records:
        score = apply_temperature(record.score, temperature)
        predicted_positive = score >= threshold
        true_positive = record.true_label == positive_class
        confidence = score if predicted_positive else 1.0 - score
        correct = int(predicted_positive == true_positive)
        index = min(int(confidence * n_bins), n_bins - 1)
        buckets[index].append((confidence, correct))

    result: list[CalibrationBin] = []
    for index, bucket in enumerate(buckets):
        lower = index / n_bins
        upper = (index + 1) / n_bins
        count = len(bucket)
        accuracy = safe_divide(sum(correct for _, correct in bucket), count)
        confidence = safe_divide(sum(confidence for confidence, _ in bucket), count)
        result.append(
            CalibrationBin(
                bin_index=index,
                lower=lower,
                upper=upper,
                count=count,
                accuracy=accuracy,
                confidence=confidence,
                gap=abs(accuracy - confidence) if count else 0.0,
            )
        )
    return result


def calibration_summary(
    records: Sequence[CalibratedPrediction],
    *,
    n_bins: int = 10,
    threshold: float = 0.5,
    temperature: float = 1.0,
    positive_class: str = "PNEUMONIA",
) -> dict[str, object]:
    bins = calibration_bins(
        records,
        n_bins=n_bins,
        threshold=threshold,
        temperature=temperature,
        positive_class=positive_class,
    )
    sample_count = len(records)
    brier_total = 0.0
    correct = 0
    for record in records:
        score = apply_temperature(record.score, temperature)
        label = 1.0 if record.true_label == positive_class else 0.0
        brier_total += (score - label) ** 2
        correct += int((score >= threshold) == bool(label))

    ece = sum(item.count * item.gap for item in bins) / sample_count
    mce = max((item.gap for item in bins), default=0.0)
    return {
        "sample_count": sample_count,
        "threshold": threshold,
        "temperature": temperature,
        "brier_score": safe_divide(brier_total, sample_count),
        "ece": ece,
        "mce": mce,
        "accuracy_at_threshold": safe_divide(correct, sample_count),
        "negative_log_likelihood": negative_log_likelihood(
            records,
            temperature=temperature,
            positive_class=positive_class,
        ),
        "bins": [item.to_dict() for item in bins],
    }
