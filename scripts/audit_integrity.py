#!/usr/bin/env python3
"""Audit split independence and prediction/result consistency.

The audit is intentionally read-only. It detects exact duplicate images,
subject identifiers crossing data splits, label conflicts, and stale evaluation
JSON files. It does not reinterpret medical labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resolve(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def subject_id(path: Path, label: str) -> str:
    """Extract the subject-like identifier encoded by Kermany filenames."""
    stem = path.stem
    if label == "PNEUMONIA":
        match = re.match(r"(person\d+)_", stem, flags=re.IGNORECASE)
    else:
        match = re.match(r"((?:normal\d+-)?im-\d+)(?:-|$)", stem, flags=re.IGNORECASE)
    return (match.group(1) if match else stem).lower()


def iter_images(root: Path) -> Iterable[tuple[str, str, Path]]:
    for split in ("train", "val", "test"):
        for label in ("NORMAL", "PNEUMONIA"):
            directory = root / split / label
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    yield split, label, path


def audit_dataset(root: Path, hash_images: bool) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    subjects: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    hashes: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for split, label, path in iter_images(root):
        counts[split][label] += 1
        subjects[label][split].add(subject_id(path, label))
        if hash_images:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[digest].append((split, label, path.relative_to(root).as_posix()))

    subject_overlap: list[dict[str, Any]] = []
    for label, by_split in subjects.items():
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = sorted(by_split[left] & by_split[right])
            if overlap:
                subject_overlap.append(
                    {"label": label, "left": left, "right": right, "count": len(overlap), "ids": overlap}
                )

    cross_split_duplicates = []
    label_conflicts = []
    for digest, rows in hashes.items():
        if len({row[0] for row in rows}) > 1:
            cross_split_duplicates.append({"sha256": digest, "files": rows})
        if len({row[1] for row in rows}) > 1:
            label_conflicts.append({"sha256": digest, "files": rows})

    return {
        "root": root.as_posix(),
        "counts": {split: dict(value) for split, value in counts.items()},
        "subject_overlap": subject_overlap,
        "cross_split_duplicate_groups": cross_split_duplicates,
        "label_conflict_groups": label_conflicts,
        "hash_images": hash_images,
        "independent_split_ok": not subject_overlap and not cross_split_duplicates and not label_conflicts,
    }


def recompute_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    y_true = [int(row["true_label"] == "PNEUMONIA") for row in rows]
    y_pred = [int(row["predicted_label"] == "PNEUMONIA") for row in rows]
    scores = [float(row["prob_PNEUMONIA"]) for row in rows]
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    tn, fp = matrix[0]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(set(y_true)) == 2 else None,
        "brier_score": float(brier_score_loss(y_true, scores)),
        "confusion_matrix": matrix,
    }


def audit_evaluations(results_dir: Path, tolerance: float = 2e-7) -> dict[str, Any]:
    checked = 0
    mismatches: list[dict[str, Any]] = []
    missing_predictions: list[dict[str, str]] = []
    for json_path in sorted(results_dir.glob("eval_*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        prediction_value = payload.get("predictions_output")
        if not prediction_value:
            continue
        prediction_path = resolve(prediction_value)
        if not prediction_path.is_file():
            missing_predictions.append({"evaluation": json_path.as_posix(), "predictions": prediction_path.as_posix()})
            continue
        with prediction_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        current = recompute_metrics(rows)
        stored = payload.get("metrics", {})
        checked += 1
        if len(rows) != payload.get("sample_count"):
            mismatches.append({"evaluation": json_path.as_posix(), "field": "sample_count", "stored": payload.get("sample_count"), "recomputed": len(rows)})
        for field, value in current.items():
            old = stored.get(field)
            # Older result files legitimately predate newly-added fields.
            if old is None:
                continue
            equal = old == value if field == "confusion_matrix" else abs(float(old) - float(value)) <= tolerance
            if not equal:
                mismatches.append({"evaluation": json_path.as_posix(), "field": field, "stored": old, "recomputed": value})
    return {
        "checked_files": checked,
        "missing_predictions": missing_predictions,
        "metric_mismatches": mismatches,
        "ok": not missing_predictions and not mismatches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit data splits and stored evaluation metrics.")
    parser.add_argument("--dataset", type=Path, action="append", default=[])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/integrity_audit.json"))
    parser.add_argument("--skip-image-hashes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = args.dataset or [Path("data/raw/chest_xray"), Path("data/processed/rsna_binary"), Path("data/processed/nih_binary")]
    payload = {
        "schema_version": 1,
        "datasets": [audit_dataset(resolve(path), not args.skip_image_hashes) for path in datasets if resolve(path).exists()],
        "evaluations": audit_evaluations(resolve(args.results_dir)),
    }
    payload["ok"] = payload["evaluations"]["ok"] and all(item["independent_split_ok"] for item in payload["datasets"])
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Integrity audit status: {'PASS' if payload['ok'] else 'FAIL'}")
    print(f"Wrote: {output.as_posix()}")
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
