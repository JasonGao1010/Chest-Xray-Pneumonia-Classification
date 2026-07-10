#!/usr/bin/env python3
"""Audit split grouping integrity and prediction/result consistency.

The audit is intentionally read-only. It detects exact duplicate images,
dataset-specific grouping identifiers crossing data splits, label conflicts,
and stale evaluation JSON files. It does not reinterpret medical labels or
claim that filename tokens are verified patient identifiers.
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


def infer_dataset_kind(root: Path) -> str:
    """Infer the supported dataset family from a local directory name."""
    value = root.as_posix().lower()
    if "nih" in value:
        return "nih"
    if "rsna" in value:
        return "rsna"
    if "kermany" in value or root.name == "chest_xray":
        return "kermany"
    return "kermany"


def kermany_filename_token(path: Path, label: str, subtype_sensitive: bool = True) -> str:
    """Return a conservative filename-derived grouping token.

    Pneumonia ``personN`` counters are namespaced by the filename subtype when
    ``subtype_sensitive`` is true.  The public data package does not provide a
    subject table that proves equal bare counters across bacterial and viral
    filenames denote the same person.
    """
    stem = path.stem
    if label == "PNEUMONIA":
        match = re.match(r"(person\d+)_", stem, flags=re.IGNORECASE)
        token = (match.group(1) if match else stem).lower()
        subtype_match = re.search(r"_(bacteria|virus)_", stem, flags=re.IGNORECASE)
        subtype = subtype_match.group(1).lower() if subtype_match else "unknown"
        return f"{subtype}:{token}" if subtype_sensitive else token
    else:
        match = re.match(r"((?:normal\d+-)?im-\d+)(?:-|$)", stem, flags=re.IGNORECASE)
        return f"normal:{(match.group(1) if match else stem).lower()}"


def filename_group_id(
    path: Path,
    label: str,
    dataset_kind: str,
    *,
    kermany_subtype_sensitive: bool = True,
) -> str:
    """Extract a dataset-specific, explicitly non-universal grouping key."""
    if dataset_kind == "nih":
        match = re.match(r"(\d{8})(?:_|$)", path.stem)
        return f"nih_patient_id:{match.group(1) if match else path.stem.lower()}"
    if dataset_kind == "rsna":
        return f"rsna_patient_id:{path.stem.lower()}"
    return (
        "kermany_filename_cluster:"
        f"{kermany_filename_token(path, label, subtype_sensitive=kermany_subtype_sensitive)}"
    )


def subject_id(path: Path, label: str) -> str:
    """Backward-compatible alias for the bare Kermany filename token.

    New code should use :func:`filename_group_id` and name the result a group
    or filename cluster, not a verified subject/patient identifier.
    """
    token = kermany_filename_token(path, label, subtype_sensitive=False)
    return token.removeprefix("normal:")


def iter_images(root: Path) -> Iterable[tuple[str, str, Path]]:
    for split in ("train", "val", "test"):
        for label in ("NORMAL", "PNEUMONIA"):
            directory = root / split / label
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    yield split, label, path


def _kermany_token_sensitivity(
    rows: list[tuple[str, str, Path]],
) -> dict[str, Any]:
    by_split_subtype: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for split, label, path in rows:
        if label != "PNEUMONIA":
            continue
        stem = path.stem
        subtype_match = re.search(r"_(bacteria|virus)_", stem, flags=re.IGNORECASE)
        subtype = subtype_match.group(1).lower() if subtype_match else "unknown"
        by_split_subtype[split][subtype].add(
            kermany_filename_token(path, label, subtype_sensitive=False)
        )

    comparisons: list[dict[str, Any]] = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        same_subtype: list[dict[str, Any]] = []
        cross_subtype: list[dict[str, Any]] = []
        for left_subtype in sorted(by_split_subtype[left]):
            for right_subtype in sorted(by_split_subtype[right]):
                overlap = sorted(
                    by_split_subtype[left][left_subtype]
                    & by_split_subtype[right][right_subtype]
                )
                if not overlap:
                    continue
                item = {
                    "left_subtype": left_subtype,
                    "right_subtype": right_subtype,
                    "count": len(overlap),
                    "tokens": overlap,
                }
                if left_subtype == right_subtype:
                    same_subtype.append(item)
                else:
                    cross_subtype.append(item)
        comparisons.append(
            {
                "left": left,
                "right": right,
                "same_subtype": same_subtype,
                "same_subtype_count": sum(item["count"] for item in same_subtype),
                "cross_subtype": cross_subtype,
                "cross_subtype_count": sum(item["count"] for item in cross_subtype),
            }
        )
    return {
        "bare_person_token_semantics": (
            "Sensitivity diagnostic only: equal bare personN counters across "
            "bacteria/virus namespaces are not treated as verified patient identity."
        ),
        "comparisons": comparisons,
    }


def audit_dataset(
    root: Path,
    hash_images: bool,
    dataset_kind: str | None = None,
) -> dict[str, Any]:
    dataset_kind = dataset_kind or infer_dataset_kind(root)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    groups: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    rows = list(iter_images(root))
    group_labels: dict[str, set[str]] = defaultdict(set)
    for split, label, path in rows:
        counts[split][label] += 1
        group = filename_group_id(
            path,
            label,
            dataset_kind,
            # The rebuilt split deliberately treats equal bare personN tokens
            # as one conservative cluster. The subtype-aware alternative is
            # reported separately below and is not promoted to patient truth.
            kermany_subtype_sensitive=False,
        )
        groups[split].add(group)
        group_labels[group].add(label)
        if hash_images:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[digest].append((split, label, path.relative_to(root).as_posix()))

    group_overlap: list[dict[str, Any]] = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(groups[left] & groups[right])
        if overlap:
            group_overlap.append(
                {"left": left, "right": right, "count": len(overlap), "ids": overlap}
            )

    cross_split_duplicates = []
    label_conflicts = []
    for digest, hash_rows in hashes.items():
        if len({row[0] for row in hash_rows}) > 1:
            cross_split_duplicates.append({"sha256": digest, "files": hash_rows})
        if len({row[1] for row in hash_rows}) > 1:
            label_conflicts.append({"sha256": digest, "files": hash_rows})

    group_label_conflicts = [
        {"group_id": group, "labels": sorted(labels)}
        for group, labels in sorted(group_labels.items())
        if len(labels) > 1
    ]
    split_integrity_ok = not group_overlap and not cross_split_duplicates and not label_conflicts
    payload: dict[str, Any] = {
        "root": root.as_posix(),
        "dataset_kind": dataset_kind,
        "grouping_unit": {
            "kermany": (
                "conservative subtype-agnostic filename cluster sensitivity "
                "(not a verified patient ID)"
            ),
            "nih": "8-digit NIH Patient ID parsed from image filename",
            "rsna": "RSNA patientId/image filename stem",
        }[dataset_kind],
        "counts": {split: dict(value) for split, value in counts.items()},
        "group_counts": {split: len(values) for split, values in groups.items()},
        "group_overlap": group_overlap,
        "group_label_conflicts": group_label_conflicts,
        "cross_split_duplicate_groups": cross_split_duplicates,
        "label_conflict_groups": label_conflicts,
        "hash_images": hash_images,
        "group_disjoint_ok": not group_overlap,
        "split_integrity_ok": split_integrity_ok,
        # Retained for old consumers; the grouping unit above defines semantics.
        "independent_split_ok": split_integrity_ok,
    }
    if dataset_kind == "kermany":
        payload["bare_token_sensitivity"] = _kermany_token_sensitivity(rows)
    return payload


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
    evaluation_paths = sorted(results_dir.glob("*eval_*.json"))
    for json_path in evaluation_paths:
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
        "file_pattern": "*eval_*.json",
        "discovered_files": len(evaluation_paths),
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
    payload: dict[str, Any] = {
        "schema_version": 2,
        "datasets": [audit_dataset(resolve(path), not args.skip_image_hashes) for path in datasets if resolve(path).exists()],
        "evaluations": audit_evaluations(resolve(args.results_dir)),
    }
    payload["ok"] = payload["evaluations"]["ok"] and all(
        item["split_integrity_ok"] for item in payload["datasets"]
    )
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Integrity audit status: {'PASS' if payload['ok'] else 'FAIL'}")
    print(f"Wrote: {output.as_posix()}")
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
