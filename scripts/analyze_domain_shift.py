#!/usr/bin/env python3
"""Measure source separability with label-matched and label-stratified controls."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.audit_integrity import filename_group_id  # noqa: E402

FEATURE_NAMES = [
    "mean",
    "std",
    "p10",
    "median",
    "p90",
    "border_mean",
    "border_std",
    "center_mean",
    "center_std",
    "aspect_ratio",
    "entropy",
]
MODEL_FEATURE_INDICES = [
    index for index, name in enumerate(FEATURE_NAMES) if name != "aspect_ratio"
]


def image_features(path: Path) -> list[float]:
    with Image.open(path) as image:
        gray = np.asarray(
            image.convert("L").resize((128, 128)), dtype=np.float32
        ) / 255.0
        width, height = image.size
    border = np.concatenate(
        [gray[:8].ravel(), gray[-8:].ravel(), gray[:, :8].ravel(), gray[:, -8:].ravel()]
    )
    center = gray[32:96, 32:96]
    hist, _ = np.histogram(gray, bins=32, range=(0, 1), density=False)
    prob = hist / max(hist.sum(), 1)
    entropy = -sum(
        float(value) * math.log(float(value) + 1e-12) for value in prob
    )
    return [
        float(gray.mean()),
        float(gray.std()),
        float(np.quantile(gray, 0.1)),
        float(np.quantile(gray, 0.5)),
        float(np.quantile(gray, 0.9)),
        float(border.mean()),
        float(border.std()),
        float(center.mean()),
        float(center.std()),
        float(width / height),
        float(entropy),
    ]


def _evenly_spaced(items: list[Any], limit: int) -> list[Any]:
    """Deterministically retain coverage across a sorted sequence."""
    if limit >= len(items):
        return list(items)
    if limit <= 0:
        return []
    indices = np.linspace(0, len(items) - 1, num=limit, dtype=int)
    return [items[int(index)] for index in indices]


def collect(
    root: Path,
    source: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dataset_kind = "kermany" if source == "kermany" else "rsna"
    for split in ("train", "val", "test"):
        for label in ("NORMAL", "PNEUMONIA"):
            directory = root / split / label
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*")):
                if path.is_file():
                    rows.append(
                        {
                            "path": path,
                            "source": source,
                            "label": label,
                            "group": f"{source}:{filename_group_id(path, label, dataset_kind, kermany_subtype_sensitive=False)}",
                        }
                    )
    return _evenly_spaced(rows, limit)


def select_label_matched(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match source counts separately within NORMAL and PNEUMONIA strata."""
    selected: list[dict[str, Any]] = []
    for label in ("NORMAL", "PNEUMONIA"):
        by_source = {
            source: sorted(
                [
                    row
                    for row in rows
                    if row["source"] == source and row["label"] == label
                ],
                key=lambda row: row["path"].as_posix(),
            )
            for source in ("kermany", "rsna")
        }
        retained = min(len(by_source["kermany"]), len(by_source["rsna"]))
        for source in ("kermany", "rsna"):
            selected.extend(_evenly_spaced(by_source[source], retained))
    return sorted(
        selected,
        key=lambda row: (row["label"], row["source"], row["path"].as_posix()),
    )


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source = Counter(row["source"] for row in rows)
    source_label = Counter((row["source"], row["label"]) for row in rows)
    return {
        "total": len(rows),
        "by_source": {name: source[name] for name in ("kermany", "rsna")},
        "by_source_and_label": {
            name: {
                label: source_label[(name, label)]
                for label in ("NORMAL", "PNEUMONIA")
            }
            for name in ("kermany", "rsna")
        },
    }


def evaluate_source_classifier(
    rows: list[dict[str, Any]],
    feature_by_path: dict[Path, list[float]],
    random_state: int = 42,
) -> dict[str, Any]:
    """Run grouped five-fold source classification for a fixed row subset."""
    if not rows:
        raise ValueError("Source-classifier analysis received no rows")
    x = np.asarray([feature_by_path[row["path"]] for row in rows])
    y = np.asarray([int(row["source"] == "rsna") for row in rows])
    groups = np.asarray([row["group"] for row in rows])
    if len(set(y.tolist())) != 2:
        raise ValueError("Both sources are required for source classification")
    x_model = x[:, MODEL_FEATURE_INDICES]
    folds = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=random_state
    )
    predictions = np.zeros(len(rows), dtype=float)
    fold_results: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(folds.split(x_model, y, groups), 1):
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=fold,
            # A single worker keeps probability accumulation byte-reproducible;
            # parallel tree reduction can differ at the last floating bit.
            n_jobs=1,
        )
        model.fit(x_model[train], y[train])
        predictions[test] = model.predict_proba(x_model[test])[:, 1]
        fold_results.append(
            {
                "fold": fold,
                "sample_count": len(test),
                "accuracy": float(accuracy_score(y[test], predictions[test] >= 0.5)),
                "roc_auc": float(roc_auc_score(y[test], predictions[test])),
            }
        )
    return {
        "counts": _counts(rows),
        "group_count": len(set(groups.tolist())),
        "folds": fold_results,
        "overall_accuracy": float(accuracy_score(y, predictions >= 0.5)),
        "overall_roc_auc": float(roc_auc_score(y, predictions)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kermany", type=Path, default=Path("data/processed/kermany_grouped_seed42")
    )
    parser.add_argument(
        "--rsna", type=Path, default=Path("data/processed/rsna_binary")
    )
    parser.add_argument("--limit-per-source", type=int, default=1500)
    parser.add_argument(
        "--output", type=Path, default=Path("results/domain_shift_diagnostic.json")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kermany_root = args.kermany if args.kermany.is_absolute() else PROJECT_ROOT / args.kermany
    rsna_root = args.rsna if args.rsna.is_absolute() else PROJECT_ROOT / args.rsna
    rows = collect(kermany_root, "kermany", args.limit_per_source) + collect(
        rsna_root, "rsna", args.limit_per_source
    )
    feature_by_path = {
        row["path"]: image_features(row["path"])
        for row in rows
    }
    unadjusted = evaluate_source_classifier(rows, feature_by_path)
    label_matched_rows = select_label_matched(rows)
    label_matched = evaluate_source_classifier(label_matched_rows, feature_by_path)
    label_stratified = {
        label: evaluate_source_classifier(
            [row for row in rows if row["label"] == label], feature_by_path
        )
        for label in ("NORMAL", "PNEUMONIA")
    }

    x = np.asarray([feature_by_path[row["path"]] for row in rows])
    y = np.asarray([int(row["source"] == "rsna") for row in rows])
    payload = {
        "schema_version": 2,
        "sampling": {
            "method": "deterministic evenly spaced subsample over sorted split/label/path rows",
            "limit_per_source": args.limit_per_source,
        },
        "features": FEATURE_NAMES,
        "classifier_features": [FEATURE_NAMES[index] for index in MODEL_FEATURE_INDICES],
        "cross_validation": {
            "method": "5-fold StratifiedGroupKFold",
            "shuffle": True,
            "random_state": 42,
            "grouping": (
                "source-prefixed conservative subtype-agnostic filename clusters for Kermany and "
                "RSNA patientId filename stems for RSNA"
            ),
        },
        "analyses": {
            "unadjusted": unadjusted,
            "label_matched": {
                **label_matched,
                "matching": "equal source counts separately within each diagnosis label",
            },
            "label_stratified": label_stratified,
        },
        # Backward-compatible aliases refer only to the explicitly unadjusted analysis.
        "sample_count": unadjusted["counts"]["total"],
        "kermany_count": unadjusted["counts"]["by_source"]["kermany"],
        "rsna_count": unadjusted["counts"]["by_source"]["rsna"],
        "folds": unadjusted["folds"],
        "overall_accuracy": unadjusted["overall_accuracy"],
        "overall_roc_auc": unadjusted["overall_roc_auc"],
        "source_feature_means": {
            "kermany": dict(zip(FEATURE_NAMES, x[y == 0].mean(axis=0).tolist())),
            "rsna": dict(zip(FEATURE_NAMES, x[y == 1].mean(axis=0).tolist())),
        },
        "interpretation": (
            "Label-matched and within-label source classification quantify residual "
            "source separability after controlling diagnosis composition. They do not "
            "identify an acquisition mechanism or establish causality for pneumonia errors."
        ),
    }
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "unadjusted_roc_auc": unadjusted["overall_roc_auc"],
                "label_matched_roc_auc": label_matched["overall_roc_auc"],
                "normal_only_roc_auc": label_stratified["NORMAL"]["overall_roc_auc"],
                "pneumonia_only_roc_auc": label_stratified["PNEUMONIA"]["overall_roc_auc"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
