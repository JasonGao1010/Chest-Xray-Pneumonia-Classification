#!/usr/bin/env python3
"""Quantify simple acquisition/style differences between Kermany and RSNA."""

from __future__ import annotations

import argparse
import json
import math
import sys
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
from scripts.audit_integrity import subject_id


def image_features(path: Path) -> list[float]:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((128, 128)), dtype=np.float32) / 255.0
        width, height = image.size
    border = np.concatenate([gray[:8].ravel(), gray[-8:].ravel(), gray[:, :8].ravel(), gray[:, -8:].ravel()])
    center = gray[32:96, 32:96]
    hist, _ = np.histogram(gray, bins=32, range=(0, 1), density=False)
    prob = hist / max(hist.sum(), 1)
    entropy = -sum(float(value) * math.log(float(value) + 1e-12) for value in prob)
    return [
        float(gray.mean()), float(gray.std()), float(np.quantile(gray, 0.1)),
        float(np.quantile(gray, 0.5)), float(np.quantile(gray, 0.9)),
        float(border.mean()), float(border.std()), float(center.mean()),
        float(center.std()), float(width / height), float(entropy),
    ]


def collect(root: Path, source: str, limit: int) -> list[tuple[Path, str, str]]:
    rows = []
    for split in ("train", "val", "test"):
        for label in ("NORMAL", "PNEUMONIA"):
            for path in sorted((root / split / label).glob("*")):
                if path.is_file():
                    rows.append((path, source, f"{source}:{label}:{subject_id(path, label)}"))
    # deterministic class/source subsampling keeps the diagnostic inexpensive
    step = max(1, len(rows) // limit)
    return rows[::step][:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kermany", type=Path, default=Path("data/processed/kermany_grouped_seed42"))
    parser.add_argument("--rsna", type=Path, default=Path("data/processed/rsna_binary"))
    parser.add_argument("--limit-per-source", type=int, default=1500)
    parser.add_argument("--output", type=Path, default=Path("results/domain_shift_diagnostic.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = collect(PROJECT_ROOT / args.kermany, "kermany", args.limit_per_source) + collect(PROJECT_ROOT / args.rsna, "rsna", args.limit_per_source)
    x = np.asarray([image_features(path) for path, _, _ in rows])
    y = np.asarray([int(source == "rsna") for _, source, _ in rows])
    groups = np.asarray([group for _, _, group in rows])
    names = ["mean", "std", "p10", "median", "p90", "border_mean", "border_std", "center_mean", "center_std", "aspect_ratio", "entropy"]
    # Aspect ratio alone separates the processed sources and the classifier
    # later receives square-resized inputs. Exclude it from the diagnostic
    # model so performance reflects pixel statistics rather than file geometry.
    model_indices = [index for index, name in enumerate(names) if name != "aspect_ratio"]
    x_model = x[:, model_indices]
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    predictions = np.zeros(len(rows), dtype=float)
    fold_results: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(folds.split(x_model, y, groups), 1):
        model = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=fold, n_jobs=-1)
        model.fit(x_model[train], y[train])
        predictions[test] = model.predict_proba(x_model[test])[:, 1]
        fold_results.append({"fold": fold, "accuracy": accuracy_score(y[test], predictions[test] >= 0.5), "roc_auc": roc_auc_score(y[test], predictions[test])})
    payload = {
        "sample_count": len(rows),
        "kermany_count": int((y == 0).sum()),
        "rsna_count": int((y == 1).sum()),
        "features": names,
        "classifier_features": [names[index] for index in model_indices],
        "folds": fold_results,
        "overall_accuracy": float(accuracy_score(y, predictions >= 0.5)),
        "overall_roc_auc": float(roc_auc_score(y, predictions)),
        "source_feature_means": {
            "kermany": dict(zip(names, x[y == 0].mean(axis=0).tolist())),
            "rsna": dict(zip(names, x[y == 1].mean(axis=0).tolist())),
        },
        "interpretation": "High source-classification performance indicates measurable acquisition/style differences; it does not identify a causal mechanism for pneumonia errors.",
    }
    output = PROJECT_ROOT / args.output
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"accuracy": payload["overall_accuracy"], "roc_auc": payload["overall_roc_auc"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
