#!/usr/bin/env python3
"""Create a patient-grouped Kermany split without copying image bytes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_integrity import IMAGE_SUFFIXES, subject_id


def resolve(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def collect(root: Path) -> list[dict[str, Any]]:
    rows = []
    for source_split in ("train", "val", "test"):
        for label in ("NORMAL", "PNEUMONIA"):
            directory = root / source_split / label
            for path in sorted(directory.glob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    rows.append(
                        {
                            "source": path,
                            "source_split": source_split,
                            "label": label,
                            "subject_id": subject_id(path, label),
                        }
                    )
    return rows


def assign_subjects(rows: list[dict[str, Any]], seed: int, train_fraction: float, val_fraction: float) -> dict[tuple[str, str], str]:
    by_label: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_label[row["label"]].add(row["subject_id"])
    rng = random.Random(seed)
    assignments: dict[tuple[str, str], str] = {}
    for label, subjects in sorted(by_label.items()):
        ordered = sorted(subjects)
        rng.shuffle(ordered)
        n_train = round(len(ordered) * train_fraction)
        n_val = round(len(ordered) * val_fraction)
        for index, item in enumerate(ordered):
            split = "train" if index < n_train else "val" if index < n_train + n_val else "test"
            assignments[(label, item)] = split
    return assignments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a patient-grouped Kermany dataset.")
    parser.add_argument("--source-root", type=Path, default=Path("data/raw/chest_xray"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/kermany_grouped_seed42"))
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/kermany_grouped_seed42.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/kermany_grouped_summary.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.train_fraction <= 0 or args.val_fraction <= 0 or args.train_fraction + args.val_fraction >= 1:
        raise ValueError("fractions must be positive and leave a non-empty test fraction")
    source_root, output_root = resolve(args.source_root), resolve(args.output_root)
    rows = collect(source_root)
    assignments = assign_subjects(rows, args.seed, args.train_fraction, args.val_fraction)
    if output_root.exists():
        shutil.rmtree(output_root)
    manifest_rows = []
    image_counts: dict[str, Counter[str]] = defaultdict(Counter)
    subject_sets: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        split = assignments[(row["label"], row["subject_id"])]
        destination = output_root / split / row["label"] / row["source"].name
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(row["source"], destination)
        digest = hashlib.sha256(row["source"].read_bytes()).hexdigest()
        image_counts[split][row["label"]] += 1
        subject_sets[split][row["label"]].add(row["subject_id"])
        manifest_rows.append(
            {
                "path": destination.relative_to(output_root).as_posix(),
                "true_label": row["label"],
                "subject_id": row["subject_id"],
                "split": split,
                "source_split": row["source_split"],
                "source_path": row["source"].relative_to(source_root).as_posix(),
                "sha256": digest,
            }
        )
    manifest = resolve(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "true_label", "subject_id", "split", "source_split", "source_path", "sha256"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(manifest_rows, key=lambda x: (x["split"], x["true_label"], x["path"])))
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    summary = {
        "ok": True,
        "seed": args.seed,
        "source_root": source_root.as_posix(),
        "output_root": output_root.as_posix(),
        "manifest": manifest.as_posix(),
        "manifest_sha256": manifest_hash,
        "image_counts": {split: dict(value) for split, value in image_counts.items()},
        "subject_counts": {split: {label: len(ids) for label, ids in by_label.items()} for split, by_label in subject_sets.items()},
        "total_images": len(manifest_rows),
        "link_mode": "hardlink",
    }
    summary_path = resolve(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
