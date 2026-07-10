#!/usr/bin/env python3
"""Validate an external binary dataset and optional metadata file."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xray_pneumonia.data import DEFAULT_CLASSES, validate_dataset_layout  # noqa: E402


def resolve_project_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_metadata_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    counts: dict[str, dict[str, int]] = {}
    rows = 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            split = str(row.get("split", ""))
            label = str(row.get("binary_label", row.get("true_label", "")))
            counts.setdefault(split, {}).setdefault(label, 0)
            counts[split][label] += 1
    return {"path": path.as_posix(), "rows": rows, "counts": counts}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check an external ImageFolder-style binary dataset.")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/rsna_binary"))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=Path("results/external_dataset_check.json"))
    parser.add_argument("--verify-images", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = resolve_project_path(args.data_root)
    metadata = resolve_project_path(args.metadata) if args.metadata is not None else data_root / "metadata.csv"
    validation = validate_dataset_layout(
        data_root,
        class_names=DEFAULT_CLASSES,
        splits=("train", "val", "test"),
        verify_images=args.verify_images,
    )
    metadata_summary = read_metadata_summary(metadata)
    report = {
        "ok": validation.ok,
        "dataset": validation.to_dict(),
        "metadata": metadata_summary,
    }
    output = resolve_project_path(args.json_output)
    write_json(output, report)
    print(f"External dataset status: {'OK' if validation.ok else 'BLOCKED'}")
    print(f"Wrote check report: {output.as_posix()}")
    return 0 if validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
