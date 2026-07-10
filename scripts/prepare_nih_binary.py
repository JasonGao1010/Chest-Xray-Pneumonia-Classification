#!/usr/bin/env python3
"""Prepare a small NIH ChestX-ray14 Pneumonia-vs-No-Finding subset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def zip_image_members(zip_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            basename = Path(name).name
            if (
                name.startswith("__MACOSX/")
                or basename.startswith("._")
                or not basename.lower().endswith(".png")
            ):
                continue
            result[basename] = name
    return result


def split_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = {"NORMAL": [], "PNEUMONIA": []}
    for row in rows:
        by_label[row["binary_label"]].append(row)

    result: list[dict[str, Any]] = []
    for label_rows in by_label.values():
        shuffled = list(label_rows)
        rng.shuffle(shuffled)
        total = len(shuffled)
        train_count = round(total * 0.70)
        val_count = round(total * 0.10)
        for index, row in enumerate(shuffled):
            if index < train_count:
                split = "train"
            elif index < train_count + val_count:
                split = "val"
            else:
                split = "test"
            copied = dict(row)
            copied["split"] = split
            result.append(copied)
    return sorted(result, key=lambda row: (row["split"], row["binary_label"], row["image_index"]))


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_index",
        "finding_labels",
        "binary_label",
        "split",
        "image_path",
        "source_zip",
        "source_member",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_sample_manifests(output_root: Path, rows: list[dict[str, Any]]) -> None:
    for split in ("train", "val", "test"):
        path = output_root / f"samples_{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "true_label"])
            writer.writeheader()
            for row in rows:
                if row["split"] == split:
                    writer.writerow({"path": row["image_path"], "true_label": row["binary_label"]})


def extract_rows(zip_path: Path, output_root: Path, rows: list[dict[str, Any]]) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        for row in rows:
            target = output_root / row["image_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(row["source_member"]) as source, target.open("wb") as dest:
                dest.write(source.read())


def write_distribution_figure(path: Path, counts: dict[str, dict[str, int]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val", "test"]
    normal = [counts.get(split, {}).get("NORMAL", 0) for split in splits]
    pneumonia = [counts.get(split, {}).get("PNEUMONIA", 0) for split in splits]
    positions = range(len(splits))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([x - 0.18 for x in positions], normal, width=0.36, label="No Finding")
    ax.bar([x + 0.18 for x in positions], pneumonia, width=0.36, label="Pneumonia")
    ax.set_xticks(list(positions), splits)
    ax.set_ylabel("Images")
    ax.set_title("NIH Binary Subset Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare NIH ChestX-ray14 binary subset.")
    parser.add_argument("--metadata-csv", type=Path, default=Path("data/raw/nih_chestxray14_Data_Entry_2017_v2020.csv"))
    parser.add_argument("--zip", type=Path, default=Path("data/raw/nih_chestxray14/images_001.zip"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/nih_binary"))
    parser.add_argument("--summary-output", type=Path, default=Path("results/nih_dataset_summary.json"))
    parser.add_argument("--figure-output", type=Path, default=Path("figures/nih_class_distribution.png"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-class", type=int, default=65)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_csv = resolve_project_path(args.metadata_csv)
    zip_path = resolve_project_path(args.zip)
    output_root = resolve_project_path(args.output_root)
    members = zip_image_members(zip_path)
    rows = load_metadata(metadata_csv)

    candidates: dict[str, list[dict[str, Any]]] = {"NORMAL": [], "PNEUMONIA": []}
    for row in rows:
        image_index = row["Image Index"]
        if image_index not in members:
            continue
        finding_labels = row["Finding Labels"]
        if finding_labels == "No Finding":
            binary_label = "NORMAL"
        elif "Pneumonia" in finding_labels.split("|"):
            binary_label = "PNEUMONIA"
        else:
            continue
        candidates[binary_label].append(
            {
                "image_index": image_index,
                "finding_labels": finding_labels,
                "binary_label": binary_label,
                "source_zip": zip_path.as_posix(),
                "source_member": members[image_index],
            }
        )

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    for label, label_rows in candidates.items():
        shuffled = list(label_rows)
        rng.shuffle(shuffled)
        selected.extend(shuffled[: args.limit_per_class])
    split_selected = split_rows(selected, args.seed)

    counts: dict[str, dict[str, int]] = {
        split: {"NORMAL": 0, "PNEUMONIA": 0} for split in ("train", "val", "test")
    }
    for row in split_selected:
        image_path = f"{row['split']}/{row['binary_label']}/{row['image_index']}"
        row["image_path"] = image_path
        counts[row["split"]][row["binary_label"]] += 1

    extract_rows(zip_path, output_root, split_selected)
    write_metadata(output_root / "metadata.csv", split_selected)
    write_sample_manifests(output_root, split_selected)
    write_distribution_figure(resolve_project_path(args.figure_output), counts)
    summary = {
        "ok": True,
        "source": "NIH ChestX-ray14 images_001.zip subset",
        "label_rule": "PNEUMONIA if Finding Labels contains Pneumonia; NORMAL if Finding Labels is exactly No Finding.",
        "weak_label_warning": "NIH ChestX-ray14 labels are report-mined weak labels and are not equivalent to Kermany or RSNA labels.",
        "metadata_csv": metadata_csv.as_posix(),
        "zip": zip_path.as_posix(),
        "output_root": output_root.as_posix(),
        "sample_count": len(split_selected),
        "counts": counts,
        "available_in_zip": {
            "NORMAL": len(candidates["NORMAL"]),
            "PNEUMONIA": len(candidates["PNEUMONIA"]),
        },
    }
    write_json(resolve_project_path(args.summary_output), summary)
    print(f"Prepared NIH binary subset: {len(split_selected)} images")
    print(json.dumps(counts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
