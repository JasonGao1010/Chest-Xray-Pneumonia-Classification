#!/usr/bin/env python3
"""Prepare RSNA Pneumonia Detection Challenge data as a binary image dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def resolve_project_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_label_rows(labels_csv: Path) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    with labels_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"patientId", "Target"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {labels_csv}: {sorted(missing)}")
        for row in reader:
            patient_id = str(row["patientId"])
            target = int(float(row["Target"]))
            record = labels.setdefault(
                patient_id,
                {
                    "patientId": patient_id,
                    "target": 0,
                    "box_count": 0,
                },
            )
            if target == 1:
                record["target"] = 1
                record["box_count"] += 1
    return labels


def read_class_info(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    mapping: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"patientId", "class"}.issubset(reader.fieldnames or []):
            return {}
        for row in reader:
            mapping[str(row["patientId"])] = str(row["class"])
    return mapping


def split_patient_ids(
    labels: dict[str, dict[str, Any]],
    *,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, str]:
    by_target: dict[int, list[str]] = {0: [], 1: []}
    for patient_id, row in labels.items():
        by_target[int(row["target"])].append(patient_id)
    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    for patient_ids in by_target.values():
        shuffled = sorted(patient_ids)
        rng.shuffle(shuffled)
        total = len(shuffled)
        train_count = round(total * train_fraction)
        val_count = round(total * val_fraction)
        for index, patient_id in enumerate(shuffled):
            if index < train_count:
                split = "train"
            elif index < train_count + val_count:
                split = "val"
            else:
                split = "test"
            assignments[patient_id] = split
    return assignments


def normalize_to_uint8(array: Any) -> Any:
    import numpy as np

    values = array.astype("float32")
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return np.zeros(values.shape, dtype="uint8")
    values = (values - minimum) / (maximum - minimum)
    return (values * 255).clip(0, 255).astype("uint8")


def convert_dicom_to_png(source: Path, destination: Path) -> None:
    import pydicom
    from PIL import Image

    dataset = pydicom.dcmread(source)
    pixels = normalize_to_uint8(dataset.pixel_array)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).convert("L").save(destination)


def copy_or_convert_image(source: Path, destination: Path) -> None:
    if source.suffix.lower() == ".dcm":
        convert_dicom_to_png(source, destination.with_suffix(".png"))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_image_index(images_dir: Path) -> dict[str, Path]:
    image_suffixes = {".dcm", ".png", ".jpg", ".jpeg"}
    index: dict[str, Path] = {}
    for source in images_dir.rglob("*"):
        if not source.is_file():
            continue
        if source.suffix.lower() not in image_suffixes:
            continue
        index.setdefault(source.stem, source)
    return index


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patientId",
        "source_path",
        "image_path",
        "split",
        "target",
        "binary_label",
        "rsna_class",
        "box_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_sample_manifests(output_root: Path, metadata_rows: list[dict[str, Any]]) -> None:
    for split in ("train", "val", "test"):
        path = output_root / f"samples_{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "true_label"])
            writer.writeheader()
            for row in metadata_rows:
                if row["split"] == split:
                    writer.writerow({"path": row["image_path"], "true_label": row["binary_label"]})


def write_distribution_figure(path: Path, counts: dict[str, dict[str, int]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val", "test"]
    normal = [counts.get(split, {}).get("NORMAL", 0) for split in splits]
    pneumonia = [counts.get(split, {}).get("PNEUMONIA", 0) for split in splits]
    x_positions = range(len(splits))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([x - 0.18 for x in x_positions], normal, width=0.36, label="NORMAL")
    ax.bar([x + 0.18 for x in x_positions], pneumonia, width=0.36, label="PNEUMONIA")
    ax.set_xticks(list(x_positions), splits)
    ax.set_ylabel("Images")
    ax.set_title("RSNA Binary Class Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RSNA challenge files into a binary dataset.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/rsna_pneumonia"))
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--class-info-csv", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/rsna_binary"))
    parser.add_argument("--splits-output", type=Path, default=Path("data/splits/rsna_binary_seed42.json"))
    parser.add_argument("--summary-output", type=Path, default=Path("results/rsna_dataset_summary.json"))
    parser.add_argument("--figure-output", type=Path, default=Path("figures/rsna_class_distribution.png"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="Prepare only patients with images present under --images-dir.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing output paths before rebuilding. Without this flag, existing outputs are never overwritten.",
    )
    return parser.parse_args()


def prepare_output_paths(paths: list[Path], *, force: bool) -> None:
    """Refuse implicit replacement and remove outputs only after explicit consent."""
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing and not force:
        rendered = "\n  - ".join(path.as_posix() for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing RSNA outputs. "
            "Choose new paths or rerun with --force:\n  - " + rendered
        )
    for path in existing:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def main() -> int:
    args = parse_args()
    if args.train_fraction <= 0 or args.val_fraction <= 0 or args.train_fraction + args.val_fraction >= 1:
        raise ValueError("fractions must be positive and leave a non-empty test fraction")
    raw_root = resolve_project_path(args.raw_root)
    images_dir = resolve_project_path(args.images_dir or raw_root / "stage_2_train_images")
    labels_csv = resolve_project_path(args.labels_csv or raw_root / "stage_2_train_labels.csv")
    class_info_csv = resolve_project_path(args.class_info_csv or raw_root / "stage_2_detailed_class_info.csv")
    output_root = resolve_project_path(args.output_root)
    splits_output = resolve_project_path(args.splits_output)
    summary_output = resolve_project_path(args.summary_output)
    figure_output = resolve_project_path(args.figure_output)

    if not labels_csv.is_file():
        raise FileNotFoundError(f"RSNA labels CSV not found: {labels_csv}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"RSNA image directory not found: {images_dir}")
    resolved_output = output_root.resolve()
    protected_roots = (raw_root.resolve(), images_dir.resolve())
    if resolved_output == PROJECT_ROOT.resolve() or any(
        resolved_output.is_relative_to(root) or root.is_relative_to(resolved_output)
        for root in protected_roots
    ):
        raise ValueError("--output-root must not overlap the project, raw-data, or image roots")
    if not args.dry_run:
        prepare_output_paths(
            [output_root, splits_output, summary_output, figure_output],
            force=args.force,
        )

    labels = read_label_rows(labels_csv)
    full_label_patient_count = len(labels)
    class_info = read_class_info(class_info_csv)
    assignments = split_patient_ids(
        labels,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    image_index = build_image_index(images_dir)
    if args.available_only:
        labels = {patient_id: row for patient_id, row in labels.items() if patient_id in image_index}

    class_limits = {"NORMAL": 0, "PNEUMONIA": 0}
    metadata_rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {
        split: {"NORMAL": 0, "PNEUMONIA": 0} for split in ("train", "val", "test")
    }
    missing_images: list[str] = []
    for patient_id, row in sorted(labels.items()):
        binary_label = "PNEUMONIA" if int(row["target"]) == 1 else "NORMAL"
        if args.limit_per_class is not None and class_limits[binary_label] >= args.limit_per_class:
            continue
        source = image_index.get(patient_id, images_dir / f"{patient_id}.dcm")
        if not source.exists():
            missing_images.append(patient_id)
            continue

        split = assignments[patient_id]
        destination = output_root / split / binary_label / f"{patient_id}.png"
        relative_image_path = destination.relative_to(output_root).as_posix()
        metadata_rows.append(
            {
                "patientId": patient_id,
                "source_path": source.as_posix(),
                "image_path": relative_image_path,
                "split": split,
                "target": int(row["target"]),
                "binary_label": binary_label,
                "rsna_class": class_info.get(patient_id, ""),
                "box_count": int(row["box_count"]),
            }
        )
        counts[split][binary_label] += 1
        class_limits[binary_label] += 1
        if not args.dry_run:
            copy_or_convert_image(source, destination)

    if not args.dry_run:
        write_metadata(output_root / "metadata.csv", metadata_rows)
        write_sample_manifests(output_root, metadata_rows)
        write_distribution_figure(figure_output, counts)

    split_payload = {
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "test_fraction": 1.0 - args.train_fraction - args.val_fraction,
        "assignments": assignments,
    }
    summary = {
        "ok": not missing_images,
        "raw_root": raw_root.as_posix(),
        "images_dir": images_dir.as_posix(),
        "labels_csv": labels_csv.as_posix(),
        "output_root": output_root.as_posix(),
        "metadata": (output_root / "metadata.csv").as_posix(),
        "sample_count": len(metadata_rows),
        "full_label_patient_count": full_label_patient_count,
        "label_patient_count": len(labels),
        "available_image_count": len(image_index),
        "available_only": bool(args.available_only),
        "counts": counts,
        "missing_images": missing_images,
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        write_json(splits_output, split_payload)
        write_json(summary_output, summary)
    print(f"Prepared RSNA binary metadata rows: {len(metadata_rows)}")
    if args.dry_run:
        print("Dry run: no output files were written or removed")
    else:
        print(f"Wrote summary: {summary_output.as_posix()}")
    return 0 if not missing_images else 1


if __name__ == "__main__":
    raise SystemExit(main())
