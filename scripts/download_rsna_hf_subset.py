#!/usr/bin/env python3
"""Download a deterministic balanced RSNA subset from a Hugging Face mirror."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HF_BASE_URL = "https://huggingface.co/datasets/Baldezo313/rsna-pneumonia-dataset/resolve/main"
IMAGE_DIRS = ("stage_2_train_images_0", "stage_2_train_images_1", "stage_2_train_images_2")


def resolve_project_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_label_rows(labels_csv: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with labels_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"patientId", "Target"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {labels_csv}: {sorted(missing)}")
        for row in reader:
            patient_id = str(row["patientId"])
            target = int(float(row["Target"]))
            labels[patient_id] = max(labels.get(patient_id, 0), target)
    return labels


def split_patient_ids(
    labels: dict[str, int],
    *,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, str]:
    by_target: dict[int, list[str]] = {0: [], 1: []}
    for patient_id, target in labels.items():
        by_target[int(target)].append(patient_id)
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


def choose_subset(labels: dict[str, int], assignments: dict[str, str], per_split_class: dict[str, int]) -> list[dict[str, str]]:
    candidate_groups: dict[tuple[str, int], list[str]] = {}
    for split, limit in per_split_class.items():
        for target in (0, 1):
            candidate_groups[(split, target)] = [
                patient_id
                for patient_id, row_target in sorted(labels.items())
                if int(row_target) == target and assignments[patient_id] == split
            ][:limit]

    selected: list[dict[str, str]] = []
    labels_by_target = {0: "NORMAL", 1: "PNEUMONIA"}
    max_limit = max(per_split_class.values(), default=0)
    for index in range(max_limit):
        for split in per_split_class:
            for target in (0, 1):
                group = candidate_groups[(split, target)]
                if index < len(group):
                    selected.append(
                        {
                            "patientId": group[index],
                            "split": split,
                            "binary_label": labels_by_target[target],
                        }
                    )
    return selected


def existing_source(raw_root: Path, patient_id: str) -> Path | None:
    for image_dir in IMAGE_DIRS:
        source = raw_root / image_dir / f"{patient_id}.dcm"
        if source.is_file() and source.stat().st_size > 1024:
            return source
    return None


def curl_download(url: str, destination: Path, *, retries: int, timeout: int) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()
    command = [
        "curl",
        "-L",
        "--retry",
        str(retries),
        "--retry-all-errors",
        "--connect-timeout",
        "15",
        "--max-time",
        str(timeout),
        "-s",
        "-w",
        "%{http_code}",
        "-o",
        str(partial),
        url,
    ]
    result = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    status = result.stdout.strip()[-3:]
    if result.returncode == 0 and status == "200" and partial.exists() and partial.stat().st_size > 1024:
        partial.replace(destination)
        return True
    if partial.exists():
        partial.unlink()
    return False


def download_patient(raw_root: Path, patient_id: str, *, retries: int, timeout: int) -> tuple[Path | None, list[str]]:
    current = existing_source(raw_root, patient_id)
    if current is not None:
        return current, []
    tried: list[str] = []
    quoted = quote(patient_id)
    for image_dir in IMAGE_DIRS:
        destination = raw_root / image_dir / f"{patient_id}.dcm"
        url = f"{HF_BASE_URL}/{image_dir}/{quoted}.dcm"
        tried.append(url)
        if curl_download(url, destination, retries=retries, timeout=timeout):
            return destination, tried
    return None, tried


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a balanced RSNA HF subset.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/rsna_pneumonia"))
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=Path("results/rsna_hf_subset_download.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-class", type=int, default=40)
    parser.add_argument("--val-per-class", type=int, default=8)
    parser.add_argument("--test-per-class", type=int, default=20)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
        help="Dataset splits to download candidates for.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--jobs", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_root = resolve_project_path(args.raw_root)
    labels_csv = resolve_project_path(args.labels_csv or raw_root / "stage_2_train_labels.csv")
    if not labels_csv.is_file():
        raise FileNotFoundError(f"RSNA labels CSV not found: {labels_csv}")

    labels = read_label_rows(labels_csv)
    assignments = split_patient_ids(
        labels,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    requested_limits = {
        "train": args.train_per_class,
        "val": args.val_per_class,
        "test": args.test_per_class,
    }
    per_split_class = {split: requested_limits[split] for split in args.splits}
    selected = choose_subset(labels, assignments, per_split_class)

    downloaded: list[dict[str, str]] = []
    failed: list[dict[str, Any]] = []

    def fetch(index_and_row: tuple[int, dict[str, str]]) -> tuple[int, dict[str, str], Path | None, list[str]]:
        index, row = index_and_row
        patient_id = row["patientId"]
        source, tried = download_patient(raw_root, patient_id, retries=args.retries, timeout=args.timeout)
        return index, row, source, tried

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [executor.submit(fetch, item) for item in enumerate(selected, start=1)]
        for future in as_completed(futures):
            index, row, source, tried = future.result()
            patient_id = row["patientId"]
            if source is None:
                failed.append({**row, "tried": tried})
                print(f"[{index}/{len(selected)}] missing {patient_id}", flush=True)
                continue
            downloaded.append({**row, "source_path": source.relative_to(raw_root).as_posix()})
            print(f"[{index}/{len(selected)}] ok {patient_id}", flush=True)

    counts: dict[str, dict[str, int]] = {
        split: {"NORMAL": 0, "PNEUMONIA": 0} for split in ("train", "val", "test")
    }
    for row in downloaded:
        counts[row["split"]][row["binary_label"]] += 1

    summary = {
        "ok": not failed,
        "source": "Hugging Face mirror Baldezo313/rsna-pneumonia-dataset",
        "raw_root": raw_root.as_posix(),
        "labels_csv": labels_csv.as_posix(),
        "seed": args.seed,
        "requested_per_split_class": per_split_class,
        "selected_count": len(selected),
        "downloaded_count": len(downloaded),
        "counts": counts,
        "failed": failed,
        "downloaded": downloaded,
    }
    output = resolve_project_path(args.summary_output)
    write_json(output, summary)
    print(f"Downloaded {len(downloaded)}/{len(selected)} RSNA DICOM files")
    print(f"Wrote summary: {output.as_posix()}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
