#!/usr/bin/env python3
"""Freeze the current processed RSNA membership with file-level provenance.

The script is read-only with respect to raw and processed datasets. It writes a
canonical CSV manifest and a JSON summary, and refuses to replace either output
unless ``--force`` is supplied explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_REPORTS = (
    Path("results/rsna_hf_subset_download.json"),
    Path("results/rsna_hf_subset_download_expanded.json"),
)


def resolve_project_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path is outside the project root: {path}") from exc


def display_path(path: Path) -> str:
    """Use a stable project-relative path when possible, otherwise an absolute path."""
    try:
        return project_relative(path)
    except ValueError:
        return path.resolve().as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def resolve_recorded_path(value: str, *, base_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts[:2] == ("data", "raw") or path.parts[:2] == ("data", "processed"):
        return PROJECT_ROOT / path
    return base_root / path


def read_source_records(
    report_paths: list[Path],
    *,
    raw_root: Path,
) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    """Map each patient to the earliest retained report that lists it.

    A downloader report can prove that a member was available when the report
    was written, but it cannot prove that the file was first downloaded in that
    invocation because the downloader also accepts pre-existing files. The
    output therefore calls this value a retained source record, not an exact
    acquisition timestamp.
    """
    records: dict[str, dict[str, str]] = {}
    report_summaries: list[dict[str, Any]] = []
    for report_path in report_paths:
        if not report_path.is_file():
            raise FileNotFoundError(f"Source report not found: {report_path}")
        payload = load_json(report_path)
        provider = str(payload.get("source") or "unknown")
        downloaded = payload.get("downloaded")
        if not isinstance(downloaded, list):
            raise ValueError(f"Missing downloaded member list: {report_path}")
        batch = report_path.stem
        for item in downloaded:
            if not isinstance(item, dict) or not item.get("patientId"):
                raise ValueError(f"Malformed downloaded row in {report_path}")
            patient_id = str(item["patientId"])
            recorded_path = str(item.get("source_path") or "")
            if not recorded_path:
                raise ValueError(f"Missing source_path for {patient_id} in {report_path}")
            resolved = resolve_recorded_path(recorded_path, base_root=raw_root)
            records.setdefault(
                patient_id,
                {
                    "source_batch": batch,
                    "source_provider": provider,
                    "source_report": project_relative(report_path),
                    "reported_source_path": project_relative(resolved),
                },
            )
        report_summaries.append(
            {
                "path": project_relative(report_path),
                "sha256": sha256_file(report_path),
                "listed_member_count": len(downloaded),
                "source_provider": provider,
            }
        )
    return records, report_summaries


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"patientId", "source_path", "image_path", "split", "target", "binary_label"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing metadata columns in {path}: {sorted(missing)}")
        rows = [{key: str(value) for key, value in row.items()} for row in reader]
    patient_ids = [row["patientId"] for row in rows]
    duplicates = sorted(patient_id for patient_id, count in Counter(patient_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate patientId values in metadata: {duplicates[:10]}")
    return rows


def prepare_outputs(paths: list[Path], *, force: bool) -> None:
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing and not force:
        rendered = "\n  - ".join(path.as_posix() for path in existing)
        raise FileExistsError(
            "Refusing to replace frozen manifest outputs. "
            "Choose new paths or rerun with --force:\n  - " + rendered
        )
    for path in existing:
        if path.is_dir() and not path.is_symlink():
            raise IsADirectoryError(f"Expected a manifest file, found a directory: {path}")
        path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the current 1707-member RSNA subset.")
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/rsna_binary/metadata.csv"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/rsna_pneumonia"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/rsna_binary"))
    parser.add_argument(
        "--source-report",
        action="append",
        type=Path,
        dest="source_reports",
        help="Retained downloader report; repeat to provide multiple reports. Defaults to the two repository reports.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/splits/rsna_available_1707_manifest.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/splits/rsna_available_1707_manifest_summary.json"),
    )
    parser.add_argument("--expected-count", type=int, default=1707)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing manifest outputs. Raw and processed datasets are never modified.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_path = resolve_project_path(args.metadata)
    raw_root = resolve_project_path(args.raw_root)
    processed_root = resolve_project_path(args.processed_root)
    report_paths = [resolve_project_path(path) for path in (args.source_reports or DEFAULT_SOURCE_REPORTS)]
    manifest_path = resolve_project_path(args.manifest)
    summary_path = resolve_project_path(args.summary)

    if args.expected_count <= 0:
        raise ValueError("--expected-count must be positive")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"RSNA metadata not found: {metadata_path}")
    if not raw_root.is_dir() or not processed_root.is_dir():
        raise FileNotFoundError("Both --raw-root and --processed-root must exist")
    prepare_outputs([manifest_path, summary_path], force=args.force)

    metadata_rows = read_metadata(metadata_path)
    if len(metadata_rows) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} metadata rows, found {len(metadata_rows)}. "
            "Pass the intended count explicitly only after auditing the membership change."
        )
    source_records, report_summaries = read_source_records(report_paths, raw_root=raw_root)

    allowed_splits = {"train", "val", "test"}
    allowed_labels = {"NORMAL", "PNEUMONIA"}
    frozen_rows: list[dict[str, Any]] = []
    split_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    batch_counts: Counter[str] = Counter()
    for row in sorted(metadata_rows, key=lambda item: item["patientId"]):
        patient_id = row["patientId"]
        split = row["split"]
        label = row["binary_label"]
        if split not in allowed_splits or label not in allowed_labels:
            raise ValueError(f"Invalid split/label for {patient_id}: {split}/{label}")
        expected_target = "1" if label == "PNEUMONIA" else "0"
        if row["target"] != expected_target:
            raise ValueError(f"Target/label conflict for {patient_id}: {row['target']}/{label}")

        source_path = resolve_recorded_path(row["source_path"], base_root=raw_root).resolve()
        processed_path = resolve_recorded_path(row["image_path"], base_root=processed_root).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source image for {patient_id}: {source_path}")
        if not processed_path.is_file():
            raise FileNotFoundError(f"Missing processed image for {patient_id}: {processed_path}")
        if source_path.stem != patient_id or processed_path.stem != patient_id:
            raise ValueError(f"Filename/patientId mismatch for {patient_id}")
        source_relative = project_relative(source_path)
        processed_relative = project_relative(processed_path)

        provenance = source_records.get(patient_id)
        if provenance is None:
            provenance = {
                "source_batch": "unknown",
                "source_provider": "unknown",
                "source_report": "unknown",
                "reported_source_path": "unknown",
            }
        elif provenance["reported_source_path"] != source_relative:
            raise ValueError(
                f"Retained source record disagrees with metadata for {patient_id}: "
                f"{provenance['reported_source_path']} != {source_relative}"
            )

        frozen_rows.append(
            {
                "patientId": patient_id,
                "target": row["target"],
                "label": label,
                "split": split,
                "source_batch": provenance["source_batch"],
                "source_provider": provenance["source_provider"],
                "source_report": provenance["source_report"],
                "source_path": source_relative,
                "source_bytes": source_path.stat().st_size,
                "source_sha256": sha256_file(source_path),
                "processed_path": processed_relative,
                "processed_bytes": processed_path.stat().st_size,
                "processed_sha256": sha256_file(processed_path),
            }
        )
        split_label_counts[split][label] += 1
        batch_counts[provenance["source_batch"]] += 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patientId",
        "target",
        "label",
        "split",
        "source_batch",
        "source_provider",
        "source_report",
        "source_path",
        "source_bytes",
        "source_sha256",
        "processed_path",
        "processed_bytes",
        "processed_sha256",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(frozen_rows)

    summary = {
        "schema_version": 1,
        "ok": True,
        "member_count": len(frozen_rows),
        "expected_member_count": args.expected_count,
        "metadata": {
            "path": project_relative(metadata_path),
            "sha256": sha256_file(metadata_path),
        },
        "manifest": {
            "path": display_path(manifest_path),
            "sha256": sha256_file(manifest_path),
            "hash_algorithm": "SHA-256",
            "path_format": "project-relative POSIX",
        },
        "split_label_counts": {
            split: dict(sorted(split_label_counts[split].items())) for split in sorted(split_label_counts)
        },
        "source_batch_counts": dict(sorted(batch_counts.items())),
        "source_reports": report_summaries,
        "source_batch_semantics": (
            "Earliest retained downloader report listing the member; this records documented availability, "
            "not a proven first-download event. Members absent from all retained reports are marked unknown."
        ),
        "raw_and_processed_data_modified": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
