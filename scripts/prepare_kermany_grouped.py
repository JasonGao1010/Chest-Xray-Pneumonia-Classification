#!/usr/bin/env python3
"""Create a conservative filename-grouped Kermany split with hard links."""

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

from scripts.audit_integrity import IMAGE_SUFFIXES, subject_id  # noqa: E402

MANIFEST_FIELDS = [
    "path",
    "true_label",
    "subject_id",
    "split",
    "source_split",
    "source_path",
    "sha256",
]
SEMANTIC_HASH_SCHEME = "csv_rows_sorted_canonical_json_v1"


def resolve(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_semantic_sha256(rows: list[dict[str, str]]) -> str:
    """Hash canonical row values, independent of CSV line endings and row order."""
    canonical_rows = sorted(
        [[str(row.get(field, "")) for field in MANIFEST_FIELDS] for row in rows]
    )
    payload = {
        "fields": MANIFEST_FIELDS,
        "rows": canonical_rows,
        "scheme": SEMANTIC_HASH_SCHEME,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def detect_line_endings(path: Path) -> str:
    payload = path.read_bytes()
    crlf = payload.count(b"\r\n")
    bare_lf = payload.count(b"\n") - crlf
    bare_cr = payload.count(b"\r") - crlf
    if crlf and not bare_lf and not bare_cr:
        return "CRLF"
    if bare_lf and not crlf and not bare_cr:
        return "LF"
    if not crlf and not bare_lf and not bare_cr:
        return "none"
    return "mixed"


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
    parser = argparse.ArgumentParser(description="Build a filename-grouped Kermany dataset.")
    parser.add_argument("--source-root", type=Path, default=Path("data/raw/chest_xray"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/kermany_grouped_seed42"))
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/kermany_grouped_seed42.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/kermany_grouped_summary.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing output paths before rebuilding. Without this flag, existing outputs are never overwritten.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Read the frozen manifest and data without rebuilding; update only --summary.",
    )
    parser.add_argument(
        "--verify-files",
        action="store_true",
        help="With --verify-only, recompute every source/image SHA-256 and validate hard links.",
    )
    return parser.parse_args()


def prepare_output_paths(paths: list[Path], *, force: bool) -> None:
    """Refuse implicit replacement and remove outputs only after explicit consent."""
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing and not force:
        rendered = "\n  - ".join(path.as_posix() for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing Kermany outputs. "
            "Choose new paths or rerun with --force:\n  - " + rendered
        )
    for path in existing:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != MANIFEST_FIELDS:
            raise ValueError(
                f"Unexpected manifest columns in {path}: {reader.fieldnames}; expected {MANIFEST_FIELDS}"
            )
        return [{field: str(row[field]) for field in MANIFEST_FIELDS} for row in reader]


def build_summary(
    *,
    rows: list[dict[str, str]],
    seed: int,
    source_root: Path,
    output_root: Path,
    manifest: Path,
    verify_files: bool,
) -> dict[str, Any]:
    image_counts: dict[str, Counter[str]] = defaultdict(Counter)
    subject_sets: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    missing_files: list[str] = []
    digest_mismatches: list[str] = []
    non_hardlinks: list[str] = []
    for row in rows:
        image_counts[row["split"]][row["true_label"]] += 1
        subject_sets[row["split"]][row["true_label"]].add(row["subject_id"])
        destination = output_root / row["path"]
        source = source_root / row["source_path"]
        for candidate in (source, destination):
            try:
                candidate.resolve().relative_to(
                    source_root.resolve() if candidate == source else output_root.resolve()
                )
            except ValueError as exc:
                raise ValueError(f"Manifest path escapes its dataset root: {candidate}") from exc
        if not source.is_file() or not destination.is_file():
            missing_files.append(row["path"])
            continue
        if verify_files:
            actual = file_sha256(source)
            if actual != row["sha256"]:
                digest_mismatches.append(row["source_path"])
            if not os.path.samefile(source, destination):
                non_hardlinks.append(row["path"])
                if file_sha256(destination) != row["sha256"]:
                    digest_mismatches.append(row["path"])
    if missing_files or digest_mismatches:
        raise ValueError(
            "Frozen Kermany verification failed: "
            f"missing={len(missing_files)}, digest_mismatches={len(digest_mismatches)}"
        )
    return {
        "ok": True,
        "seed": seed,
        "source_root": source_root.as_posix(),
        "output_root": output_root.as_posix(),
        "manifest": manifest.as_posix(),
        "manifest_sha256": file_sha256(manifest),
        "manifest_sha256_raw": file_sha256(manifest),
        "manifest_line_endings": detect_line_endings(manifest),
        "manifest_sha256_semantic": manifest_semantic_sha256(rows),
        "manifest_semantic_hash_scheme": SEMANTIC_HASH_SCHEME,
        "image_counts": {split: dict(value) for split, value in image_counts.items()},
        "subject_counts": {
            split: {label: len(ids) for label, ids in by_label.items()}
            for split, by_label in subject_sets.items()
        },
        "subject_count_field_note": (
            "Counts are based on filename-derived grouping tokens, not verified patient identifiers."
        ),
        "total_images": len(rows),
        "link_mode": "hardlink",
        "verification": {
            "file_hashes_recomputed": verify_files,
            "missing_file_count": len(missing_files),
            "digest_mismatch_count": len(digest_mismatches),
            "non_hardlink_count": len(non_hardlinks) if verify_files else None,
        },
    }


def write_summary_safely(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing summary without --force: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        if not force:
            raise FileExistsError(f"Temporary summary already exists: {temporary}")
        temporary.unlink()
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_frozen_outputs(args: argparse.Namespace) -> int:
    source_root = resolve(args.source_root)
    output_root = resolve(args.output_root)
    manifest = resolve(args.manifest)
    summary_path = resolve(args.summary)
    if not manifest.is_file():
        raise FileNotFoundError(f"Frozen manifest not found: {manifest}")
    if not source_root.is_dir() or not output_root.is_dir():
        raise FileNotFoundError("Frozen source and output roots must both exist")
    rows = read_manifest(manifest)
    summary = build_summary(
        rows=rows,
        seed=args.seed,
        source_root=source_root,
        output_root=output_root,
        manifest=manifest,
        verify_files=args.verify_files,
    )
    write_summary_safely(summary_path, summary, force=args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.verify_files and not args.verify_only:
        raise ValueError("--verify-files requires --verify-only")
    if args.verify_only:
        return verify_frozen_outputs(args)
    if args.train_fraction <= 0 or args.val_fraction <= 0 or args.train_fraction + args.val_fraction >= 1:
        raise ValueError("fractions must be positive and leave a non-empty test fraction")
    source_root, output_root = resolve(args.source_root), resolve(args.output_root)
    manifest = resolve(args.manifest)
    summary_path = resolve(args.summary)
    resolved_output = output_root.resolve()
    resolved_source = source_root.resolve()
    if (
        resolved_output == PROJECT_ROOT.resolve()
        or resolved_output.is_relative_to(resolved_source)
        or resolved_source.is_relative_to(resolved_output)
    ):
        raise ValueError("--output-root must not overlap the project or source roots")
    prepare_output_paths([output_root, manifest, summary_path], force=args.force)
    rows = collect(source_root)
    assignments = assign_subjects(rows, args.seed, args.train_fraction, args.val_fraction)
    manifest_rows = []
    for row in rows:
        split = assignments[(row["label"], row["subject_id"])]
        destination = output_root / split / row["label"] / row["source"].name
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(row["source"], destination)
        digest = file_sha256(row["source"])
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
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(manifest_rows, key=lambda x: (x["split"], x["true_label"], x["path"])))
    summary = build_summary(
        rows=[{field: str(row[field]) for field in MANIFEST_FIELDS} for row in manifest_rows],
        seed=args.seed,
        source_root=source_root,
        output_root=output_root,
        manifest=manifest,
        verify_files=False,
    )
    write_summary_safely(summary_path, summary, force=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
