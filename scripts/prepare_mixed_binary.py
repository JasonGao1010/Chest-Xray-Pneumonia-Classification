#!/usr/bin/env python3
"""Build an ImageFolder-style binary dataset by merging two prepared datasets."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xray_pneumonia.data import (  # noqa: E402
    DEFAULT_CLASSES,
    IMAGE_EXTENSIONS,
    validate_dataset_layout,
)


def resolve_project_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_split(source_root: Path, output_root: Path, split: str, prefix: str) -> dict[str, int]:
    counts = {class_name: 0 for class_name in DEFAULT_CLASSES}
    for class_name in DEFAULT_CLASSES:
        source_dir = source_root / split / class_name
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing source class directory: {source_dir}")
        destination_dir = output_root / split / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(source_dir.iterdir()):
            if not source.is_file() or source.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            destination = destination_dir / f"{prefix}_{source.name}"
            shutil.copy2(source, destination)
            counts[class_name] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge two binary ImageFolder datasets.")
    parser.add_argument("--source-a", type=Path, required=True)
    parser.add_argument("--source-b", type=Path, required=True)
    parser.add_argument("--prefix-a", type=str, default="a")
    parser.add_argument("--prefix-b", type=str, default="b")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory. Existing outputs are preserved by default.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_a = resolve_project_path(args.source_a)
    source_b = resolve_project_path(args.source_b)
    output_root = resolve_project_path(args.output_root)
    summary_output = resolve_project_path(args.summary_output)

    resolved_output = output_root.resolve()
    resolved_sources = (source_a.resolve(), source_b.resolve())
    if resolved_output == PROJECT_ROOT.resolve() or any(
        resolved_output.is_relative_to(source) or source.is_relative_to(resolved_output)
        for source in resolved_sources
    ):
        raise ValueError("--output-root must not overlap the project or either source root")
    resolved_summary = summary_output.resolve()
    if any(
        resolved_summary == source or resolved_summary.is_relative_to(source)
        for source in resolved_sources
    ):
        raise ValueError("--summary-output must not be inside either source root")
    existing_outputs = [
        path
        for path in (output_root, summary_output)
        if path.exists() or path.is_symlink()
    ]
    if existing_outputs:
        if not args.force:
            raise FileExistsError(
                "Refusing to overwrite existing mixed-data outputs. Choose new "
                "paths or rerun with --force."
            )
        for path in existing_outputs:
            if not path.exists() and not path.is_symlink():
                continue
            if path.is_symlink() or path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)

    counts: dict[str, dict[str, int]] = {}
    by_source: dict[str, dict[str, dict[str, int]]] = {
        args.prefix_a: {},
        args.prefix_b: {},
    }
    for split in args.splits:
        counts[split] = {class_name: 0 for class_name in DEFAULT_CLASSES}
        a_counts = copy_split(source_a, output_root, split, args.prefix_a)
        b_counts = copy_split(source_b, output_root, split, args.prefix_b)
        by_source[args.prefix_a][split] = a_counts
        by_source[args.prefix_b][split] = b_counts
        for class_name in DEFAULT_CLASSES:
            counts[split][class_name] = a_counts[class_name] + b_counts[class_name]

    validation = validate_dataset_layout(
        output_root,
        class_names=DEFAULT_CLASSES,
        splits=tuple(args.splits),
        verify_images=False,
    )
    payload = {
        "ok": validation.ok,
        "sources": {
            args.prefix_a: source_a.as_posix(),
            args.prefix_b: source_b.as_posix(),
        },
        "output_root": output_root.as_posix(),
        "splits": list(args.splits),
        "counts": counts,
        "by_source": by_source,
        "dataset": validation.to_dict(),
    }
    write_json(summary_output, payload)
    print(f"Prepared mixed dataset: {output_root.as_posix()}")
    print(f"Wrote summary: {summary_output.as_posix()}")
    return 0 if validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
