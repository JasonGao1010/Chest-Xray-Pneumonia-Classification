#!/usr/bin/env python3
"""Validate the local Chest X-Ray Images (Pneumonia) dataset layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xray_pneumonia.data import (  # noqa: E402
    DEFAULT_CLASSES,
    DEFAULT_SPLITS,
    format_validation_report,
    validate_dataset_layout,
)


def load_yaml_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}

    try:
        import yaml
    except ImportError:
        print(
            "Warning: PyYAML is not installed; using built-in defaults and CLI "
            "overrides instead of reading the config file.",
            file=sys.stderr,
        )
        return {}

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def nested_get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate train/val/test chest X-ray dataset layout."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="YAML config to read default classes and data_root from, or 'none'.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root. Overrides paths.data_root from config.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Expected class names. Defaults to config project.classes.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Expected split names. Defaults to train val test.",
    )
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Open every discovered image with Pillow to detect corrupt files.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Return exit code 0 even when the dataset is incomplete.",
    )
    parser.add_argument(
        "--no-require-non-empty",
        action="store_true",
        help="Do not treat empty class directories as invalid.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for the structured validation JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = None if args.config.lower() == "none" else Path(args.config)
    config = load_yaml_config(config_path)

    data_root = args.data_root or Path(
        nested_get(config, "paths", "data_root", default="data/raw/chest_xray")
    )
    classes = tuple(args.classes or nested_get(config, "project", "classes", default=DEFAULT_CLASSES))
    splits = tuple(args.splits or DEFAULT_SPLITS)

    result = validate_dataset_layout(
        data_root=data_root,
        class_names=classes,
        splits=splits,
        verify_images=args.verify_images,
        require_non_empty=not args.no_require_non_empty,
    )

    print(format_validation_report(result))

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(result.to_json() + "\n", encoding="utf-8")
        print(f"\nWrote JSON report: {args.json_output.as_posix()}")

    if result.ok or args.allow_missing:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
