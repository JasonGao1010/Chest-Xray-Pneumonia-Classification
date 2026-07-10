#!/usr/bin/env python3
"""Validate the final CXRShift identity protocol and public configuration mapping."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xray_pneumonia.protocol import Identity, load_protocol  # noqa: E402

EXPECTED_MODELS = {"DenseNet121", "ConvNeXt-Tiny", "ViT-B/16"}
EXPECTED_RECIPES = {"ERM", "ERM-Reg", "JT", "JT-DBS"}
EXPECTED_DATASETS = {"Kermany-FG", "RSNA-1707"}


def main() -> int:
    protocol = load_protocol()
    errors: list[str] = []
    if protocol.get("protocol_id") != "CXRShift":
        errors.append("protocol_id must be CXRShift")
    for field, expected in (
        ("models", EXPECTED_MODELS),
        ("recipes", EXPECTED_RECIPES),
        ("datasets", EXPECTED_DATASETS),
    ):
        actual = set(protocol.get(field, {}))
        if actual != expected:
            errors.append(f"{field}: expected {sorted(expected)}, found {sorted(actual)}")

    config_expectations = {
        "DenseNet121__ERM.yaml": ("DenseNet121", "ERM"),
        "ConvNeXt-Tiny__ERM.yaml": ("ConvNeXt-Tiny", "ERM"),
        "ViT-B16__ERM.yaml": ("ViT-B/16", "ERM"),
        "DenseNet121__ERM-Reg.yaml": ("DenseNet121", "ERM-Reg"),
    }
    for filename, expected in config_expectations.items():
        data = yaml.safe_load((ROOT / "configs" / filename).read_text(encoding="utf-8"))
        identity = data.get("protocol", {})
        actual = (identity.get("model"), identity.get("recipe"))
        if identity.get("id") != "CXRShift" or actual != expected:
            errors.append(f"configs/{filename}: invalid protocol identity {identity}")

    run_ids = {
        Identity(model, recipe, seed).run_id
        for model in EXPECTED_MODELS
        for recipe in EXPECTED_RECIPES
        for seed in protocol["seeds"]
    }
    if len(run_ids) != len(EXPECTED_MODELS) * len(EXPECTED_RECIPES) * len(protocol["seeds"]):
        errors.append("run_id template produces collisions")

    if errors:
        print("Protocol validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("CXRShift protocol: valid")
    print(f"models={len(EXPECTED_MODELS)} recipes={len(EXPECTED_RECIPES)} datasets={len(EXPECTED_DATASETS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
