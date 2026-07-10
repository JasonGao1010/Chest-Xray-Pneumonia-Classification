"""Canonical project identities and compatibility aliases for CXRShift."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT_ROOT / "protocol/naming_protocol.yaml"


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Identity:
    model: str
    recipe: str
    seed: int

    @property
    def model_slug(self) -> str:
        data = load_protocol()["models"][self.model]
        return str(data.get("artifact_slug", self.model))

    @property
    def run_id(self) -> str:
        protocol = load_protocol()
        return protocol["run_id"].format(
            protocol_id=protocol["protocol_id"],
            model_slug=self.model_slug,
            recipe_id=self.recipe,
            seed=self.seed,
        )


def artifact_name(identity: Identity, dataset: str, split: str, artifact: str, extension: str) -> str:
    protocol = load_protocol()
    if dataset not in protocol["datasets"]:
        raise KeyError(f"Unknown dataset identity: {dataset}")
    return protocol["artifact_name"].format(
        run_id=identity.run_id,
        dataset_id=dataset,
        split=split,
        artifact=artifact,
        extension=extension,
    )


def legacy_model_slug(model: str) -> str:
    return str(load_protocol()["models"][model]["legacy_slug"])


def legacy_family(recipe: str) -> str:
    return str(load_protocol()["recipes"][recipe]["legacy_family"])


def legacy_dataset_slug(dataset: str) -> str:
    return str(load_protocol()["datasets"][dataset]["legacy_slug"])
