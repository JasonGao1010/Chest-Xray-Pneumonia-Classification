#!/usr/bin/env python3
"""Train an EfficientNetV2 baseline for chest X-ray pneumonia classification."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xray_pneumonia.data import (  # noqa: E402
    DEFAULT_CLASSES,
    DEFAULT_SPLITS,
    XRayImageDataset,
    build_class_weights,
    stratified_holdout_indices,
    validate_dataset_layout,
)

REQUIRED_TRAINING_MODULES = ("torch", "torchvision", "timm")
TORCHVISION_MODELS = {
    "torchvision:vit_b_16",
    "torchvision:densenet121",
    "torchvision:convnext_tiny",
}

DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "classes": list(DEFAULT_CLASSES),
    },
    "paths": {
        "data_root": "data/raw/chest_xray",
        "results_dir": "results",
        "checkpoints_dir": "outputs/checkpoints",
    },
    "data": {
        "image_size": 224,
        "train_split": "train",
        "val_split": "val",
        "normalize": "imagenet",
        "augment": {
            "horizontal_flip": True,
            "random_rotation_degrees": 7,
            "color_jitter": False,
        },
    },
    "training": {
        "model": "tf_efficientnetv2_s.in1k",
        "pretrained": True,
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "seed": 42,
        "mixed_precision": False,
        "num_workers": 2,
    },
}


@dataclass(frozen=True)
class TrainSettings:
    classes: tuple[str, ...]
    data_root: Path
    train_split: str
    val_split: str
    results_dir: Path
    checkpoints_dir: Path
    image_size: int
    normalize: str
    model: str
    pretrained: bool
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    scheduler: str
    seed: int
    mixed_precision: bool
    use_class_weights: bool
    num_workers: int
    device: str
    max_train_batches: int | None
    max_val_batches: int | None
    smoke_test: bool
    augment: dict[str, Any]
    train_holdout_fraction: float
    holdout_seed: int
    holdout_manifest_output: Path | None
    init_checkpoint: Path | None
    domain_balanced_prefixes: tuple[str, ...]
    freeze_mode: str
    label_smoothing: float


def deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {
        key: deep_merge(value, {}) if isinstance(value, dict) else value
        for key, value in base.items()
    }
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}

    try:
        import yaml
    except ImportError:
        print(
            "Warning: PyYAML is not installed; using built-in training defaults "
            "and CLI overrides instead of reading the config file.",
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


def resolve_project_path(path: Path | str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def dependency_status() -> dict[str, bool]:
    return {name: module_available(name) for name in REQUIRED_TRAINING_MODULES}


def model_status(settings: TrainSettings) -> dict[str, Any]:
    if settings.model in TORCHVISION_MODELS:
        return {
            "ok": True,
            "name": settings.model,
            "pretrained": settings.pretrained,
            "message": "model can be created with torchvision",
        }

    if not module_available("timm"):
        return {
            "ok": False,
            "name": settings.model,
            "message": "timm is not available",
        }

    try:
        import timm

        timm.create_model(settings.model, pretrained=False, num_classes=len(settings.classes))
        has_pretrained = settings.model in timm.list_models(pretrained=True)
    except Exception as exc:
        return {
            "ok": False,
            "name": settings.model,
            "pretrained": settings.pretrained,
            "message": f"could not create model with timm: {exc}",
        }

    if settings.pretrained and not has_pretrained:
        return {
            "ok": False,
            "name": settings.model,
            "pretrained": settings.pretrained,
            "message": "model exists, but timm has no registered pretrained weights for it",
        }

    return {
        "ok": True,
        "name": settings.model,
        "pretrained": settings.pretrained,
        "message": "model can be created with timm",
    }


def create_classifier_model(
    model_name: str,
    pretrained: bool,
    num_classes: int,
    timm_module: Any,
    torch_module: Any,
) -> Any:
    if model_name == "torchvision:vit_b_16":
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        if pretrained:
            model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            in_features = model.heads.head.in_features
            model.heads.head = torch_module.nn.Linear(in_features, num_classes)
            return model
        return vit_b_16(weights=None, num_classes=num_classes)

    if model_name == "torchvision:densenet121":
        from torchvision.models import DenseNet121_Weights, densenet121

        if pretrained:
            model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            in_features = model.classifier.in_features
            model.classifier = torch_module.nn.Linear(in_features, num_classes)
            return model
        return densenet121(weights=None, num_classes=num_classes)

    if model_name == "torchvision:convnext_tiny":
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        if pretrained:
            model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = torch_module.nn.Linear(in_features, num_classes)
            return model
        return convnext_tiny(weights=None, num_classes=num_classes)

    return timm_module.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def to_jsonable_settings(settings: TrainSettings) -> dict[str, Any]:
    return {
        "classes": list(settings.classes),
        "data_root": settings.data_root.as_posix(),
        "train_split": settings.train_split,
        "val_split": settings.val_split,
        "results_dir": settings.results_dir.as_posix(),
        "checkpoints_dir": settings.checkpoints_dir.as_posix(),
        "image_size": settings.image_size,
        "normalize": settings.normalize,
        "model": settings.model,
        "pretrained": settings.pretrained,
        "epochs": settings.epochs,
        "batch_size": settings.batch_size,
        "learning_rate": settings.learning_rate,
        "weight_decay": settings.weight_decay,
        "optimizer": settings.optimizer,
        "scheduler": settings.scheduler,
        "seed": settings.seed,
        "mixed_precision": settings.mixed_precision,
        "use_class_weights": settings.use_class_weights,
        "num_workers": settings.num_workers,
        "device": settings.device,
        "max_train_batches": settings.max_train_batches,
        "max_val_batches": settings.max_val_batches,
        "smoke_test": settings.smoke_test,
        "augment": settings.augment,
        "train_holdout_fraction": settings.train_holdout_fraction,
        "holdout_seed": settings.holdout_seed,
        "holdout_manifest_output": (
            settings.holdout_manifest_output.as_posix()
            if settings.holdout_manifest_output is not None
            else None
        ),
        "init_checkpoint": (
            settings.init_checkpoint.as_posix()
            if settings.init_checkpoint is not None
            else None
        ),
        "domain_balanced_prefixes": list(settings.domain_balanced_prefixes),
        "freeze_mode": settings.freeze_mode,
        "label_smoothing": settings.label_smoothing,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or readiness-check the chest X-ray baseline classifier."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="YAML config to read defaults from, or 'none'.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root. Overrides paths.data_root from config.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory for training result JSON files.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=None,
        help="Directory for checkpoint subdirectories.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for check-only or training summary JSON.",
    )
    parser.add_argument(
        "--check-only",
        "--dry-run",
        action="store_true",
        dest="check_only",
        help="Validate dataset and dependency readiness without importing torch.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one short training epoch with a few batches.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument(
        "--train-holdout-fraction",
        type=float,
        default=0.0,
        help=(
            "Stratified fraction of the training split to hold out for validation/calibration. "
            "When set, the official tiny val split is not used for checkpoint selection."
        ),
    )
    parser.add_argument(
        "--holdout-seed",
        type=int,
        default=None,
        help="Seed for the stratified train holdout split. Defaults to the training seed.",
    )
    parser.add_argument(
        "--holdout-manifest-output",
        type=Path,
        default=None,
        help="Optional CSV path for held-out train samples used as validation/calibration.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint used to initialize model weights before training. "
            "This does not restore the old optimizer or scheduler state."
        ),
    )
    parser.add_argument(
        "--domain-balanced-prefixes",
        nargs="+",
        default=None,
        help=(
            "Optional filename prefixes used to balance source domains in mixed training, "
            "for example: kermany rsna."
        ),
    )
    parser.add_argument(
        "--freeze-mode",
        choices=["none", "classifier", "last_block"],
        default="none",
        help=(
            "Optional fine-tuning mode. classifier freezes the backbone and trains "
            "only the classification head; last_block also unfreezes the final feature block."
        ),
    )

    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument("--pretrained", action="store_true", default=None)
    pretrained_group.add_argument("--no-pretrained", action="store_false", dest="pretrained")

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--mixed-precision", action="store_true", default=None)
    amp_group.add_argument("--no-mixed-precision", action="store_false", dest="mixed_precision")
    weights_group = parser.add_mutually_exclusive_group()
    weights_group.add_argument("--class-weights", action="store_true", default=None)
    weights_group.add_argument("--no-class-weights", action="store_false", dest="class_weights")
    return parser.parse_args()


def build_settings(args: argparse.Namespace, config: dict[str, Any]) -> TrainSettings:
    classes = tuple(nested_get(config, "project", "classes", default=DEFAULT_CLASSES))
    data_root = resolve_project_path(
        args.data_root or nested_get(config, "paths", "data_root", default="data/raw/chest_xray")
    )
    results_dir = resolve_project_path(
        args.results_dir or nested_get(config, "paths", "results_dir", default="results")
    )
    checkpoints_dir = resolve_project_path(
        args.checkpoints_dir
        or nested_get(config, "paths", "checkpoints_dir", default="outputs/checkpoints")
    )
    augment = dict(nested_get(config, "data", "augment", default={}))

    epochs = int(args.epochs or nested_get(config, "training", "epochs", default=10))
    batch_size = int(args.batch_size or nested_get(config, "training", "batch_size", default=32))
    max_train_batches = args.max_train_batches
    max_val_batches = args.max_val_batches
    if args.smoke_test:
        epochs = min(epochs, 1)
        batch_size = min(batch_size, 8)
        max_train_batches = max_train_batches or 2
        max_val_batches = max_val_batches or 1

    pretrained = nested_get(config, "training", "pretrained", default=True)
    if args.pretrained is not None:
        pretrained = args.pretrained

    mixed_precision = nested_get(config, "training", "mixed_precision", default=True)
    if args.mixed_precision is not None:
        mixed_precision = args.mixed_precision

    use_class_weights = nested_get(config, "training", "use_class_weights", default=True)
    if args.class_weights is not None:
        use_class_weights = args.class_weights

    return TrainSettings(
        classes=classes,
        data_root=data_root,
        train_split=str(nested_get(config, "data", "train_split", default="train")),
        val_split=str(nested_get(config, "data", "val_split", default="val")),
        results_dir=results_dir,
        checkpoints_dir=checkpoints_dir,
        image_size=int(args.image_size or nested_get(config, "data", "image_size", default=224)),
        normalize=str(nested_get(config, "data", "normalize", default="imagenet")),
        model=str(
            args.model
            or nested_get(config, "training", "model", default="tf_efficientnetv2_s.in1k")
        ),
        pretrained=bool(pretrained),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=float(
            args.learning_rate or nested_get(config, "training", "learning_rate", default=0.0003)
        ),
        weight_decay=float(
            args.weight_decay or nested_get(config, "training", "weight_decay", default=0.0001)
        ),
        optimizer=str(nested_get(config, "training", "optimizer", default="adamw")),
        scheduler=str(nested_get(config, "training", "scheduler", default="cosine")),
        seed=int(
            args.seed
            if args.seed is not None
            else nested_get(config, "training", "seed", default=42)
        ),
        mixed_precision=bool(mixed_precision),
        use_class_weights=bool(use_class_weights),
        num_workers=int(
            args.num_workers
            if args.num_workers is not None
            else nested_get(config, "training", "num_workers", default=2)
        ),
        device=args.device,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        smoke_test=bool(args.smoke_test),
        augment=augment,
        train_holdout_fraction=float(args.train_holdout_fraction or 0.0),
        holdout_seed=int(args.holdout_seed if args.holdout_seed is not None else nested_get(config, "training", "seed", default=42)),
        holdout_manifest_output=(
            resolve_project_path(args.holdout_manifest_output)
            if args.holdout_manifest_output is not None
            else None
        ),
        init_checkpoint=(
            resolve_project_path(args.init_checkpoint)
            if args.init_checkpoint is not None
            else None
        ),
        domain_balanced_prefixes=tuple(args.domain_balanced_prefixes or ()),
        freeze_mode=str(args.freeze_mode),
        label_smoothing=float(
            args.label_smoothing
            if args.label_smoothing is not None
            else nested_get(config, "training", "label_smoothing", default=0.0)
        ),
    )


def readiness_report(settings: TrainSettings) -> dict[str, Any]:
    validation_splits = (
        (settings.train_split,)
        if settings.train_holdout_fraction > 0
        else (settings.train_split, settings.val_split)
    )
    validation = validate_dataset_layout(
        settings.data_root,
        class_names=settings.classes,
        splits=validation_splits,
    )
    deps = dependency_status()
    missing = [name for name, available in deps.items() if not available]
    model = model_status(settings)
    checkpoint_ok = (
        settings.init_checkpoint is None
        or settings.init_checkpoint.is_file()
    )
    return {
        "ok": validation.ok and not missing and model["ok"] and checkpoint_ok,
        "dataset_ok": validation.ok,
        "model_ok": model["ok"],
        "init_checkpoint_ok": checkpoint_ok,
        "missing_required_dependencies": missing,
        "dependencies": deps,
        "model": model,
        "init_checkpoint": (
            settings.init_checkpoint.as_posix()
            if settings.init_checkpoint is not None
            else None
        ),
        "settings": to_jsonable_settings(settings),
        "dataset": validation.to_dict(),
    }


def print_readiness(report: dict[str, Any]) -> None:
    print("Training readiness:")
    print(f"  Status: {'OK' if report['ok'] else 'BLOCKED'}")
    print(f"  Dataset: {'OK' if report['dataset_ok'] else 'INVALID'}")
    print(
        "  Model: "
        f"{'OK' if report['model_ok'] else 'INVALID'} "
        f"({report['model']['name']})"
    )
    if not report["model_ok"]:
        print(f"  Model issue: {report['model']['message']}")
    missing = report["missing_required_dependencies"]
    if missing:
        print(f"  Missing required dependencies: {', '.join(missing)}")
    else:
        print("  Missing required dependencies: none")
    if report["init_checkpoint"] is not None:
        print(
            "  Init checkpoint: "
            f"{'OK' if report['init_checkpoint_ok'] else 'MISSING'} "
            f"({report['init_checkpoint']})"
        )


def load_checkpoint(torch_module: Any, checkpoint_path: Path) -> dict[str, Any]:
    try:
        return torch_module.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(checkpoint_path, map_location="cpu")


def import_training_stack():
    missing = [name for name, available in dependency_status().items() if not available]
    if missing:
        raise RuntimeError(
            "Missing required training dependencies: "
            + ", ".join(missing)
            + ". Install requirements.txt before training."
        )

    import numpy as np
    import timm
    import torch
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
    from torchvision import transforms

    return np, timm, torch, DataLoader, Subset, WeightedRandomSampler, transforms


def set_seed(seed: int, np_module: Any, torch_module: Any) -> None:
    random.seed(seed)
    np_module.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    # Make repeated runs auditable. Some CUDA operators can otherwise choose
    # different kernels between runs even when all random seeds are fixed.
    if hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.benchmark = False
        torch_module.backends.cudnn.deterministic = True
    try:
        torch_module.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass


def build_transforms(settings: TrainSettings, transforms_module: Any, train: bool):
    steps: list[Any] = []
    if train:
        steps.append(transforms_module.Resize((settings.image_size, settings.image_size)))
        if settings.augment.get("horizontal_flip", False):
            steps.append(transforms_module.RandomHorizontalFlip())
        rotation = float(settings.augment.get("random_rotation_degrees", 0) or 0)
        if rotation > 0:
            steps.append(transforms_module.RandomRotation(rotation))
        if settings.augment.get("color_jitter", False):
            steps.append(
                transforms_module.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.05,
                    hue=0.02,
                )
            )
    else:
        steps.append(transforms_module.Resize((settings.image_size, settings.image_size)))

    steps.append(transforms_module.ToTensor())
    if settings.normalize.lower() == "imagenet":
        steps.append(
            transforms_module.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        )
    return transforms_module.Compose(steps)


def class_counts_from_indices(
    samples: list[tuple[Path, int]],
    indices: list[int],
    class_names: tuple[str, ...],
) -> dict[str, int]:
    counts = {class_name: 0 for class_name in class_names}
    for index in indices:
        _, label = samples[index]
        counts[class_names[int(label)]] += 1
    return counts


def domain_for_sample(path: Path, prefixes: tuple[str, ...]) -> str:
    name = path.name
    for prefix in prefixes:
        normalized = prefix.rstrip("_")
        if name.startswith(f"{normalized}_"):
            return normalized
    return "other"


def build_domain_balanced_sampler(
    torch_module: Any,
    samples: list[tuple[Path, int]],
    indices: list[int] | None,
    prefixes: tuple[str, ...],
) -> tuple[Any | None, dict[str, int]]:
    if not prefixes:
        return None, {}
    active_indices = indices if indices is not None else list(range(len(samples)))
    domain_counts: dict[str, int] = {}
    domains: list[str] = []
    for index in active_indices:
        domain = domain_for_sample(samples[index][0], prefixes)
        domains.append(domain)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    weights = [1.0 / domain_counts[domain] for domain in domains]
    sampler = torch_module.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
    return sampler, domain_counts


def write_holdout_manifest(
    path: Path,
    data_root: Path,
    samples: list[tuple[Path, int]],
    indices: list[int],
    class_names: tuple[str, ...],
    source_split: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "true_label", "source_split"])
        writer.writeheader()
        for index in indices:
            sample_path, label = samples[index]
            writer.writerow(
                {
                    "path": sample_path.relative_to(data_root).as_posix(),
                    "true_label": class_names[int(label)],
                    "source_split": source_split,
                }
            )


def choose_device(requested: str, torch_module: Any):
    if requested != "auto":
        return torch_module.device(requested)
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def build_optimizer(settings: TrainSettings, torch_module: Any, model: Any):
    name = settings.optimizer.lower()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters are available for the requested freeze mode.")
    if name == "adamw":
        return torch_module.optim.AdamW(
            parameters,
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )
    if name == "sgd":
        return torch_module.optim.SGD(
            parameters,
            lr=settings.learning_rate,
            momentum=0.9,
            weight_decay=settings.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {settings.optimizer}")


def build_scheduler(settings: TrainSettings, torch_module: Any, optimizer: Any):
    name = settings.scheduler.lower()
    if name in {"none", "off", "disabled"}:
        return None
    if name == "cosine":
        return torch_module.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(settings.epochs, 1),
        )
    raise ValueError(f"Unsupported scheduler: {settings.scheduler}")


def set_trainable(module: Any, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def apply_freeze_mode(model: Any, model_name: str, freeze_mode: str) -> dict[str, Any]:
    if freeze_mode == "none":
        return {"mode": freeze_mode, "trainable_parameters": count_trainable_parameters(model)}

    set_trainable(model, False)
    unfrozen: list[str] = []
    if model_name == "torchvision:densenet121":
        set_trainable(model.classifier, True)
        unfrozen.append("classifier")
        if freeze_mode == "last_block":
            set_trainable(model.features.denseblock4, True)
            set_trainable(model.features.norm5, True)
            unfrozen.extend(["features.denseblock4", "features.norm5"])
    elif model_name == "torchvision:convnext_tiny":
        set_trainable(model.classifier, True)
        unfrozen.append("classifier")
        if freeze_mode == "last_block":
            set_trainable(model.features[-1], True)
            unfrozen.append("features[-1]")
    elif model_name == "torchvision:vit_b_16":
        set_trainable(model.heads, True)
        unfrozen.append("heads")
        if freeze_mode == "last_block":
            set_trainable(model.encoder.layers[-1], True)
            unfrozen.append("encoder.layers[-1]")
    else:
        raise ValueError(f"Freeze mode is not configured for model: {model_name}")

    return {
        "mode": freeze_mode,
        "unfrozen": unfrozen,
        "trainable_parameters": count_trainable_parameters(model),
    }


def count_trainable_parameters(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def autocast_context(torch_module: Any, enabled: bool, device: Any):
    if not enabled:
        return nullcontext()
    if hasattr(torch_module, "autocast"):
        return torch_module.autocast(device_type=device.type)
    return torch_module.cuda.amp.autocast()


def build_scaler(torch_module: Any, enabled: bool):
    if hasattr(torch_module, "amp") and hasattr(torch_module.amp, "GradScaler"):
        try:
            return torch_module.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch_module.amp.GradScaler(enabled=enabled)
    return torch_module.cuda.amp.GradScaler(enabled=enabled)


def run_epoch(
    *,
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
    torch_module: Any,
    optimizer: Any | None = None,
    scaler: Any | None = None,
    use_amp: bool = False,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    total = 0

    for batch_index, (images, labels) in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(torch_module, use_amp, device):
                logits = model(images)
                loss = criterion(logits, labels)
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            with torch_module.no_grad():
                with autocast_context(torch_module, use_amp, device):
                    logits = model(images)
                    loss = criterion(logits, labels)

        batch_size = int(labels.size(0))
        loss_sum += float(loss.detach().cpu()) * batch_size
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += batch_size

    if total == 0:
        return {"loss": 0.0, "accuracy": 0.0, "samples": 0}
    return {
        "loss": loss_sum / total,
        "accuracy": correct / total,
        "samples": total,
    }


def train(settings: TrainSettings, json_output: Path | None) -> dict[str, Any]:
    if settings.train_holdout_fraction < 0 or settings.train_holdout_fraction >= 1:
        raise ValueError("--train-holdout-fraction must be in the interval [0, 1)")
    if settings.label_smoothing < 0 or settings.label_smoothing >= 1:
        raise ValueError("--label-smoothing must be in the interval [0, 1)")

    validation_splits = (
        (settings.train_split,)
        if settings.train_holdout_fraction > 0
        else (settings.train_split, settings.val_split)
    )
    validation = validate_dataset_layout(
        settings.data_root,
        class_names=settings.classes,
        splits=validation_splits,
    )
    if not validation.ok:
        raise RuntimeError("Dataset validation failed; run scripts/check_dataset.py for details.")

    np_module, timm_module, torch_module, DataLoader, Subset, _WeightedRandomSampler, transforms_module = import_training_stack()
    set_seed(settings.seed, np_module, torch_module)
    device = choose_device(settings.device, torch_module)
    use_amp = settings.mixed_precision and device.type == "cuda"
    run_id = f"{timestamp()}_{settings.model}{'_smoke' if settings.smoke_test else ''}"

    train_dataset_full = XRayImageDataset(
        settings.data_root,
        settings.train_split,
        class_names=settings.classes,
        transform=build_transforms(settings, transforms_module, train=True),
    )
    holdout_manifest_path: Path | None = None
    holdout_summary: dict[str, Any] | None = None
    validation_source = settings.val_split
    class_weight_counts = dict(validation.counts[settings.train_split])

    if settings.train_holdout_fraction > 0:
        train_eval_dataset_full = XRayImageDataset(
            settings.data_root,
            settings.train_split,
            class_names=settings.classes,
            transform=build_transforms(settings, transforms_module, train=False),
        )
        train_indices, holdout_indices = stratified_holdout_indices(
            train_eval_dataset_full.samples,
            settings.train_holdout_fraction,
            seed=settings.holdout_seed,
        )
        train_dataset = Subset(train_dataset_full, train_indices)
        val_dataset = Subset(train_eval_dataset_full, holdout_indices)
        validation_source = f"{settings.train_split}_stratified_holdout"
        class_weight_counts = class_counts_from_indices(
            train_eval_dataset_full.samples,
            train_indices,
            settings.classes,
        )
        holdout_counts = class_counts_from_indices(
            train_eval_dataset_full.samples,
            holdout_indices,
            settings.classes,
        )
        holdout_manifest_path = (
            settings.holdout_manifest_output
            or PROJECT_ROOT / "outputs" / "splits" / f"holdout_{run_id}.csv"
        )
        write_holdout_manifest(
            holdout_manifest_path,
            settings.data_root,
            train_eval_dataset_full.samples,
            holdout_indices,
            settings.classes,
            settings.train_split,
        )
        holdout_summary = {
            "source_split": settings.train_split,
            "fraction": settings.train_holdout_fraction,
            "seed": settings.holdout_seed,
            "train_indices": len(train_indices),
            "holdout_indices": len(holdout_indices),
            "train_counts": class_weight_counts,
            "holdout_counts": holdout_counts,
            "manifest": holdout_manifest_path.as_posix(),
        }
    else:
        train_dataset = train_dataset_full
        train_indices = None
        val_dataset = XRayImageDataset(
            settings.data_root,
            settings.val_split,
            class_names=settings.classes,
            transform=build_transforms(settings, transforms_module, train=False),
        )

    train_sampler, domain_counts = build_domain_balanced_sampler(
        torch_module,
        train_dataset_full.samples,
        train_indices,
        settings.domain_balanced_prefixes,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=settings.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_classifier_model(
        settings.model,
        settings.pretrained,
        len(settings.classes),
        timm_module,
        torch_module,
    )
    init_checkpoint_summary: dict[str, Any] | None = None
    if settings.init_checkpoint is not None:
        checkpoint = load_checkpoint(torch_module, settings.init_checkpoint)
        checkpoint_classes = tuple(checkpoint.get("classes") or ())
        checkpoint_settings = dict(checkpoint.get("settings") or {})
        checkpoint_model = checkpoint_settings.get("model")
        if checkpoint_model and str(checkpoint_model) != settings.model:
            raise ValueError(
                "Init checkpoint model mismatch: "
                f"checkpoint has {checkpoint_model!r}, requested {settings.model!r}"
            )
        if checkpoint_classes and checkpoint_classes != settings.classes:
            raise ValueError(
                "Init checkpoint classes mismatch: "
                f"checkpoint has {checkpoint_classes!r}, requested {settings.classes!r}"
            )
        model.load_state_dict(checkpoint["model_state"])
        init_checkpoint_summary = {
            "path": settings.init_checkpoint.as_posix(),
            "epoch": checkpoint.get("epoch"),
            "source_model": checkpoint_model,
        }
    freeze_summary = apply_freeze_mode(model, settings.model, settings.freeze_mode)
    model.to(device)

    class_weights = (
        build_class_weights(class_weight_counts, settings.classes)
        if settings.use_class_weights
        else [1.0 for _ in settings.classes]
    )
    weights_tensor = torch_module.tensor(class_weights, dtype=torch_module.float32, device=device)
    criterion = torch_module.nn.CrossEntropyLoss(
        weight=weights_tensor,
        label_smoothing=settings.label_smoothing,
    )
    optimizer = build_optimizer(settings, torch_module, model)
    scheduler = build_scheduler(settings, torch_module, optimizer)
    scaler = build_scaler(torch_module, use_amp)

    checkpoint_dir = settings.checkpoints_dir / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    for epoch in range(1, settings.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            torch_module=torch_module,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            max_batches=settings.max_train_batches,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            torch_module=torch_module,
            use_amp=use_amp,
            max_batches=settings.max_val_batches,
        )
        if scheduler is not None:
            scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)
        print(
            "Epoch "
            f"{epoch}/{settings.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "classes": settings.classes,
            "settings": to_jsonable_settings(settings),
            "history": history,
        }
        torch_module.save(payload, last_path)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch_module.save(payload, best_path)

    report = {
        "ok": True,
        "run_id": run_id,
        "settings": to_jsonable_settings(settings),
        "dataset": validation.to_dict(),
        "validation_source": validation_source,
        "class_weight_counts": class_weight_counts,
        "domain_balanced_counts": domain_counts,
        "freeze": freeze_summary,
        "holdout": holdout_summary,
        "init_checkpoint": init_checkpoint_summary,
        "history": history,
        "best_checkpoint": best_path.as_posix(),
        "last_checkpoint": last_path.as_posix(),
    }
    output_path = json_output or settings.results_dir / f"train_{run_id}.json"
    write_json(output_path, report)
    write_json(settings.results_dir / "training_latest.json", report)
    print(f"Wrote training summary: {output_path.as_posix()}")
    print(f"Wrote latest training summary: {(settings.results_dir / 'training_latest.json').as_posix()}")
    return report


def main() -> int:
    args = parse_args()
    config_path = None if args.config.lower() == "none" else resolve_project_path(args.config)
    config = deep_merge(DEFAULT_CONFIG, load_yaml_config(config_path))
    settings = build_settings(args, config)

    if args.check_only:
        report = readiness_report(settings)
        print_readiness(report)
        if args.json_output is not None:
            output_path = resolve_project_path(args.json_output)
            write_json(output_path, report)
            print(f"Wrote readiness report: {output_path.as_posix()}")
        return 0 if report["ok"] else 1

    output_path = resolve_project_path(args.json_output) if args.json_output is not None else None
    train(settings, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
