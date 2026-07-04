#!/usr/bin/env python3
"""Evaluate a saved chest X-ray classifier checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xray_pneumonia.data import DEFAULT_CLASSES, XRayImageDataset, XRaySampleDataset, validate_dataset_layout


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_project_path(path: Path | str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_checkpoint(torch_module: Any, checkpoint_path: Path) -> dict[str, Any]:
    try:
        return torch_module.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(checkpoint_path, map_location="cpu")


def latest_checkpoint_from_training_summary(path: Path) -> Path:
    summary = load_json(path)
    checkpoint = summary.get("best_checkpoint") or summary.get("last_checkpoint")
    if not checkpoint:
        raise ValueError(f"Training summary has no checkpoint path: {path}")
    return Path(checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained X-ray checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint .pt file. Defaults to best checkpoint in results/training_latest.json.",
    )
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=Path("results/training_latest.json"),
        help="Training summary used when --checkpoint is omitted.",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--samples-csv",
        type=Path,
        default=None,
        help=(
            "Optional explicit sample manifest with columns path,true_label. "
            "When set, --split is used only for naming outputs."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--confusion-matrix-output", type=Path, default=None)
    parser.add_argument(
        "--no-update-latest",
        action="store_true",
        help="Do not overwrite results/evaluation_latest.json.",
    )
    return parser.parse_args()


def choose_device(requested: str, torch_module: Any):
    if requested != "auto":
        return torch_module.device(requested)
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def build_eval_transform(settings: dict[str, Any], transforms_module: Any):
    image_size = int(settings.get("image_size", 224))
    normalize = str(settings.get("normalize", "imagenet")).lower()
    steps: list[Any] = [
        transforms_module.Resize((image_size, image_size)),
        transforms_module.ToTensor(),
    ]
    if normalize == "imagenet":
        steps.append(
            transforms_module.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        )
    return transforms_module.Compose(steps)


def load_samples_csv(path: Path, data_root: Path, class_names: tuple[str, ...]) -> list[tuple[Path, int]]:
    class_to_idx = {class_name: index for index, class_name in enumerate(class_names)}
    samples: list[tuple[Path, int]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "true_label"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required sample columns in {path}: {sorted(missing)}")
        for row in reader:
            label = str(row["true_label"])
            if label not in class_to_idx:
                raise ValueError(f"Unknown label {label!r} in {path}")
            sample_path = Path(str(row["path"]))
            if not sample_path.is_absolute():
                sample_path = data_root / sample_path
            samples.append((sample_path, class_to_idx[label]))
    if not samples:
        raise ValueError(f"No sample rows found in {path}")
    return samples


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    positive_scores: list[float],
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    positive_label = min(1, len(class_names) - 1)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist(),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, positive_scores))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def create_classifier_model(
    model_name: str,
    num_classes: int,
    timm_module: Any,
    torch_module: Any,
) -> Any:
    if model_name == "torchvision:vit_b_16":
        from torchvision.models import vit_b_16

        return vit_b_16(weights=None, num_classes=num_classes)

    return timm_module.create_model(model_name, pretrained=False, num_classes=num_classes)


def write_predictions(path: Path, rows: list[dict[str, Any]], class_names: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "true_label",
        "predicted_label",
        *[f"prob_{class_name}" for class_name in class_names],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrix(path: Path, matrix: list[list[int]], class_names: tuple[str, ...]) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import timm
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    checkpoint_path = resolve_project_path(args.checkpoint) if args.checkpoint else latest_checkpoint_from_training_summary(
        resolve_project_path(args.training_summary)
    )
    checkpoint = load_checkpoint(torch, checkpoint_path)
    settings = dict(checkpoint.get("settings") or {})
    class_names = tuple(checkpoint.get("classes") or settings.get("classes") or DEFAULT_CLASSES)
    data_root = resolve_project_path(args.data_root or settings.get("data_root", "data/raw/chest_xray"))

    device = choose_device(args.device, torch)
    samples_csv = resolve_project_path(args.samples_csv) if args.samples_csv is not None else None
    if samples_csv is not None:
        explicit_samples = load_samples_csv(samples_csv, data_root, class_names)
        dataset = XRaySampleDataset(
            data_root,
            explicit_samples,
            split=args.split,
            class_names=class_names,
            transform=build_eval_transform(settings, transforms),
        )
        dataset_report: dict[str, Any] = {
            "ok": True,
            "root": data_root.as_posix(),
            "samples_csv": samples_csv.as_posix(),
            "sample_count": len(dataset),
        }
    else:
        validation = validate_dataset_layout(data_root, class_names=class_names, splits=(args.split,))
        if not validation.ok:
            raise RuntimeError(f"Dataset validation failed for split {args.split}: {validation.to_dict()}")
        dataset = XRayImageDataset(
            data_root,
            args.split,
            class_names=class_names,
            transform=build_eval_transform(settings, transforms),
        )
        dataset_report = validation.to_dict()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model_name = str(settings.get("model", "tf_efficientnetv2_s.in1k"))
    model = create_classifier_model(model_name, len(class_names), timm, torch)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    positive_scores: list[float] = []
    prediction_rows: list[dict[str, Any]] = []
    sample_index = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            labels_cpu = labels.cpu().tolist()
            predictions_cpu = predictions.cpu().tolist()
            probabilities_cpu = probabilities.cpu().tolist()
            for label, prediction, probs in zip(labels_cpu, predictions_cpu, probabilities_cpu):
                image_path, _ = dataset.samples[sample_index]
                row = {
                    "path": image_path.relative_to(data_root).as_posix(),
                    "true_label": class_names[label],
                    "predicted_label": class_names[prediction],
                }
                row.update(
                    {
                        f"prob_{class_name}": f"{float(probability):.8f}"
                        for class_name, probability in zip(class_names, probs)
                    }
                )
                prediction_rows.append(row)
                y_true.append(int(label))
                y_pred.append(int(prediction))
                positive_scores.append(float(probs[min(1, len(class_names) - 1)]))
                sample_index += 1

    metrics = compute_metrics(y_true, y_pred, positive_scores, class_names)
    run_id = checkpoint_path.parent.name or timestamp()
    json_output = resolve_project_path(args.json_output or f"results/eval_{args.split}_{run_id}.json")
    predictions_output = resolve_project_path(
        args.predictions_output or f"results/predictions_{args.split}_{run_id}.csv"
    )
    confusion_matrix_output = resolve_project_path(
        args.confusion_matrix_output or f"figures/confusion_matrix_{args.split}_{run_id}.png"
    )

    report = {
        "ok": True,
        "checkpoint": checkpoint_path.as_posix(),
        "split": args.split,
        "classes": list(class_names),
        "model": model_name,
        "device": str(device),
        "sample_count": len(dataset),
        "dataset": dataset_report,
        "metrics": metrics,
        "predictions_output": predictions_output.as_posix(),
        "confusion_matrix_output": confusion_matrix_output.as_posix(),
    }
    write_predictions(predictions_output, prediction_rows, class_names)
    write_confusion_matrix(confusion_matrix_output, metrics["confusion_matrix"], class_names)
    write_json(json_output, report)
    if not args.no_update_latest:
        write_json(PROJECT_ROOT / "results" / "evaluation_latest.json", report)
    print(
        "Evaluation "
        f"{args.split}: accuracy={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"roc_auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'NA'}"
    )
    print(f"Wrote evaluation summary: {json_output.as_posix()}")
    print(f"Wrote predictions: {predictions_output.as_posix()}")
    print(f"Wrote confusion matrix: {confusion_matrix_output.as_posix()}")
    return report


def main() -> int:
    args = parse_args()
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
