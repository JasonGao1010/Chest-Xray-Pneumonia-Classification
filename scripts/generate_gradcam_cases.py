#!/usr/bin/env python3
"""Generate simple Grad-CAM grids for CNN checkpoints and prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.evaluate import build_eval_transform, create_classifier_model, load_checkpoint  # noqa: E402


def resolve_project_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_rows(path: Path, case_type: str, limit: int) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if case_type == "errors":
        rows = [row for row in rows if row["true_label"] != row["predicted_label"]]
    elif case_type == "true_positive":
        rows = [
            row for row in rows
            if row["true_label"] == "PNEUMONIA" and row["predicted_label"] == "PNEUMONIA"
        ]
    elif case_type == "false_positive":
        rows = [
            row for row in rows
            if row["true_label"] == "NORMAL" and row["predicted_label"] == "PNEUMONIA"
        ]
    elif case_type == "false_negative":
        rows = [
            row for row in rows
            if row["true_label"] == "PNEUMONIA" and row["predicted_label"] == "NORMAL"
        ]
    rows.sort(key=lambda row: max(float(row.get("prob_NORMAL", 0)), float(row.get("prob_PNEUMONIA", 0))), reverse=True)
    return rows[:limit]


def target_layer(model: Any, model_name: str) -> Any:
    if model_name == "torchvision:densenet121":
        return model.features.denseblock4
    if model_name == "torchvision:convnext_tiny":
        return model.features
    raise ValueError(f"Grad-CAM target layer is not configured for {model_name}")


def make_gradcam(model: Any, layer: Any, image_tensor: Any, class_index: int, torch_module: Any) -> Any:
    activations: list[Any] = []
    gradients: list[Any] = []

    def forward_hook(_module: Any, _inputs: Any, output: Any) -> None:
        activations.append(output.detach())

    def backward_hook(_module: Any, _grad_input: Any, grad_output: Any) -> None:
        gradients.append(grad_output[0].detach())

    handle_forward = layer.register_forward_hook(forward_hook)
    handle_backward = layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image_tensor)
        score = logits[:, class_index].sum()
        score.backward()
        activation = activations[-1]
        gradient = gradients[-1]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = torch_module.relu(cam)
        cam = torch_module.nn.functional.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam[0, 0]
        cam = cam - cam.min()
        if float(cam.max()) > 0:
            cam = cam / cam.max()
        return cam.detach().cpu()
    finally:
        handle_forward.remove()
        handle_backward.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM grid for selected prediction cases.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/chest_xray"))
    parser.add_argument("--case-type", choices=["errors", "true_positive", "false_positive", "false_negative"], default="errors")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    import matplotlib.pyplot as plt
    from PIL import Image
    import timm
    import torch
    from torchvision import transforms

    args = parse_args()
    checkpoint_path = resolve_project_path(args.checkpoint)
    predictions_path = resolve_project_path(args.predictions)
    data_root = resolve_project_path(args.data_root)
    output = resolve_project_path(args.output)

    checkpoint = load_checkpoint(torch, checkpoint_path)
    settings = dict(checkpoint.get("settings") or {})
    class_names = tuple(checkpoint.get("classes") or settings.get("classes") or ("NORMAL", "PNEUMONIA"))
    model_name = str(settings.get("model"))
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = create_classifier_model(model_name, len(class_names), timm, torch)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    layer = target_layer(model, model_name)
    transform = build_eval_transform(settings, transforms)
    rows = load_rows(predictions_path, args.case_type, args.limit)
    if not rows:
        raise RuntimeError(f"No rows selected for case type {args.case_type}")

    columns = min(4, len(rows))
    rows_count = (len(rows) + columns - 1) // columns
    fig, axes = plt.subplots(rows_count, columns, figsize=(columns * 3.2, rows_count * 3.2))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for axis, row in zip(axes_list, rows):
        image_path = Path(row["path"])
        if not image_path.is_absolute():
            image_path = data_root / image_path
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            tensor = transform(rgb).unsqueeze(0).to(device)
            class_index = class_names.index(row["predicted_label"])
            cam = make_gradcam(model, layer, tensor, class_index, torch)
            axis.imshow(rgb.resize((224, 224)), cmap="gray")
            axis.imshow(cam.numpy(), cmap="jet", alpha=0.38)
        axis.set_title(
            f"{row['true_label']}->{row['predicted_label']}\nP={float(row['prob_PNEUMONIA']):.3f}",
            fontsize=8,
        )
        axis.axis("off")
    for axis in axes_list[len(rows):]:
        axis.axis("off")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Wrote Grad-CAM grid: {output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
