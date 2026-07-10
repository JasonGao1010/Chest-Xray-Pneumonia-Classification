"""Single-image inference utilities for the chest X-ray classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_CHECKPOINT = Path("models/vit_b16_best.pt")


def resolve_project_path(path: Path | str, project_root: Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    root = project_root or Path(__file__).resolve().parents[2]
    return root / path


def load_checkpoint(torch_module: Any, checkpoint_path: Path) -> dict[str, Any]:
    try:
        return torch_module.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(checkpoint_path, map_location="cpu")


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


def create_classifier_model(
    model_name: str,
    num_classes: int,
    timm_module: Any,
) -> Any:
    if model_name in {"torchvision:vit_b_16", "vit_b16"}:
        from torchvision.models import vit_b_16

        return vit_b_16(weights=None, num_classes=num_classes)

    if model_name in {"torchvision:densenet121", "densenet121"}:
        from torchvision.models import densenet121

        return densenet121(weights=None, num_classes=num_classes)

    if model_name in {"torchvision:convnext_tiny", "convnext_tiny"}:
        from torchvision.models import convnext_tiny

        return convnext_tiny(weights=None, num_classes=num_classes)

    return timm_module.create_model(model_name, pretrained=False, num_classes=num_classes)


@dataclass
class XRayPredictor:
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT
    device: str = "auto"
    threshold: float = 0.5
    project_root: Path | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in the interval [0, 1]")
        import timm
        import torch
        from torchvision import transforms

        self._torch = torch
        self.checkpoint_path = resolve_project_path(self.checkpoint_path, self.project_root)
        checkpoint = load_checkpoint(torch, self.checkpoint_path)
        self.settings = dict(checkpoint.get("settings") or {})
        self.class_names = tuple(checkpoint.get("classes") or self.settings.get("classes") or ("NORMAL", "PNEUMONIA"))
        if len(self.class_names) != 2 or "PNEUMONIA" not in self.class_names:
            raise ValueError(
                "The web predictor requires exactly two classes including PNEUMONIA"
            )
        self.model_name = str(self.settings.get("model", "tf_efficientnetv2_s.in1k"))
        self.image_size = int(self.settings.get("image_size", 224))
        self.device_obj = choose_device(self.device, torch)
        self.transform = build_eval_transform(self.settings, transforms)
        self.positive_index = self.class_names.index("PNEUMONIA")
        self.negative_index = 1 - self.positive_index

        model = create_classifier_model(self.model_name, len(self.class_names), timm)
        model.load_state_dict(checkpoint["model_state"])
        model.to(self.device_obj)
        model.eval()
        self.model = model

    def predict_image(self, image: Image.Image) -> dict[str, Any]:
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device_obj)
        with self._torch.no_grad():
            logits = self.model(tensor)
            probabilities_tensor = self._torch.softmax(logits, dim=1)[0].detach().cpu()

        probabilities = {
            class_name: float(probabilities_tensor[index].item())
            for index, class_name in enumerate(self.class_names)
        }
        pneumonia_probability = float(probabilities_tensor[self.positive_index].item())
        predicted_index = self.positive_index if pneumonia_probability >= self.threshold else self.negative_index
        predicted_label = self.class_names[predicted_index]

        return {
            "predicted_label": predicted_label,
            "predicted_label_cn": (
                "PNEUMONIA 标签倾向"
                if predicted_label == "PNEUMONIA"
                else "NORMAL 标签倾向"
            ),
            "probabilities": probabilities,
            "normal_probability": probabilities.get("NORMAL"),
            "pneumonia_probability": pneumonia_probability,
            "threshold": self.threshold,
            "confidence": max(probabilities.values()) if probabilities else None,
            "model": self.model_name,
            "checkpoint": Path(self.checkpoint_path).as_posix(),
            "device": str(self.device_obj),
            "image_size": self.image_size,
        }
