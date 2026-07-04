"""Dataset utilities for the chest X-ray pneumonia project."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

DEFAULT_CLASSES = ("NORMAL", "PNEUMONIA")
DEFAULT_SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _as_tuple(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def list_image_files(
    directory: Path | str,
    extensions: Sequence[str] = IMAGE_EXTENSIONS,
) -> list[Path]:
    """Return image-like files below a directory in deterministic order."""
    directory = Path(directory)
    if not directory.exists():
        return []

    suffixes = {ext.lower() for ext in extensions}
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


@dataclass
class DatasetValidationResult:
    """Structured result from validating an ImageFolder-style dataset."""

    root: Path
    classes: tuple[str, ...]
    splits: tuple[str, ...]
    counts: dict[str, dict[str, int]]
    missing_dirs: list[str] = field(default_factory=list)
    empty_dirs: list[str] = field(default_factory=list)
    unexpected_class_dirs: dict[str, list[str]] = field(default_factory=dict)
    invalid_images: list[dict[str, str]] = field(default_factory=list)
    image_extensions: tuple[str, ...] = IMAGE_EXTENSIONS

    @property
    def total_images(self) -> int:
        return sum(sum(class_counts.values()) for class_counts in self.counts.values())

    @property
    def ok(self) -> bool:
        return not (
            self.missing_dirs
            or self.empty_dirs
            or self.unexpected_class_dirs
            or self.invalid_images
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "root": self.root.as_posix(),
            "classes": list(self.classes),
            "splits": list(self.splits),
            "counts": self.counts,
            "total_images": self.total_images,
            "missing_dirs": self.missing_dirs,
            "empty_dirs": self.empty_dirs,
            "unexpected_class_dirs": self.unexpected_class_dirs,
            "invalid_images": self.invalid_images,
            "image_extensions": list(self.image_extensions),
            "ok": self.ok,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def validate_dataset_layout(
    data_root: Path | str,
    class_names: Sequence[str] = DEFAULT_CLASSES,
    splits: Sequence[str] = DEFAULT_SPLITS,
    extensions: Sequence[str] = IMAGE_EXTENSIONS,
    verify_images: bool = False,
    require_non_empty: bool = True,
) -> DatasetValidationResult:
    """Validate the expected train/val/test and class directory layout."""
    root = Path(data_root)
    classes = _as_tuple(class_names)
    split_names = _as_tuple(splits)
    ext_names = _as_tuple(extensions)

    counts: dict[str, dict[str, int]] = {
        split: {class_name: 0 for class_name in classes} for split in split_names
    }
    missing_dirs: list[str] = []
    empty_dirs: list[str] = []
    unexpected_class_dirs: dict[str, list[str]] = {}
    invalid_images: list[dict[str, str]] = []

    if not root.exists():
        missing_dirs.append(root.as_posix())
        return DatasetValidationResult(
            root=root,
            classes=classes,
            splits=split_names,
            counts=counts,
            missing_dirs=missing_dirs,
            empty_dirs=empty_dirs,
            unexpected_class_dirs=unexpected_class_dirs,
            invalid_images=invalid_images,
            image_extensions=ext_names,
        )

    verifier = _verify_image if verify_images else None

    for split in split_names:
        split_dir = root / split
        if not split_dir.is_dir():
            missing_dirs.append(_relative(split_dir, root))
            continue

        unexpected = sorted(
            path.name
            for path in split_dir.iterdir()
            if path.is_dir() and path.name not in classes
        )
        if unexpected:
            unexpected_class_dirs[split] = unexpected

        for class_name in classes:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                missing_dirs.append(_relative(class_dir, root))
                continue

            files = list_image_files(class_dir, ext_names)
            counts[split][class_name] = len(files)
            if require_non_empty and not files:
                empty_dirs.append(_relative(class_dir, root))

            if verifier is not None:
                invalid_images.extend(
                    {"path": _relative(path, root), "error": error}
                    for path, error in verifier(files)
                )

    return DatasetValidationResult(
        root=root,
        classes=classes,
        splits=split_names,
        counts=counts,
        missing_dirs=missing_dirs,
        empty_dirs=empty_dirs,
        unexpected_class_dirs=unexpected_class_dirs,
        invalid_images=invalid_images,
        image_extensions=ext_names,
    )


def _verify_image(files: Sequence[Path]) -> list[tuple[Path, str]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required when verify_images=True") from exc

    invalid: list[tuple[Path, str]] = []
    for path in files:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:  # pragma: no cover - exact Pillow errors vary
            invalid.append((path, str(exc)))
    return invalid


def build_samples(
    data_root: Path | str,
    split: str,
    class_names: Sequence[str] = DEFAULT_CLASSES,
    extensions: Sequence[str] = IMAGE_EXTENSIONS,
) -> list[tuple[Path, int]]:
    """Build ``(image_path, class_index)`` samples for one split."""
    root = Path(data_root)
    samples: list[tuple[Path, int]] = []
    for class_index, class_name in enumerate(class_names):
        class_dir = root / split / class_name
        for path in list_image_files(class_dir, extensions):
            samples.append((path, class_index))
    return sorted(samples, key=lambda item: item[0].as_posix())


def stratified_holdout_indices(
    samples: Sequence[tuple[Path, int]],
    holdout_fraction: float,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split sample indices into train/holdout lists while preserving labels."""
    if holdout_fraction <= 0 or holdout_fraction >= 1:
        raise ValueError("holdout_fraction must be in the interval (0, 1)")

    by_label: dict[int, list[int]] = {}
    for index, (_, label) in enumerate(samples):
        by_label.setdefault(int(label), []).append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    holdout_indices: list[int] = []
    for label_indices in by_label.values():
        shuffled = list(label_indices)
        rng.shuffle(shuffled)
        holdout_count = max(1, round(len(shuffled) * holdout_fraction))
        holdout_count = min(holdout_count, len(shuffled) - 1)
        holdout_indices.extend(shuffled[:holdout_count])
        train_indices.extend(shuffled[holdout_count:])

    return sorted(train_indices), sorted(holdout_indices)


try:
    from torch.utils.data import Dataset as _TorchDataset
except Exception:  # pragma: no cover - used only when torch is unavailable
    _TorchDataset = object  # type: ignore[assignment]


class XRayImageDataset(_TorchDataset):
    """Minimal torch-compatible dataset for the expected X-ray folder layout."""

    def __init__(
        self,
        data_root: Path | str,
        split: str,
        class_names: Sequence[str] = DEFAULT_CLASSES,
        transform: Callable | None = None,
        extensions: Sequence[str] = IMAGE_EXTENSIONS,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.class_names = _as_tuple(class_names)
        self.class_to_idx = {
            class_name: class_index
            for class_index, class_name in enumerate(self.class_names)
        }
        self.transform = transform
        self.samples = build_samples(
            self.data_root,
            split,
            self.class_names,
            extensions,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required to load dataset images") from exc

        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label


class XRaySampleDataset(_TorchDataset):
    """Torch-compatible dataset built from an explicit sample list."""

    def __init__(
        self,
        data_root: Path | str,
        samples: Sequence[tuple[Path | str, int]],
        split: str = "manifest",
        class_names: Sequence[str] = DEFAULT_CLASSES,
        transform: Callable | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.class_names = _as_tuple(class_names)
        self.class_to_idx = {
            class_name: class_index
            for class_index, class_name in enumerate(self.class_names)
        }
        self.transform = transform
        self.samples = [
            (path if isinstance(path, Path) else self.data_root / str(path), int(label))
            for path, label in samples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required to load dataset images") from exc

        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label


def build_class_weights(
    counts: Mapping[str, int],
    class_names: Sequence[str] = DEFAULT_CLASSES,
) -> list[float]:
    """Return inverse-frequency weights normalized around 1.0."""
    values = [int(counts.get(class_name, 0)) for class_name in class_names]
    total = sum(values)
    if total <= 0 or any(value <= 0 for value in values):
        return [1.0 for _ in class_names]
    num_classes = len(values)
    return [total / (num_classes * value) for value in values]


def format_validation_report(result: DatasetValidationResult) -> str:
    """Create a compact human-readable dataset validation report."""
    lines = [
        f"Dataset root: {result.root.as_posix()}",
        f"Status: {'OK' if result.ok else 'INVALID'}",
        f"Total images: {result.total_images}",
        "",
        "Counts:",
    ]

    header = ["split", *result.classes, "total"]
    widths = [max(len(item), 5) for item in header]
    rows: list[list[str]] = []
    for split in result.splits:
        class_counts = result.counts[split]
        row = [
            split,
            *(str(class_counts[class_name]) for class_name in result.classes),
            str(sum(class_counts.values())),
        ]
        rows.append(row)
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    lines.append("  " + "  ".join(value.ljust(width) for value, width in zip(header, widths)))
    for row in rows:
        lines.append("  " + "  ".join(value.ljust(width) for value, width in zip(row, widths)))

    if result.missing_dirs:
        lines.extend(["", "Missing directories:"])
        lines.extend(f"  - {path}" for path in result.missing_dirs)

    if result.empty_dirs:
        lines.extend(["", "Empty class directories:"])
        lines.extend(f"  - {path}" for path in result.empty_dirs)

    if result.unexpected_class_dirs:
        lines.extend(["", "Unexpected class directories:"])
        for split, directories in sorted(result.unexpected_class_dirs.items()):
            lines.append(f"  - {split}: {', '.join(directories)}")

    if result.invalid_images:
        lines.extend(["", "Invalid images:"])
        for item in result.invalid_images[:20]:
            lines.append(f"  - {item['path']}: {item['error']}")
        if len(result.invalid_images) > 20:
            lines.append(f"  - ... {len(result.invalid_images) - 20} more")

    return "\n".join(lines)
