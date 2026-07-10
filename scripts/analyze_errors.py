#!/usr/bin/env python3
"""Export error cases and optional image grids from prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xray_pneumonia.calibration import load_binary_predictions  # noqa: E402


def resolve_project_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def classify_error(
    true_label: str,
    score: float,
    positive_class: str,
    threshold: float,
    high_confidence: float,
) -> tuple[str, str]:
    predicted_positive = score >= threshold
    actual_positive = true_label == positive_class
    if predicted_positive == actual_positive:
        if 0.4 <= score <= 0.6:
            return "correct_borderline", "correct"
        return "correct", "correct"
    if predicted_positive and not actual_positive:
        if score >= high_confidence:
            return "high_confidence_false_positive", "false_positive"
        return "false_positive", "false_positive"
    if (not predicted_positive) and actual_positive:
        if (1.0 - score) >= high_confidence:
            return "high_confidence_false_negative", "false_negative"
        return "false_negative", "false_negative"
    return "unknown", "unknown"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "true_label",
        "predicted_label",
        "prob_pneumonia",
        "confidence",
        "error_type",
        "coarse_type",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_grid(path: Path, data_root: Path, rows: list[dict[str, Any]], title: str, limit: int) -> int:
    from PIL import Image
    import matplotlib.pyplot as plt

    selected = rows[:limit]
    if not selected:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = min(4, len(selected))
    rows_count = (len(selected) + columns - 1) // columns
    fig, axes = plt.subplots(rows_count, columns, figsize=(columns * 3, rows_count * 3))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    rendered = 0
    for axis, row in zip(axes_list, selected):
        image_path = Path(str(row["path"]))
        if not image_path.is_absolute():
            image_path = data_root / image_path
        try:
            with Image.open(image_path) as image:
                axis.imshow(image.convert("L"), cmap="gray")
            rendered += 1
        except Exception:
            axis.text(0.5, 0.5, "image missing", ha="center", va="center")
        axis.set_title(
            f"{row['error_type']}\nP={float(row['prob_pneumonia']):.3f}",
            fontsize=8,
        )
        axis.axis("off")
    for axis in axes_list[len(selected):]:
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return rendered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze false positives, false negatives, and borderline samples.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/chest_xray"))
    parser.add_argument("--positive-class", type=str, default="PNEUMONIA")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--high-confidence", type=float, default=0.9)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--grid-output", type=Path, default=None)
    parser.add_argument("--grid-limit", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions_path = resolve_project_path(args.predictions)
    data_root = resolve_project_path(args.data_root)
    records = load_binary_predictions(predictions_path, positive_class=args.positive_class)

    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {
        "correct": 0,
        "false_positive": 0,
        "false_negative": 0,
        "high_confidence_false_positive": 0,
        "high_confidence_false_negative": 0,
        "borderline": 0,
    }
    for record in records:
        error_type, coarse_type = classify_error(
            record.true_label,
            record.score,
            args.positive_class,
            args.threshold,
            args.high_confidence,
        )
        if 0.4 <= record.score <= 0.6:
            counts["borderline"] += 1
            if error_type == "correct":
                error_type = "correct_borderline"
        if coarse_type in {"false_positive", "false_negative"}:
            counts[coarse_type] += 1
        if error_type in {"high_confidence_false_positive", "high_confidence_false_negative"}:
            counts[error_type] += 1
        elif error_type == "correct":
            counts["correct"] += 1
        confidence = record.score if record.score >= args.threshold else 1.0 - record.score
        rows.append(
            {
                "path": record.path,
                "true_label": record.true_label,
                "predicted_label": record.predicted_label,
                "prob_pneumonia": f"{record.score:.8f}",
                "confidence": f"{confidence:.8f}",
                "error_type": error_type,
                "coarse_type": coarse_type,
            }
        )

    report_rows = [
        row
        for row in rows
        if row["coarse_type"] in {"false_positive", "false_negative"} or "borderline" in row["error_type"]
    ]
    report_rows.sort(key=lambda row: (row["coarse_type"], -float(row["confidence"])))
    csv_output = resolve_project_path(args.csv_output)
    json_output = resolve_project_path(args.json_output)
    write_rows(csv_output, report_rows)

    grid_rendered = None
    if args.grid_output is not None:
        grid_rendered = write_grid(
            resolve_project_path(args.grid_output),
            data_root,
            report_rows,
            title=f"Error Cases ({predictions_path.stem})",
            limit=args.grid_limit,
        )

    payload = {
        "ok": True,
        "predictions": predictions_path.as_posix(),
        "threshold": args.threshold,
        "positive_class": args.positive_class,
        "sample_count": len(records),
        "exported_case_count": len(report_rows),
        "counts": counts,
        "csv_output": csv_output.as_posix(),
        "grid_rendered": grid_rendered,
    }
    write_json(json_output, payload)
    print(
        "Errors "
        f"FP={counts['false_positive']} "
        f"FN={counts['false_negative']} "
        f"borderline={counts['borderline']}"
    )
    print(f"Wrote error cases: {csv_output.as_posix()}")
    print(f"Wrote error report: {json_output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
