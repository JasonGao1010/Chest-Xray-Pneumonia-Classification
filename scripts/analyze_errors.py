#!/usr/bin/env python3
"""Export error cases and optional image grids from prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xray_pneumonia.calibration import load_binary_predictions  # noqa: E402
from xray_pneumonia.plotting import configure_report_figure_typography  # noqa: E402


def resolve_project_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


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
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def select_grid_cases(
    rows: list[dict[str, Any]],
    limit: int,
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic, FP/FN-balanced grid with explicit quotas.

    The four primary buckets are high-confidence FP/FN and closest-threshold
    FP/FN.  A case selected in an extreme bucket is removed before the near-
    threshold bucket is filled.  Any shortage is redistributed to remaining
    errors; correct borderline cases are used only if fewer errors than slots
    exist.
    """
    bucket_names = (
        "extreme_false_positive",
        "extreme_false_negative",
        "near_threshold_false_positive",
        "near_threshold_false_negative",
    )
    base, remainder = divmod(max(limit, 0), len(bucket_names))
    quota = {
        name: base + int(index < remainder)
        for index, name in enumerate(bucket_names)
    }

    false_positive = [row for row in rows if row["coarse_type"] == "false_positive"]
    false_negative = [row for row in rows if row["coarse_type"] == "false_negative"]
    borderline_correct = [
        row for row in rows if row["coarse_type"] == "correct" and "borderline" in row["error_type"]
    ]
    candidates = {
        "extreme_false_positive": sorted(
            false_positive,
            key=lambda row: (-float(row["prob_pneumonia"]), str(row["path"])),
        ),
        "extreme_false_negative": sorted(
            false_negative,
            key=lambda row: (float(row["prob_pneumonia"]), str(row["path"])),
        ),
        "near_threshold_false_positive": sorted(
            false_positive,
            key=lambda row: (
                abs(float(row["prob_pneumonia"]) - threshold),
                str(row["path"]),
            ),
        ),
        "near_threshold_false_negative": sorted(
            false_negative,
            key=lambda row: (
                abs(float(row["prob_pneumonia"]) - threshold),
                str(row["path"]),
            ),
        ),
    }

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str]] = set()
    selected_by_bucket: Counter[str] = Counter()

    def add(row: dict[str, Any], bucket: str) -> bool:
        key = (str(row["path"]), str(row["coarse_type"]))
        if key in selected_keys or len(selected) >= limit:
            return False
        selected.append(row)
        selected_keys.add(key)
        selected_by_bucket[bucket] += 1
        return True

    for bucket in bucket_names:
        for row in candidates[bucket]:
            if selected_by_bucket[bucket] >= quota[bucket]:
                break
            add(row, bucket)

    remaining_errors = sorted(
        false_positive + false_negative,
        key=lambda row: (
            abs(float(row["prob_pneumonia"]) - threshold),
            row["coarse_type"],
            str(row["path"]),
        ),
    )
    for row in remaining_errors:
        if len(selected) >= limit:
            break
        add(row, "redistributed_remaining_error")

    for row in sorted(
        borderline_correct,
        key=lambda item: (
            abs(float(item["prob_pneumonia"]) - threshold),
            str(item["path"]),
        ),
    ):
        if len(selected) >= limit:
            break
        add(row, "correct_borderline_fallback")

    selected_coarse = Counter(str(row["coarse_type"]) for row in selected)
    metadata = {
        "policy_version": 1,
        "policy": (
            "deterministic four-bucket quota: extreme FP, extreme FN, nearest-threshold "
            "FP, nearest-threshold FN; redistribute shortages to remaining errors; "
            "use correct borderline cases only after exhausting errors"
        ),
        "requested_limit": limit,
        "quota_targets": quota,
        "selected_count": len(selected),
        "selected_by_bucket": dict(selected_by_bucket),
        "selected_by_coarse_type": dict(selected_coarse),
        "selected_paths": [str(row["path"]) for row in selected],
    }
    return selected, metadata


def write_grid(path: Path, data_root: Path, rows: list[dict[str, Any]], title: str) -> int:
    from PIL import Image
    import matplotlib as mpl

    configure_report_figure_typography(mpl)
    import matplotlib.pyplot as plt

    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = min(4, len(rows))
    rows_count = (len(rows) + columns - 1) // columns
    fig, axes = plt.subplots(rows_count, columns, figsize=(columns * 3, rows_count * 3))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    rendered = 0
    for axis, row in zip(axes_list, rows):
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
    for axis in axes_list[len(rows):]:
        axis.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
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
    negative_labels = sorted(
        {record.true_label for record in records if record.true_label != args.positive_class}
    )
    if len(negative_labels) != 1:
        raise ValueError(
            "Error analysis requires exactly one negative class in addition to the positive class"
        )
    negative_class = negative_labels[0]

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
        elif coarse_type == "correct":
            counts["correct"] += 1
        if error_type in {"high_confidence_false_positive", "high_confidence_false_negative"}:
            counts[error_type] += 1
        confidence = record.score if record.score >= args.threshold else 1.0 - record.score
        predicted_label = (
            args.positive_class
            if record.score >= args.threshold
            else negative_class
        )
        rows.append(
            {
                "path": record.path,
                "true_label": record.true_label,
                "predicted_label": predicted_label,
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
    report_rows.sort(
        key=lambda row: (
            0 if row["coarse_type"] in {"false_positive", "false_negative"} else 1,
            row["coarse_type"],
            -float(row["confidence"]),
            str(row["path"]),
        )
    )
    csv_output = resolve_project_path(args.csv_output)
    json_output = resolve_project_path(args.json_output)
    write_rows(csv_output, report_rows)

    grid_rendered = None
    grid_selection = None
    grid_output = None
    if args.grid_output is not None:
        grid_output = resolve_project_path(args.grid_output)
        selected, grid_selection = select_grid_cases(
            report_rows,
            limit=args.grid_limit,
            threshold=args.threshold,
        )
        grid_rendered = write_grid(
            grid_output,
            data_root,
            selected,
            title=f"Error Cases ({predictions_path.stem})",
        )

    payload = {
        "ok": True,
        "predictions": display_path(predictions_path),
        "threshold": args.threshold,
        "positive_class": args.positive_class,
        "sample_count": len(records),
        "exported_case_count": len(report_rows),
        "counts": counts,
        "csv_output": display_path(csv_output),
        "grid_output": display_path(grid_output) if grid_output else None,
        "grid_rendered": grid_rendered,
        "grid_selection": grid_selection,
        "grid_rendering": {
            "font_family": "Times New Roman",
            "dpi": 300,
            "image_mode": "grayscale",
            "crop": "none",
            "intensity_adjustment": "none",
        }
        if grid_output
        else None,
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
