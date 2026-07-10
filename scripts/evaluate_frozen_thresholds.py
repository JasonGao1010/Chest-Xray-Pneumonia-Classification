#!/usr/bin/env python3
"""Select operating points on validation predictions and freeze them on test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xray_pneumonia.thresholds import (  # noqa: E402
    compute_threshold_metrics,
    load_prediction_csv,
    make_threshold_grid,
    select_best,
    select_best_with_recall_floor,
    threshold_curve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=float, default=0.001)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validation = load_prediction_csv(PROJECT_ROOT / args.validation)
    test = load_prediction_csv(PROJECT_ROOT / args.test)
    curve = threshold_curve(validation, make_threshold_grid(args.step))
    selected = {
        "balanced_accuracy": select_best(curve, "balanced_accuracy"),
        "recall_at_least_0.90": select_best_with_recall_floor(curve, 0.90),
        "recall_at_least_0.95": select_best_with_recall_floor(curve, 0.95),
    }
    payload = {"validation": args.validation.as_posix(), "test": args.test.as_posix(), "selection_split": "validation", "operating_points": {}}
    for name, val_metrics in selected.items():
        if val_metrics is None:
            continue
        test_metrics = compute_threshold_metrics(test, val_metrics.threshold)
        payload["operating_points"][name] = {"threshold": val_metrics.threshold, "validation": val_metrics.to_dict(), "test": test_metrics.to_dict()}
    output = PROJECT_ROOT / args.output
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
