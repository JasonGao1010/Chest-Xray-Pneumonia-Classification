#!/usr/bin/env python3
"""Hard-gate a rebuilt dataset and result package against the published evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_METRICS = ("balanced_accuracy", "recall", "specificity", "roc_auc", "auprc")
RELATIVE_TOLERANCE = 0.05


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def relative_difference(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("rebuild"))
    parser.add_argument("--reference", type=Path, default=Path("results/strict_summary.json"))
    parser.add_argument("--tolerance", type=float, default=RELATIVE_TOLERANCE)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def keyed_groups(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["dataset"], row["model"]): row for row in payload["groups"]}


def keyed_comparisons(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["candidate_family"], row["dataset"]): row for row in payload["paired_comparisons"]}


def main() -> int:
    args = parse_args()
    work = args.work_dir if args.work_dir.is_absolute() else ROOT / args.work_dir
    reference_path = args.reference if args.reference.is_absolute() else ROOT / args.reference
    rebuilt = load(work / "results/strict_summary.json")
    reference = load(reference_path)
    kermany = load(work / "audit/kermany_grouped_summary.json")
    rsna = load(work / "audit/rsna_dataset_summary.json")
    failures: list[str] = []

    if kermany.get("total_images") != 5856 or not kermany.get("ok"):
        failures.append("Kermany identity gate failed")
    if rsna.get("sample_count") != 1707 or rsna.get("hash_mismatches") or rsna.get("missing_images"):
        failures.append("RSNA 1707-member identity gate failed")
    required_secondary: list[Path] = [work / "results/domain_shift_diagnostic.json"]
    for family in ("strict", "mixed_simple"):
        for dataset in ("kermany_grouped", "rsna"):
            for seed in (42, 43, 44):
                required_secondary.extend([
                    work / f"results/{family}_calibration_{dataset}_densenet121_seed{seed}.json",
                    work / f"results/{family}_operating_points_{dataset}_densenet121_seed{seed}.json",
                ])
            required_secondary.extend([
                work / f"results/{family}_error_cases_{dataset}_densenet121_seed42.csv",
                work / f"results/{family}_error_summary_{dataset}_densenet121_seed42.json",
            ])
    missing_secondary = [path.relative_to(work).as_posix() for path in required_secondary if not path.is_file()]
    if missing_secondary:
        failures.append(f"Missing secondary analysis artifacts: {missing_secondary[:5]}")

    new_groups, old_groups = keyed_groups(rebuilt), keyed_groups(reference)
    if set(new_groups) != set(old_groups) or len(new_groups) != 6:
        failures.append("Strict result matrix is incomplete or has unexpected groups")
    comparisons: list[dict[str, Any]] = []
    for key in sorted(set(new_groups) & set(old_groups)):
        for field in ("sample_count", "bootstrap_group_count"):
            if new_groups[key].get(field) != old_groups[key].get(field):
                failures.append(f"Identity count mismatch: {key}/{field}")
        for metric in PRIMARY_METRICS:
            new = float(new_groups[key]["ensemble"][metric])
            old = float(old_groups[key]["ensemble"][metric])
            difference = relative_difference(new, old)
            comparisons.append({"group": list(key), "metric": metric, "reference": old, "rebuilt": new, "relative_difference": difference})
            if difference >= args.tolerance:
                failures.append(f"Metric outside tolerance: {key}/{metric} ({difference:.4%})")

    new_candidates, old_candidates = keyed_comparisons(rebuilt), keyed_comparisons(reference)
    if set(new_candidates) != set(old_candidates) or len(new_candidates) != 6:
        failures.append("Candidate comparison matrix is incomplete or unexpected")
    for key in sorted(set(new_candidates) & set(old_candidates)):
        for field in ("sample_count", "bootstrap_group_count"):
            if new_candidates[key].get(field) != old_candidates[key].get(field):
                failures.append(f"Candidate identity count mismatch: {key}/{field}")
        for metric in PRIMARY_METRICS:
            new = float(new_candidates[key]["candidate_ensemble"][metric])
            old = float(old_candidates[key]["candidate_ensemble"][metric])
            difference = relative_difference(new, old)
            comparisons.append({"group": list(key), "metric": metric, "reference": old, "rebuilt": new, "relative_difference": difference})
            if difference >= args.tolerance:
                failures.append(f"Candidate metric outside tolerance: {key}/{metric} ({difference:.4%})")

    report = {
        "status": "VERIFIED" if not failures else "FAILED",
        "data_identity": {"kermany_images": kermany.get("total_images"), "rsna_members": rsna.get("sample_count")},
        "metric_tolerance": args.tolerance,
        "secondary_artifact_count": len(required_secondary),
        "metric_comparisons": comparisons,
        "failures": failures,
    }
    output = args.output or work / "reproduction_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
