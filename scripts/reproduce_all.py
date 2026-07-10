#!/usr/bin/env python3
"""Run the complete, frozen data-to-main-table reproduction workflow."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "densenet121": "configs/densenet121.yaml",
    "convnext_tiny": "configs/convnext_tiny.yaml",
    "vit_b16": "configs/vit_b16.yaml",
}
SEEDS = (42, 43, 44)
FAMILIES = ("strict", "robust", "mixed_simple", "mixed_domain_balanced")


def command(*parts: object) -> list[str]:
    return [sys.executable, *map(str, parts)]


def run(cmd: list[str], *, dry_run: bool) -> None:
    print("+", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["data", "experiments", "analyze", "verify", "all"], default="all")
    parser.add_argument("--kermany-raw", type=Path, default=Path("data/raw/chest_xray"))
    parser.add_argument("--rsna-raw", type=Path, default=Path("data/raw/rsna_pneumonia"))
    parser.add_argument("--work-dir", type=Path, default=Path("rebuild"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Allow data-stage replacement inside --work-dir only.")
    return parser.parse_args()


def data_stage(args: argparse.Namespace) -> None:
    work = args.work_dir
    force = ["--force"] if args.force else []
    run(command("scripts/prepare_kermany_grouped.py", "--source-root", args.kermany_raw,
                "--output-root", work / "data/kermany_grouped_seed42", "--from-manifest",
                "--manifest", "data/splits/kermany_grouped_seed42.csv", "--summary",
                work / "audit/kermany_grouped_summary.json", *force), dry_run=args.dry_run)
    run(command("scripts/prepare_rsna_binary.py", "--raw-root", args.rsna_raw,
                "--images-dir", args.rsna_raw / "stage_2_train_images",
                "--member-manifest", "data/splits/rsna_available_1707_manifest.csv",
                "--output-root", work / "data/rsna_binary", "--splits-output",
                work / "audit/rsna_splits.json", "--summary-output",
                work / "audit/rsna_dataset_summary.json", "--figure-output",
                work / "figures/rsna_class_distribution.png", *force), dry_run=args.dry_run)
    run(command("scripts/prepare_mixed_binary.py", "--source-a", work / "data/kermany_grouped_seed42",
                "--source-b", work / "data/rsna_binary", "--prefix-a", "kermany",
                "--prefix-b", "rsna", "--output-root", work / "data/kermany_rsna_mixed",
                "--summary-output", work / "audit/mixed_dataset_summary.json", *force), dry_run=args.dry_run)


def train(args: argparse.Namespace, family: str, model: str, seed: int) -> Path:
    work = args.work_dir
    config = "configs/densenet121_robust.yaml" if family == "robust" else MODELS[model]
    data_root = work / ("data/kermany_rsna_mixed" if family.startswith("mixed") else "data/kermany_grouped_seed42")
    summary = work / f"results/{family}_train_{model}_seed{seed}.json"
    extra = ["--domain-balanced-prefixes", "kermany", "rsna"] if family == "mixed_domain_balanced" else []
    run(command("scripts/train.py", "--config", config, "--data-root", data_root,
                "--results-dir", work / "results", "--checkpoints-dir", work / "checkpoints",
                "--seed", seed, "--device", args.device, "--json-output", summary, *extra), dry_run=args.dry_run)
    return summary


def evaluate(args: argparse.Namespace, family: str, model: str, seed: int, summary: Path, dataset: str, split: str) -> None:
    work = args.work_dir
    token = "kermany_grouped" if dataset == "kermany" else "rsna"
    root = work / ("data/kermany_grouped_seed42" if dataset == "kermany" else "data/rsna_binary")
    run(command("scripts/evaluate.py", "--training-summary", summary, "--data-root", root,
                "--split", split, "--device", args.device, "--json-output",
                work / f"results/{family}_eval_{token}_{split}_{model}_seed{seed}.json",
                "--predictions-output", work / f"results/{family}_predictions_{token}_{split}_{model}_seed{seed}.csv",
                "--no-update-latest"), dry_run=args.dry_run)


def experiment_stage(args: argparse.Namespace) -> None:
    for model in MODELS:
        for seed in SEEDS:
            summary = train(args, "strict", model, seed)
            evaluate(args, "strict", model, seed, summary, "kermany", "test")
            evaluate(args, "strict", model, seed, summary, "rsna", "test")
            if model == "densenet121":
                evaluate(args, "strict", model, seed, summary, "kermany", "val")
                evaluate(args, "strict", model, seed, summary, "rsna", "val")
    for family in FAMILIES[1:]:
        for seed in SEEDS:
            summary = train(args, family, "densenet121", seed)
            evaluate(args, family, "densenet121", seed, summary, "kermany", "test")
            evaluate(args, family, "densenet121", seed, summary, "rsna", "test")
            if family == "mixed_simple":
                evaluate(args, family, "densenet121", seed, summary, "kermany", "val")
                evaluate(args, family, "densenet121", seed, summary, "rsna", "val")


def analyze_stage(args: argparse.Namespace) -> None:
    work = args.work_dir
    run(command("scripts/summarize_strict_results.py", "--results-dir", work / "results",
                "--output-json", work / "results/strict_summary.json", "--output-csv",
                work / "results/strict_summary.csv", "--require-complete"), dry_run=args.dry_run)
    run(command("scripts/analyze_domain_shift.py", "--kermany", work / "data/kermany_grouped_seed42",
                "--rsna", work / "data/rsna_binary", "--output",
                work / "results/domain_shift_diagnostic.json"), dry_run=args.dry_run)
    for family in ("strict", "mixed_simple"):
        for dataset, root in (("kermany_grouped", work / "data/kermany_grouped_seed42"),
                              ("rsna", work / "data/rsna_binary")):
            for seed in SEEDS:
                validation = work / f"results/{family}_predictions_{dataset}_val_densenet121_seed{seed}.csv"
                test = work / f"results/{family}_predictions_{dataset}_test_densenet121_seed{seed}.csv"
                run(command("scripts/analyze_calibration.py", "--predictions", test,
                            "--calibration-predictions", validation, "--temperature-max", 20,
                            "--json-output", work / f"results/{family}_calibration_{dataset}_densenet121_seed{seed}.json"),
                    dry_run=args.dry_run)
                run(command("scripts/evaluate_frozen_thresholds.py", "--validation", validation,
                            "--test", test, "--output",
                            work / f"results/{family}_operating_points_{dataset}_densenet121_seed{seed}.json"),
                    dry_run=args.dry_run)
                if seed == 42:
                    run(command("scripts/analyze_errors.py", "--predictions", test, "--data-root", root,
                                "--csv-output", work / f"results/{family}_error_cases_{dataset}_densenet121_seed42.csv",
                                "--json-output", work / f"results/{family}_error_summary_{dataset}_densenet121_seed42.json",
                                "--grid-output", work / f"figures/{family}_error_grid_{dataset}_densenet121_seed42.png"),
                        dry_run=args.dry_run)


def verify_stage(args: argparse.Namespace) -> None:
    run(command("scripts/verify_reproduction.py", "--work-dir", args.work_dir), dry_run=args.dry_run)


def main() -> int:
    args = parse_args()
    if not args.work_dir.is_absolute():
        args.work_dir = ROOT / args.work_dir
    for name, fn in (("data", data_stage), ("experiments", experiment_stage),
                     ("analyze", analyze_stage), ("verify", verify_stage)):
        if args.stage in {name, "all"}:
            fn(args)
    if not args.dry_run:
        receipt = args.work_dir / "reproduction_receipt.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({
            "status": "complete" if args.stage in {"all", "verify"} else "stage_complete",
            "stage": args.stage,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "executable": sys.executable,
            },
        }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
