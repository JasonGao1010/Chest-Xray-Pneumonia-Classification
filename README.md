# Chest X-ray Pneumonia Classification

PyTorch project for binary chest X-ray classification, data-integrity auditing,
calibration, cross-source stress testing, and a local research demo.

This repository is for research and teaching only. It is not a medical device,
and its outputs must not be used for diagnosis or patient risk assessment.

## Main finding

Performance measured within one public dataset does not transfer reliably to a
different source. Three model families were trained with seeds 42, 43, and 44.
Their probability ensembles performed well on the held-out Kermany split but
degraded substantially on the RSNA subset:

| Model | Kermany balanced accuracy | RSNA balanced accuracy | RSNA specificity |
|---|---:|---:|---:|
| DenseNet121 | 0.9765 | 0.6591 | 0.3535 |
| ConvNeXt-Tiny | 0.9727 | 0.6695 | 0.4140 |
| ViT-B/16 | 0.9817 | 0.6594 | 0.3628 |

The Kermany test contains 1,170 images grouped into 624 conservative
filename-derived clusters. The RSNA test contains 442 images grouped by 442
`patientId` values. These datasets differ in population, labels, image format,
and distribution, so the comparison is a cross-source stress test rather than
a clinical validation.

Simple supervised mixed-source training improved the DenseNet121 RSNA ensemble
balanced accuracy from 0.6591 to 0.7813, while reducing recall from 0.9648 to
0.7533 and slightly reducing Kermany accuracy. It therefore changes the
operating trade-off and does not establish generalization to an unseen hospital.
The paired grouped-bootstrap comparisons are recorded in
[`results/strict_summary.json`](results/strict_summary.json).

## Evidence boundaries

- Public Kermany metadata do not provide a verified patient mapping. The strict
  split uses conservative filename-derived clusters and must not be described as
  confirmed patient-level separation.
- RSNA labels represent challenge target findings, not adjudicated clinical
  diagnoses. The 1,707-image working subset came from a third-party mirror; 134
  members have frozen hashes but incomplete retained download provenance.
- The small historical NIH subset has weak report-derived labels and cross-split
  patient overlap, so it is excluded from strict results.
- Bootstrap intervals quantify test-sample uncertainty for frozen predictions;
  they do not include the uncertainty of retraining additional seeds.

See [`data/README.md`](data/README.md) for data provenance and split semantics.

## Repository contents

```text
configs/        Training configurations
data/           Dataset instructions and tracked split manifests
figures/        Selected result figures
models/         Model-weight provenance and download instructions
results/        Machine-readable metrics, predictions, and audit records
scripts/        Data preparation, training, evaluation, and audit commands
src/            Reusable Python package
tests/          Automated tests
web/            Local inference interface
```

The main evidence entry points are:

- [`results/strict_summary.json`](results/strict_summary.json): three-seed
  ensembles, grouped bootstrap intervals, and paired comparisons;
- [`results/integrity_audit.json`](results/integrity_audit.json): known problems
  in the original Kermany and historical NIH layouts;
- [`results/integrity_audit_grouped.json`](results/integrity_audit_grouped.json):
  checks for the grouped Kermany and RSNA evaluation layouts;
- [`results/domain_shift_diagnostic.json`](results/domain_shift_diagnostic.json):
  source-separability analysis with diagnosis-label controls;
- [`scripts/README.md`](scripts/README.md): reproducible commands and output
  conventions.

## Installation

Python 3.13.12 and the exact package versions in `requirements.txt` were used
for the final audit. A separate virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest -q
```

The current test suite contains 42 tests. GPU training additionally depends on
a compatible NVIDIA driver and CUDA runtime.

## Data preparation

Raw and processed images are intentionally excluded from Git. Dataset sources,
licences, expected directory layouts, frozen manifests, and non-overwriting
rebuild commands are documented in [`data/README.md`](data/README.md).

To verify an existing local setup without replacing data:

```bash
python scripts/check_dataset.py --verify-images
python scripts/prepare_kermany_grouped.py --verify-only --verify-files \
  --summary /tmp/kermany_grouped_summary_check.json
```

Commands using `--force` may replace generated data and should only be used
after reviewing their target paths.

## Training and evaluation

Run commands from the repository root. Write reproductions to a new directory
so the published evidence remains unchanged:

```bash
python scripts/train.py \
  --config configs/densenet121.yaml \
  --data-root data/processed/kermany_grouped_seed42 \
  --results-dir rebuild/results \
  --checkpoints-dir rebuild/checkpoints \
  --seed 42 \
  --json-output rebuild/results/strict_train_densenet121_seed42.json

python scripts/evaluate.py \
  --training-summary rebuild/results/strict_train_densenet121_seed42.json \
  --data-root data/processed/kermany_grouped_seed42 \
  --split test --device auto \
  --json-output rebuild/results/strict_eval_kermany_test_densenet121_seed42.json \
  --predictions-output rebuild/results/strict_predictions_kermany_test_densenet121_seed42.csv \
  --no-update-latest
```

The full multi-seed, calibration, threshold, error-analysis, and audit commands
are listed in [`scripts/README.md`](scripts/README.md).

## Local demo

The downloadable `v1.0.0` weight is an earlier ViT-B/16 model trained on the
original Kermany directory split. It is not one of the strict three-seed models.
See [`models/README.md`](models/README.md) for its source and SHA-256 digest.

```bash
python scripts/serve_patient_app.py \
  --checkpoint models/vit_b16_best.pt --device cpu --threshold 0.5
```

Open `http://127.0.0.1:7860`. The interface is a research demonstration and its
predictions do not represent clinical performance.

## Licence

Project code is released under the [MIT License](LICENSE). Datasets, pretrained
weights, and third-party assets remain subject to their original licences and
terms.
