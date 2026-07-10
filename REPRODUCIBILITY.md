# Complete reproduction guide

This repository is designed so that a user can reproduce the published work
from the repository plus independently downloaded copies of the two source
datasets. No unpublished checkpoint is required.

## 1. Hardware and software

The reference audit used Python 3.13.12 and the exact Python packages in
`requirements.txt`. Training all 18 runs requires an NVIDIA GPU and substantial
time and storage. CUDA kernels can differ between hardware and driver versions,
so the acceptance gate requires exact data identity and a 5% relative tolerance
for primary model metrics. Statistical summaries from fixed prediction files are
deterministic.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest -q
```

## 2. Download the source datasets

Download Kermany Chest X-Ray Images (Pneumonia), version 2, from Mendeley Data
or its Kaggle mirror. Place the supplied `train`, `val`, and `test` directories
under:

```text
data/raw/chest_xray/
```

Download the official RSNA Pneumonia Detection Challenge files from Kaggle.
Place these files under:

```text
data/raw/rsna_pneumonia/
  stage_2_train_images/*.dcm
  stage_2_train_labels.csv
  stage_2_detailed_class_info.csv
```

The workflow does not use the historical Hugging Face mirror. It extracts the
frozen 1,707 RSNA members from the official dataset and verifies each DICOM
against the SHA-256 recorded in
`data/splits/rsna_available_1707_manifest.csv`. Any missing or different file is
a hard failure.

## 3. Run the complete workflow

Review all 100 data, training, evaluation, analysis, and verification commands without executing them:

```bash
python scripts/reproduce_all.py --dry-run
```

Run data preparation, 18 training jobs, evaluations, analyses, and acceptance
checks:

```bash
python scripts/reproduce_all.py --stage all --device auto
```

All newly generated files are written under `rebuild/`. Existing published
evidence is not overwritten. If a failed data build left partial files, inspect
them and then explicitly allow replacement inside `rebuild/`:

```bash
python scripts/reproduce_all.py --stage data --force
```

Stages can also be run separately in this order:

```bash
python scripts/reproduce_all.py --stage data
python scripts/reproduce_all.py --stage experiments --device cuda
python scripts/reproduce_all.py --stage analyze
python scripts/reproduce_all.py --stage verify
```

## 4. Acceptance criteria

`scripts/verify_reproduction.py` requires all of the following:

- exactly 5,856 hash-matched Kermany images;
- exactly 1,707 hash-matched RSNA DICOM members and the frozen splits;
- all six strict model/dataset groups: three architectures on two test sets;
- all six candidate comparisons: three DenseNet121 variants on two test sets;
- all seeds 42, 43, and 44 for every required group;
- primary ensemble metrics within 5% relative difference of the published
  values.

Success produces `rebuild/reproduction_verification.json` with
`"status": "VERIFIED"`. Missing files never count as a successful partial run.

## 5. Experiment matrix

| Family | Model | Seeds | Training data | Test data |
|---|---|---|---|---|
| strict | DenseNet121, ConvNeXt-Tiny, ViT-B/16 | 42, 43, 44 | grouped Kermany | grouped Kermany, frozen RSNA |
| robust | DenseNet121 | 42, 43, 44 | grouped Kermany, stronger augmentation | grouped Kermany, frozen RSNA |
| mixed_simple | DenseNet121 | 42, 43, 44 | Kermany + RSNA train/val | grouped Kermany, frozen RSNA |
| mixed_domain_balanced | DenseNet121 | 42, 43, 44 | mixed data with source-balanced sampling | grouped Kermany, frozen RSNA |

The downloadable `v1.0.0` ViT checkpoint is only for the local demonstration
and is not used anywhere in this reproduction workflow.
