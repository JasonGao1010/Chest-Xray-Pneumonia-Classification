import csv
import json
from pathlib import Path

from scripts.audit_integrity import (
    audit_dataset,
    audit_evaluations,
    filename_group_id,
    subject_id,
)


def test_kermany_subject_id_parser():
    assert subject_id(Path("person100_bacteria_475.jpeg"), "PNEUMONIA") == "person100"
    assert subject_id(Path("NORMAL2-IM-1427-0001.jpeg"), "NORMAL") == "normal2-im-1427"
    assert subject_id(Path("IM-0115-0001.jpeg"), "NORMAL") == "im-0115"


def test_audit_detects_filename_group_crossing_splits(tmp_path):
    for split in ("train", "test"):
        directory = tmp_path / split / "PNEUMONIA"
        directory.mkdir(parents=True)
        (directory / f"person1_{split}_1.jpeg").write_bytes(split.encode())
    report = audit_dataset(tmp_path, hash_images=True, dataset_kind="kermany")
    assert report["split_integrity_ok"] is False
    assert report["group_overlap"][0]["count"] == 1


def test_kermany_bare_token_collision_is_flagged_without_patient_claim(tmp_path):
    train = tmp_path / "train" / "PNEUMONIA"
    test = tmp_path / "test" / "PNEUMONIA"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    (train / "person7_bacteria_1.jpeg").write_bytes(b"train")
    (test / "person7_virus_1.jpeg").write_bytes(b"test")

    report = audit_dataset(tmp_path, hash_images=True, dataset_kind="kermany")

    assert report["group_disjoint_ok"] is False
    comparison = next(
        item
        for item in report["bare_token_sensitivity"]["comparisons"]
        if item["left"] == "train" and item["right"] == "test"
    )
    assert comparison["same_subtype_count"] == 0
    assert comparison["cross_subtype_count"] == 1
    assert filename_group_id(
        Path("person7_bacteria_1.jpeg"), "PNEUMONIA", "kermany"
    ) != filename_group_id(
        Path("person7_virus_1.jpeg"), "PNEUMONIA", "kermany"
    )


def test_nih_audit_uses_eight_digit_patient_id_across_labels(tmp_path):
    train = tmp_path / "train" / "NORMAL"
    test = tmp_path / "test" / "PNEUMONIA"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    (train / "00001234_001.png").write_bytes(b"train")
    (test / "00001234_002.png").write_bytes(b"test")

    report = audit_dataset(tmp_path, hash_images=True, dataset_kind="nih")

    assert report["split_integrity_ok"] is False
    assert report["group_overlap"][0]["ids"] == ["nih_patient_id:00001234"]


def test_evaluation_audit_includes_prefixed_formal_eval_files(tmp_path):
    predictions = tmp_path / "strict_predictions_demo.csv"
    with predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "true_label",
                "predicted_label",
                "prob_NORMAL",
                "prob_PNEUMONIA",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "path": "a.png",
                    "true_label": "NORMAL",
                    "predicted_label": "NORMAL",
                    "prob_NORMAL": "0.9",
                    "prob_PNEUMONIA": "0.1",
                },
                {
                    "path": "b.png",
                    "true_label": "PNEUMONIA",
                    "predicted_label": "PNEUMONIA",
                    "prob_NORMAL": "0.1",
                    "prob_PNEUMONIA": "0.9",
                },
            ]
        )
    evaluation = {
        "sample_count": 2,
        "predictions_output": predictions.as_posix(),
        "metrics": {"accuracy": 1.0, "confusion_matrix": [[1, 0], [0, 1]]},
    }
    (tmp_path / "strict_eval_demo.json").write_text(
        json.dumps(evaluation), encoding="utf-8"
    )

    report = audit_evaluations(tmp_path)

    assert report["discovered_files"] == 1
    assert report["checked_files"] == 1
    assert report["ok"] is True
