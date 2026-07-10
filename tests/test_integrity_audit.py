from pathlib import Path

from scripts.audit_integrity import audit_dataset, subject_id


def test_kermany_subject_id_parser():
    assert subject_id(Path("person100_bacteria_475.jpeg"), "PNEUMONIA") == "person100"
    assert subject_id(Path("NORMAL2-IM-1427-0001.jpeg"), "NORMAL") == "normal2-im-1427"
    assert subject_id(Path("IM-0115-0001.jpeg"), "NORMAL") == "im-0115"


def test_audit_detects_subject_crossing_splits(tmp_path):
    for split in ("train", "test"):
        directory = tmp_path / split / "PNEUMONIA"
        directory.mkdir(parents=True)
        (directory / f"person1_{split}_1.jpeg").write_bytes(split.encode())
    report = audit_dataset(tmp_path, hash_images=True)
    assert report["independent_split_ok"] is False
    assert report["subject_overlap"][0]["count"] == 1
