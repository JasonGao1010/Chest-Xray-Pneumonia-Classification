from pathlib import Path

import pytest

from scripts.prepare_mixed_binary import main as prepare_mixed_main
from scripts.prepare_nih_binary import split_rows


def test_nih_split_keeps_each_patient_in_one_split():
    rows = [
        {"image_index": "00000001_001.png", "binary_label": "NORMAL"},
        {"image_index": "00000001_002.png", "binary_label": "PNEUMONIA"},
        {"image_index": "00000002_001.png", "binary_label": "NORMAL"},
        {"image_index": "00000003_001.png", "binary_label": "PNEUMONIA"},
        {"image_index": "00000004_001.png", "binary_label": "NORMAL"},
    ]

    split = split_rows(rows, seed=42)
    assignments: dict[str, set[str]] = {}
    for row in split:
        assignments.setdefault(row["patient_id"], set()).add(row["split"])

    assert all(len(values) == 1 for values in assignments.values())
    assert len(split) == len(rows)


def test_mixed_builder_refuses_existing_output(monkeypatch, tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_mixed_binary.py",
            "--source-a",
            str(tmp_path / "a"),
            "--source-b",
            str(tmp_path / "b"),
            "--output-root",
            str(output),
            "--summary-output",
            str(tmp_path / "summary.json"),
        ],
    )

    with pytest.raises(FileExistsError):
        prepare_mixed_main()


def test_mixed_builder_rejects_output_inside_source(monkeypatch, tmp_path: Path):
    source_a = tmp_path / "a"
    source_b = tmp_path / "b"
    source_a.mkdir()
    source_b.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_mixed_binary.py",
            "--source-a",
            str(source_a),
            "--source-b",
            str(source_b),
            "--output-root",
            str(source_a / "generated"),
            "--summary-output",
            str(tmp_path / "summary.json"),
        ],
    )

    with pytest.raises(ValueError, match="overlap"):
        prepare_mixed_main()
