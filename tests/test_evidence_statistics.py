from pathlib import Path

from scripts.analyze_domain_shift import select_label_matched
from scripts.analyze_errors import select_grid_cases
from scripts.analyze_errors import classify_error
from scripts.summarize_strict_results import (
    filename_group_from_path,
    paired_bootstrap_ci,
)


def test_kermany_group_key_is_subtype_aware():
    bacteria = filename_group_from_path(
        "test/PNEUMONIA/person12_bacteria_1.jpeg", "kermany_grouped"
    )
    virus = filename_group_from_path(
        "test/PNEUMONIA/person12_virus_1.jpeg", "kermany_grouped"
    )
    assert bacteria != virus
    assert bacteria.endswith("bacteria:person12")
    assert virus.endswith("virus:person12")


def test_paired_bootstrap_reports_candidate_minus_baseline():
    y = [0, 0, 1, 1]
    baseline = [0.4, 0.6, 0.4, 0.6]
    candidate = [0.1, 0.2, 0.8, 0.9]
    groups = ["n1", "n2", "p1", "p2"]

    interval = paired_bootstrap_ci(
        y, baseline, candidate, groups, iterations=200, seed=7
    )

    assert interval["accuracy"][0] >= 0.0
    assert interval["brier_score"][1] < 0.0


def test_label_matching_equalizes_each_label_stratum():
    rows = []
    counts = {
        ("kermany", "NORMAL"): 2,
        ("rsna", "NORMAL"): 4,
        ("kermany", "PNEUMONIA"): 5,
        ("rsna", "PNEUMONIA"): 3,
    }
    for (source, label), count in counts.items():
        for index in range(count):
            rows.append(
                {
                    "path": Path(f"/{source}/{label}/{index}.png"),
                    "source": source,
                    "label": label,
                    "group": f"{source}:{label}:{index}",
                }
            )

    matched = select_label_matched(rows)

    for label, expected in (("NORMAL", 2), ("PNEUMONIA", 3)):
        assert sum(
            row["source"] == "kermany" and row["label"] == label for row in matched
        ) == expected
        assert sum(
            row["source"] == "rsna" and row["label"] == label for row in matched
        ) == expected


def _error_row(path: str, coarse_type: str, score: float) -> dict[str, str]:
    return {
        "path": path,
        "true_label": "NORMAL" if coarse_type == "false_positive" else "PNEUMONIA",
        "predicted_label": "PNEUMONIA" if coarse_type == "false_positive" else "NORMAL",
        "prob_pneumonia": f"{score:.8f}",
        "confidence": f"{max(score, 1.0 - score):.8f}",
        "error_type": coarse_type,
        "coarse_type": coarse_type,
    }


def test_grid_selection_uses_balanced_fp_fn_quotas_before_correct_cases():
    rows = [
        _error_row(f"fp-{index}", "false_positive", 0.51 + index * 0.04)
        for index in range(10)
    ] + [
        _error_row(f"fn-{index}", "false_negative", 0.49 - index * 0.04)
        for index in range(10)
    ]
    rows.append(
        {
            **_error_row("correct", "correct", 0.5),
            "error_type": "correct_borderline",
        }
    )

    selected, metadata = select_grid_cases(rows, limit=8, threshold=0.5)

    assert len(selected) == 8
    assert metadata["selected_by_coarse_type"] == {
        "false_positive": 4,
        "false_negative": 4,
    }
    assert "correct" not in metadata["selected_paths"]


def test_error_classification_respects_requested_threshold():
    error_type, coarse_type = classify_error(
        "NORMAL", score=0.7, positive_class="PNEUMONIA", threshold=0.8, high_confidence=0.9
    )
    assert (error_type, coarse_type) == ("correct", "correct")
