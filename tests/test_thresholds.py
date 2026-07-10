from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from xray_pneumonia.thresholds import (
    PredictionRecord,
    binary_roc_auc,
    compute_threshold_metrics,
    load_prediction_csv,
    make_threshold_grid,
    select_best_with_recall_floor,
    threshold_curve,
)


class ThresholdAnalysisTest(unittest.TestCase):
    def test_compute_threshold_metrics(self) -> None:
        records = [
            PredictionRecord("n1.jpeg", "NORMAL", 0.10),
            PredictionRecord("n2.jpeg", "NORMAL", 0.80),
            PredictionRecord("p1.jpeg", "PNEUMONIA", 0.70),
            PredictionRecord("p2.jpeg", "PNEUMONIA", 0.90),
        ]

        metrics = compute_threshold_metrics(records, 0.75)

        self.assertEqual(metrics.tn, 1)
        self.assertEqual(metrics.fp, 1)
        self.assertEqual(metrics.fn, 1)
        self.assertEqual(metrics.tp, 1)
        self.assertAlmostEqual(metrics.accuracy, 0.5)
        self.assertAlmostEqual(metrics.recall, 0.5)
        self.assertAlmostEqual(metrics.specificity, 0.5)

    def test_auc_is_one_for_perfect_rank_order(self) -> None:
        records = [
            PredictionRecord("n1.jpeg", "NORMAL", 0.10),
            PredictionRecord("n2.jpeg", "NORMAL", 0.20),
            PredictionRecord("p1.jpeg", "PNEUMONIA", 0.80),
            PredictionRecord("p2.jpeg", "PNEUMONIA", 0.90),
        ]

        self.assertAlmostEqual(binary_roc_auc(records), 1.0)

    def test_select_recall_floor_prefers_specificity(self) -> None:
        records = [
            PredictionRecord("n1.jpeg", "NORMAL", 0.10),
            PredictionRecord("n2.jpeg", "NORMAL", 0.60),
            PredictionRecord("p1.jpeg", "PNEUMONIA", 0.70),
            PredictionRecord("p2.jpeg", "PNEUMONIA", 0.90),
        ]
        curve = threshold_curve(records, make_threshold_grid(0.1))

        selected = select_best_with_recall_floor(curve, 1.0)

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertAlmostEqual(selected.threshold, 0.7)
        self.assertEqual(selected.fp, 0)

    def test_load_prediction_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            path.write_text(
                "path,true_label,predicted_label,prob_NORMAL,prob_PNEUMONIA\n"
                "a.jpeg,NORMAL,NORMAL,0.9,0.1\n"
                "b.jpeg,PNEUMONIA,PNEUMONIA,0.1,0.9\n",
                encoding="utf-8",
            )

            records = load_prediction_csv(path)

        self.assertEqual([record.true_label for record in records], ["NORMAL", "PNEUMONIA"])
        self.assertEqual([record.score for record in records], [0.1, 0.9])

    def test_compute_threshold_metrics_rejects_invalid_threshold(self) -> None:
        records = [PredictionRecord("a.png", "NORMAL", 0.2)]
        with self.assertRaises(ValueError):
            compute_threshold_metrics(records, 1.1)


if __name__ == "__main__":
    unittest.main()
