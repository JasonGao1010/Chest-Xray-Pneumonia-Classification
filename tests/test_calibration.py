from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from xray_pneumonia.calibration import (
    calibration_summary,
    fit_temperature_grid,
    load_binary_predictions,
)


class CalibrationTest(unittest.TestCase):
    def test_load_binary_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            path.write_text(
                "path,true_label,predicted_label,prob_NORMAL,prob_PNEUMONIA\n"
                "a.jpeg,NORMAL,NORMAL,0.9,0.1\n"
                "b.jpeg,PNEUMONIA,PNEUMONIA,0.2,0.8\n",
                encoding="utf-8",
            )

            records = load_binary_predictions(path)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].true_label, "NORMAL")
        self.assertAlmostEqual(records[1].score, 0.8)

    def test_calibration_summary_contains_ece_and_brier(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            path.write_text(
                "path,true_label,predicted_label,prob_NORMAL,prob_PNEUMONIA\n"
                "n1.jpeg,NORMAL,NORMAL,0.9,0.1\n"
                "n2.jpeg,NORMAL,PNEUMONIA,0.4,0.6\n"
                "p1.jpeg,PNEUMONIA,PNEUMONIA,0.1,0.9\n"
                "p2.jpeg,PNEUMONIA,NORMAL,0.6,0.4\n",
                encoding="utf-8",
            )
            records = load_binary_predictions(path)

        summary = calibration_summary(records, n_bins=5)

        self.assertEqual(summary["sample_count"], 4)
        self.assertIn("ece", summary)
        self.assertIn("brier_score", summary)
        self.assertEqual(len(summary["bins"]), 5)

    def test_fit_temperature_grid_returns_positive_temperature(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            path.write_text(
                "path,true_label,predicted_label,prob_NORMAL,prob_PNEUMONIA\n"
                "n1.jpeg,NORMAL,NORMAL,0.8,0.2\n"
                "p1.jpeg,PNEUMONIA,PNEUMONIA,0.2,0.8\n",
                encoding="utf-8",
            )
            records = load_binary_predictions(path)

        fit = fit_temperature_grid(records, minimum=0.5, maximum=1.0, step=0.1)

        self.assertGreater(fit["temperature"], 0)
        self.assertIn("nll", fit)

    def test_temperature_grid_can_search_above_default_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            path.write_text(
                "path,true_label,predicted_label,prob_NORMAL,prob_PNEUMONIA\n"
                "n1.jpeg,NORMAL,PNEUMONIA,0.01,0.99\n"
                "p1.jpeg,PNEUMONIA,NORMAL,0.99,0.01\n",
                encoding="utf-8",
            )
            records = load_binary_predictions(path)
        fit = fit_temperature_grid(records, minimum=1.0, maximum=10.0, step=0.5)
        self.assertGreater(fit["temperature"], 5.0)

    def test_load_binary_predictions_rejects_invalid_probability(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.csv"
            path.write_text(
                "path,true_label,predicted_label,prob_PNEUMONIA\n"
                "x.png,NORMAL,NORMAL,1.2\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_binary_predictions(path)


if __name__ == "__main__":
    unittest.main()
