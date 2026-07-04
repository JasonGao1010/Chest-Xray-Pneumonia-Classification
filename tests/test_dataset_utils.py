from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from xray_pneumonia.data import (
    build_class_weights,
    build_samples,
    stratified_holdout_indices,
    validate_dataset_layout,
)


class DatasetUtilsTest(unittest.TestCase):
    def test_validate_counts_for_one_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            normal_dir = root / "train" / "NORMAL"
            pneumonia_dir = root / "train" / "PNEUMONIA"
            normal_dir.mkdir(parents=True)
            pneumonia_dir.mkdir(parents=True)
            (normal_dir / "n1.jpeg").write_bytes(b"placeholder")
            (pneumonia_dir / "p1.jpg").write_bytes(b"placeholder")
            (pneumonia_dir / "p2.png").write_bytes(b"placeholder")

            result = validate_dataset_layout(root, splits=("train",))

        self.assertTrue(result.ok)
        self.assertEqual(result.counts["train"]["NORMAL"], 1)
        self.assertEqual(result.counts["train"]["PNEUMONIA"], 2)
        self.assertEqual(result.total_images, 3)

    def test_missing_root_is_invalid(self) -> None:
        result = validate_dataset_layout("/tmp/xray_dataset_that_should_not_exist")

        self.assertFalse(result.ok)
        self.assertIn("/tmp/xray_dataset_that_should_not_exist", result.missing_dirs)

    def test_build_samples_uses_class_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train" / "NORMAL").mkdir(parents=True)
            (root / "train" / "PNEUMONIA").mkdir(parents=True)
            (root / "train" / "NORMAL" / "a.jpeg").write_bytes(b"placeholder")
            (root / "train" / "PNEUMONIA" / "b.jpeg").write_bytes(b"placeholder")

            samples = build_samples(root, "train")

        self.assertEqual([label for _, label in samples], [0, 1])

    def test_class_weights_fall_back_when_counts_are_zero(self) -> None:
        self.assertEqual(build_class_weights({"NORMAL": 0, "PNEUMONIA": 3}), [1.0, 1.0])
        weights = build_class_weights({"NORMAL": 1, "PNEUMONIA": 3})
        self.assertAlmostEqual(weights[0], 2.0)
        self.assertAlmostEqual(weights[1], 2.0 / 3.0)

    def test_stratified_holdout_indices_keep_each_class(self) -> None:
        samples = [
            (Path(f"n{i}.jpeg"), 0)
            for i in range(10)
        ] + [
            (Path(f"p{i}.jpeg"), 1)
            for i in range(20)
        ]

        train_indices, holdout_indices = stratified_holdout_indices(
            samples,
            holdout_fraction=0.2,
            seed=7,
        )

        self.assertEqual(len(train_indices) + len(holdout_indices), len(samples))
        self.assertEqual(len(set(train_indices).intersection(holdout_indices)), 0)
        holdout_labels = [samples[index][1] for index in holdout_indices]
        self.assertEqual(holdout_labels.count(0), 2)
        self.assertEqual(holdout_labels.count(1), 4)


if __name__ == "__main__":
    unittest.main()
