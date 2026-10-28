"""HOG features and classical classifier construction (research-only dependency)."""
import csv
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from hnrs_ml.classical import FAMILIES, build_classifier, features, hog_features

ROOT = Path(__file__).resolve().parents[1]
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def stroke(vertical):
    image = np.zeros((28, 28), np.float32)
    if vertical:
        image[4:24, 13:15] = 1
    else:
        image[13:15, 4:24] = 1
    return image


class HogTests(unittest.TestCase):
    def test_shape_range_and_blank_input(self):
        result = hog_features(np.random.default_rng(0).random((5, 1, 28, 28), dtype=np.float32))
        self.assertEqual(result.shape, (5, 1296))
        self.assertTrue(np.isfinite(result).all() and result.min() >= 0 and result.max() <= 1)
        self.assertEqual(float(np.abs(hog_features(np.zeros((1, 28, 28)))).sum()), 0.0)

    def test_orientation_bins_follow_gradient_direction(self):
        # A vertical stroke has horizontal gradients (bin 0); a horizontal stroke vertical ones (bin 4/5).
        vertical, horizontal = (hog_features(stroke(v)[None]).reshape(-1, 9).sum(0) for v in (True, False))
        self.assertEqual(int(vertical.argmax()), 0)
        self.assertIn(int(horizontal.argmax()), (4, 5))

    def test_chunking_does_not_change_features(self):
        images = np.random.default_rng(1).random((7, 28, 28), dtype=np.float32)
        np.testing.assert_allclose(hog_features(images, chunk=3), hog_features(images), rtol=1e-6)

    def test_pixel_features_and_unknown_kind(self):
        self.assertEqual(features(np.zeros((2, 1, 28, 28)), "pixels").shape, (2, 784))
        with self.assertRaises(ValueError):
            features(np.zeros((1, 28, 28)), "edges")


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn is a research-only dependency (requirements-research.txt)")
class ClassifierTests(unittest.TestCase):
    def test_every_family_fits_and_separates_two_shapes(self):
        rng = np.random.default_rng(2)
        images = np.stack([np.clip(stroke(i % 2 == 0) + rng.normal(0, .05, (28, 28)), 0, 1) for i in range(40)])
        labels = np.arange(40) % 2
        for family, spec in FAMILIES.items():
            x = features(images, spec["features"])
            model = build_classifier(family, spec["grid"][0], jobs=1).fit(x[:30], labels[:30])
            self.assertEqual(list(model.predict(x[30:])), list(labels[30:]), family)
        with self.assertRaises(ValueError):
            build_classifier("perceptron", {})

    def test_recorded_results_are_consistent(self):
        path = ROOT / "results/classical_baselines/benchmark.csv"
        if not path.exists():
            self.skipTest("Run python -m research.compare_classical_models to record results")
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({r["family"] for r in rows}, set(FAMILIES))
        for row in rows:
            self.assertEqual(int(row["test_count"]), 10000)
            self.assertAlmostEqual(int(row["test_correct"]) / 10000, float(row["test_accuracy"]))


if __name__ == "__main__":
    unittest.main()
