"""Regression checks for integration risks, not model accuracy claims."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn

from hnrs_ml.data import development_split
from hnrs_ml.evaluate import classification_metrics
from hnrs_ml.models import build_model
from hnrs_ml.predict import DigitPredictor


class EncodedLabels(nn.Module):
    """A fake classifier used only to test order and leading-zero handling."""
    def forward(self, batch):
        labels = (batch[:, 0, 0, 0] * 9).round().long()
        return torch.nn.functional.one_hot(labels, num_classes=10).float() * 20


class ContractTests(unittest.TestCase):
    def test_split_disjoint_complete_repeatable(self):
        train, validation = development_split()
        self.assertEqual(len(train), 54000)
        self.assertEqual(len(validation), 6000)
        self.assertEqual(len(torch.cat([train, validation]).unique()), 60000)
        self.assertTrue(torch.equal(train, development_split()[0]))

    def test_input_validation(self):
        for bad in [torch.zeros(28, 28), torch.zeros(2, 3, 28, 28),
                    torch.zeros(2, 28, 28, dtype=torch.uint8), torch.full((1, 28, 28), float("nan")),
                    torch.full((1, 28, 28), 255.0), torch.full((1, 28, 28), -0.1)]:
            with self.subTest(shape=bad.shape):
                with self.assertRaises(ValueError):
                    DigitPredictor.prepare_batch(bad)
        self.assertEqual(DigitPredictor.prepare_batch(np.zeros((2, 28, 28), dtype=np.float32)).shape, (2, 1, 28, 28))

    def test_number_order_leading_zeros_empty_and_batch_boundaries(self):
        predictor = DigitPredictor.__new__(DigitPredictor)
        predictor.model_name, predictor.model = "fixture", EncodedLabels()
        batch = torch.zeros(5, 1, 28, 28)
        batch[:, 0, 0, 0] = torch.tensor([0, 0, 7, 3, 1]) / 9
        self.assertEqual(predictor.predict_number(batch)["text"], "00731")
        self.assertEqual([r["digit"] for r in predictor.predict_digits(batch, batch_size=2)], [0, 0, 7, 3, 1])
        self.assertEqual(predictor.predict_number(torch.zeros(0, 1, 28, 28))["text"], "")

    def test_checkpoints_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("mlp", "cnn"):
                model = build_model(name).eval()
                path = Path(directory) / f"{name}.pt"
                torch.save({"format_version": 1, "model_name": name, "state_dict": model.state_dict()}, path)
                predictor = DigitPredictor(path)
                batch = torch.rand(3, 1, 28, 28)
                with torch.inference_mode():
                    self.assertTrue(torch.equal(model(batch), predictor.model(batch)))

    def test_confusion_metrics_known_example(self):
        metrics = classification_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["confusion_matrix"][0][:2], [1, 1])
        self.assertEqual(metrics["per_class"][1]["recall"], 1.0)
        self.assertAlmostEqual(metrics["per_class"][1]["precision"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
