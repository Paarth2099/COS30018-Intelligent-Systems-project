"""Model-lab training core, custom-model use in the recogniser and Tk charts."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import tkinter as tk
import unittest

import numpy as np
from PIL import Image, ImageDraw
import torch

import gui_charts
from gui_core import MODEL_NAMES, RecognitionService
from gui_lab import shipped_history, top_confusions
from gui_training import TaskData, TrainingSettings, train
from hnrs_chars.data import CLASSES
from hnrs_chars.predict import CharacterPredictor
from hnrs_ml.predict import DigitPredictor

ROOT = Path(__file__).resolve().parents[1]


def synthetic(classes, count=240):
    """Bright bar whose row encodes the class: learnable in a few batches."""
    images = torch.zeros(count, 28, 28, dtype=torch.uint8)
    targets = torch.arange(count) % classes
    for index, label in enumerate(targets.tolist()):
        row = 2 + (label * 23) // max(1, classes - 1)
        images[index, row:row + 3, 4:24] = 255
    order = torch.randperm(count, generator=torch.Generator().manual_seed(0))
    return TaskData(images, targets, order[:180], order[180:],
                    [str(i) for i in range(10)] if classes == 10 else list(CLASSES))


class TrainingTests(unittest.TestCase):
    def test_settings_validation(self):
        TrainingSettings().validate()
        for change in [dict(task="cifar"), dict(model="vgg"), dict(epochs=0), dict(learning_rate=0),
                       dict(batch_size=4), dict(train_samples=10), dict(optimizer="rmsprop")]:
            with self.assertRaises(ValueError):
                TrainingSettings(**change).validate()

    def test_mnist_training_saves_loadable_checkpoint_and_history(self):
        events = []
        settings = TrainingSettings(model="cnn", epochs=3, learning_rate=0.01, batch_size=32,
                                    train_samples=180, validation_samples=100)
        with TemporaryDirectory() as tmp:
            record = train(settings, synthetic(10), Path(tmp) / "run", progress=events.append)
            self.assertFalse(record["stopped"])
            self.assertEqual([row["epoch"] for row in record["history"]], [1, 2, 3])
            self.assertGreater(record["validation_accuracy"], 0.5)
            self.assertEqual(np.array(record["confusion_matrix"]).sum(), 60)
            self.assertEqual({e["kind"] for e in events}, {"batch", "epoch", "done"})
            predictor = DigitPredictor(record["checkpoint"])
            self.assertEqual(predictor.model_name, "cnn")
            self.assertTrue((Path(tmp) / "run/history.csv").is_file())
            saved = json.loads((Path(tmp) / "run/run.json").read_text())
            self.assertEqual(saved["checkpoint_sha256"], record["checkpoint_sha256"])
            with self.assertRaises(FileExistsError):
                train(settings, synthetic(10), Path(tmp) / "run")

    def test_emnist_checkpoint_matches_character_predictor_contract(self):
        settings = TrainingSettings(task="emnist", model="mlp", epochs=1, batch_size=64,
                                    train_samples=180, validation_samples=100)
        with TemporaryDirectory() as tmp:
            record = train(settings, synthetic(62), Path(tmp) / "run")
            self.assertEqual(len(record["confusion_matrix"]), 62)
            self.assertEqual(CharacterPredictor(record["checkpoint"]).model_name, "mlp")

    def test_stop_request_keeps_completed_epochs_only(self):
        calls = []

        def stop_after_first_epoch():
            calls.append(1)
            return len(calls) > 6  # 180 samples / 32 per batch = 6 batches in epoch 1.
        settings = TrainingSettings(model="mlp", epochs=5, batch_size=32, train_samples=180, validation_samples=100)
        with TemporaryDirectory() as tmp:
            record = train(settings, synthetic(10), Path(tmp) / "run", should_stop=stop_after_first_epoch)
            self.assertTrue(record["stopped"])
            self.assertEqual(len(record["history"]), 1)
            self.assertIsNotNone(record["checkpoint"])
            stopped = train(settings, synthetic(10), Path(tmp) / "early", should_stop=lambda: True)
            self.assertIsNone(stopped["checkpoint"])

    def test_custom_model_is_selectable_and_hash_checked(self):
        settings = TrainingSettings(model="cnn", epochs=1, batch_size=32, train_samples=180, validation_samples=100)
        with TemporaryDirectory() as tmp:
            record = train(settings, synthetic(10), Path(tmp) / "run")
            service = RecognitionService(ROOT)
            service.register_custom("Numbers", "lab-cnn", record["checkpoint"], record["checkpoint_sha256"])
            self.assertEqual(service.models_for("Numbers"), list(MODEL_NAMES) + ["lab-cnn"])
            self.assertEqual(service.models_for("Alphanumeric"), list(MODEL_NAMES))
            image = Image.new("RGB", (120, 100), "white")
            ImageDraw.Draw(image).line([(60, 15), (60, 85)], fill="black", width=8)
            result, used = service.recognise(image, {"mode": "Numbers", "model": "lab-cnn",
                                                     "method": "components", "representation": "otsu"})
            self.assertEqual(len(result["prediction"]["text"]), 1)
            self.assertEqual(used["checkpoint_sha256"], record["checkpoint_sha256"])
            Path(record["checkpoint"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed since training"):
                service.recognise(image, {"mode": "Numbers", "model": "lab-cnn",
                                          "method": "components", "representation": "otsu"})
            with self.assertRaises(ValueError):
                service.register_custom("Numbers", "cnn", record["checkpoint"], "x")

    def test_shipped_histories_and_confusion_summary(self):
        for task in ("mnist", "emnist"):
            for model in ("mlp", "cnn", "resnet18", "mobilenet_v3_small"):
                rows = shipped_history(ROOT, task, model)
                self.assertEqual(len(rows), 5, (task, model))
                self.assertTrue(all(0 < r["validation_accuracy"] <= 1 for r in rows))
        self.assertEqual(top_confusions([[5, 2], [7, 1]], ["a", "b"]), [(7, "b", "a"), (2, "a", "b")])


class ChartTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"No Tk display: {error}")
        self.root.withdraw()
        self.canvas = tk.Canvas(self.root, width=400, height=300)

    def tearDown(self):
        self.root.destroy()

    def test_charts_draw_and_handle_empty_data(self):
        gui_charts.line_chart(self.canvas, {"train": [(1, .5), (2, .8)], "validation": [(1, .4), (2, .7)]}, "Acc")
        self.assertGreater(len(self.canvas.find_all()), 20)
        gui_charts.line_chart(self.canvas, {"train": []}, "Empty")
        self.assertEqual(len(self.canvas.find_all()), 1)
        gui_charts.bar_chart(self.canvas, ["a", "b"], [0.9, 0.95], "Bars", highlight={"b"})
        self.assertGreater(len(self.canvas.find_all()), 5)
        gui_charts.heatmap(self.canvas, [[3, 1], [0, 4]], ["0", "1"], "Confusion")
        texts = [self.canvas.itemcget(i, "text") for i in self.canvas.find_all() if self.canvas.type(i) == "text"]
        self.assertIn("3", texts)
        gui_charts.heatmap(self.canvas, np.eye(62, dtype=int).tolist(), list(CLASSES), "Large")
        self.assertGreater(len(self.canvas.find_all()), 62 * 62)


if __name__ == "__main__":
    unittest.main()
