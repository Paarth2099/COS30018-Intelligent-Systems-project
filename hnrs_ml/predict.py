"""Prediction for prepared digit crops; no image segmentation is performed here."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .models import build_model


class DigitPredictor:
    def __init__(self, checkpoint):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        self.model_name = saved["model_name"]
        self.model = build_model(self.model_name)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()

    @staticmethod
    def prepare_batch(crops):
        batch = torch.as_tensor(crops)
        if not batch.is_floating_point():
            raise ValueError("Pass float crops in [0,1]; divide raw 0..255 pixels by 255 first")
        if batch.ndim == 3:
            batch = batch.unsqueeze(1)
        if batch.ndim != 4 or tuple(batch.shape[1:]) != (1, 28, 28):
            raise ValueError("Expected [N,1,28,28] or [N,28,28], one digit per crop")
        batch = batch.detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(batch).all():
            raise ValueError("Crops contain NaN or infinite pixels")
        if batch.numel() and (batch.min() < 0 or batch.max() > 1):
            raise ValueError("Pixels must be scaled to [0,1]")
        return batch

    @torch.inference_mode()
    def predict_digits(self, crops, batch_size=256):
        """Keep input order; confidence is a softmax score, not calibrated certainty."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        batch = self.prepare_batch(crops)
        result = []
        for chunk in batch.split(batch_size):
            if not len(chunk):
                continue
            probabilities = self.model(chunk).softmax(dim=1)
            scores, labels = probabilities.max(dim=1)
            for label, score in zip(labels.tolist(), scores.tolist()):
                result.append({"digit": label, "confidence": score})
        return result

    def predict_number(self, ordered_crops):
        """Return text so a leading zero in '007' is preserved."""
        predictions = self.predict_digits(ordered_crops)
        return {"text": "".join(str(item["digit"]) for item in predictions),
                "digits": predictions, "model": self.model_name}


def main():
    parser = argparse.ArgumentParser(description="Recognise already prepared 28x28 digit files in supplied order")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--crops", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    crops = []
    for path in args.crops:
        with Image.open(path) as image:
            if image.size != (28, 28):
                parser.error(f"{path}: expected an already prepared 28x28 crop")
            crops.append(np.asarray(image.convert("L"), dtype=np.float32) / 255)
    print(json.dumps(DigitPredictor(args.checkpoint).predict_number(np.stack(crops)), indent=2))


if __name__ == "__main__":
    main()
