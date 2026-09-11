"""Download, validate and visualise MNIST. This script does not train a model."""
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def check_dataset(dataset, expected_count):
    """Fail clearly if the downloaded data does not match the agreed contract."""
    if tuple(dataset.data.shape) != (expected_count, 28, 28):
        raise ValueError(f"Unexpected image shape: {dataset.data.shape}")
    if tuple(dataset.targets.shape) != (expected_count,):
        raise ValueError("Unexpected label shape")
    if dataset.data.dtype != torch.uint8:
        raise ValueError("Raw images should contain unsigned 8-bit pixels")
    if sorted(dataset.targets.unique().tolist()) != list(range(10)):
        raise ValueError("Expected labels 0 through 9")
    image, label = dataset[0]
    if image.shape != (1, 28, 28) or image.dtype != torch.float32:
        raise ValueError("Prepared images must be float32 tensors of shape (1,28,28)")
    if not (0 <= image.min().item() <= image.max().item() <= 1):
        raise ValueError("Prepared pixels must lie in [0,1]")
    return {
        "count": len(dataset), "raw_image_shape": list(dataset.data.shape),
        "label_shape": list(dataset.targets.shape),
        "class_counts": torch.bincount(dataset.targets, minlength=10).tolist(),
        "prepared_image_shape": list(image.shape), "prepared_dtype": str(image.dtype),
        "first_label": label,
    }


def make_grid(dataset, output):
    """Show four genuine training examples of each digit, labelled by ground truth."""
    canvas = Image.new("RGB", (1000, 610), "#f4f6fa")
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 14), "MNIST training samples: true labels (not model predictions)", fill="#172238")
    for digit in range(10):
        indices = (dataset.targets == digit).nonzero().flatten()[:4]
        for row, index in enumerate(indices):
            x, y = digit * 98 + 14, row * 137 + 48
            tile = Image.fromarray(dataset.data[index].numpy()).resize((84, 84), Image.Resampling.NEAREST)
            canvas.paste(tile.convert("RGB"), (x, y))
            draw.text((x, y + 89), f"Label: {digit}", fill="#172238")
            draw.text((x, y + 106), f"Index: {index.item()}", fill="#465572")
    canvas.save(output)


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=base / "data")
    parser.add_argument("--output-dir", type=Path, default=base / "evidence")
    parser.add_argument("--offline", action="store_true", help="Use an existing dataset without downloading")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for name, is_train, size in [("training", True, 60000), ("test", False, 10000)]:
        datasets[name] = MNIST(root=args.data_dir, train=is_train, download=not args.offline, transform=ToTensor())
        check_dataset(datasets[name], size)
    report = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Dataset inspection only; no model trained or evaluated",
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "torchvision": torchvision.__version__,
                        "numpy": np.__version__, "pillow": PIL.__version__},
        "training": check_dataset(datasets["training"], 60000),
        "test": check_dataset(datasets["test"], 10000),
        "input_contract": "One digit per image; float32 [N,1,28,28]; pixels 0..1; light strokes on dark background",
        "future_split": "Later split the 60000 training examples into fixed training/validation subsets. Keep the 10000 test examples for final evaluation.",
    }
    make_grid(datasets["training"], args.output_dir / "mnist_samples.png")
    text = json.dumps(report, indent=2)
    (args.output_dir / "dataset_report.json").write_text(text + "\n", encoding="utf-8")
    (args.output_dir / "run_output.txt").write_text(text + "\nAll dataset checks passed.\n", encoding="utf-8")
    print(text)
    print("All dataset checks passed.")
    print(f"Open {args.output_dir / 'mnist_samples.png'} to view the labelled digits.")


if __name__ == "__main__":
    main()
