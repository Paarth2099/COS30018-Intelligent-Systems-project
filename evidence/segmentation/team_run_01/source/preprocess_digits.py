"""Paarth's initial single-digit development experiment.

Dark handwriting on pale paper only. This is not number segmentation.
All variants share Otsu-based localisation and contrast stretching, so this
compares pixel representations after a common crop, not three independent
localisation algorithms. Shadows can be mistaken for ink: inspect previews.
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

METHODS = ("grayscale", "fixed", "otsu")
ROOT = Path(__file__).resolve().parent


def otsu_threshold(pixels):
    """Choose a threshold maximising between-class intensity variance."""
    hist = np.bincount(pixels.ravel(), minlength=256).astype(float)
    weight = np.cumsum(hist)
    moment = np.cumsum(hist * np.arange(256))
    denominator = weight * (pixels.size - weight)
    score = np.full(256, -1.0)
    valid = denominator > 0
    score[valid] = (moment[-1] * weight[valid] - moment[valid] * pixels.size) ** 2 / denominator[valid]
    if not valid.any():
        raise ValueError("Uniform image: no foreground/background contrast")
    return int(np.argmax(score))


def prepare_variants(image):
    """Return three float32 28x28 arrays, diagnostic crop, and warnings.

    Manula can supply one already separated digit as a PIL image here.
    Each prepared array is light ink on black, with values in [0, 1].
    """
    gray = ImageOps.grayscale(image)
    # Bound processing cost while preserving the original aspect ratio.
    gray.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    if np.asarray(gray).max() - int(np.asarray(gray).min()) < 10:
        raise ValueError("Very low contrast: inspect for blank or faint input")
    adjusted = ImageOps.autocontrast(gray)
    pixels = np.asarray(adjusted)
    threshold = otsu_threshold(pixels)
    mask = pixels <= threshold
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("No foreground detected")
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    warning = ""
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        warning = "Foreground touches image boundary; inspect for shadows, paper edges or clipped digit"
    if mask.mean() > 0.45:
        warning += "; Large foreground area: background may be mistaken for ink"
    views = {
        "grayscale": ImageOps.invert(adjusted),
        "fixed": Image.fromarray(np.where(pixels <= 128, 255, 0).astype(np.uint8)),
        "otsu": Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)),
    }
    variants = {}
    for method, view in views.items():
        crop = view.crop(box)
        scale = 20 / max(crop.size)
        size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
        crop = crop.resize(size, Image.Resampling.LANCZOS)
        canvas = Image.new("L", (28, 28), 0)
        canvas.paste(crop, ((28 - crop.width) // 2, (28 - crop.height) // 2))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        if not np.any(array):
            raise ValueError(f"{method} produced an empty digit")
        variants[method] = array
    return variants, gray.crop(box), warning.strip("; ")


def save_preview(original, crop, variants, destination):
    panels = [("Original", original), ("Common crop", crop)]
    for method, array in variants.items():
        panels.append((method, Image.fromarray(np.rint(array * 255).astype(np.uint8))))
    preview = Image.new("RGB", (240 * len(panels), 280), "white")
    draw = ImageDraw.Draw(preview)
    for index, (label, panel) in enumerate(panels):
        draw.text((index * 240 + 8, 8), label, fill="black")
        panel = panel.convert("RGB")
        if index >= 2:
            panel = panel.resize((224, 224), Image.Resampling.NEAREST)
        else:
            panel.thumbnail((224, 224))
        preview.paste(panel, (index * 240 + 8, 40))
    preview.save(destination)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Image or folder; filename must end label-0 through label-9")
    parser.add_argument("--output", type=Path, required=True, help="New directory: existing runs are never overwritten")
    parser.add_argument("--checkpoint", type=Path, help="Optional Haresh .pt checkpoint for predictions")
    args = parser.parse_args()
    source = args.input.resolve()
    if not source.exists():
        parser.error(f"Input does not exist: {source}")
    candidates = [source] if source.is_file() else sorted(source.rglob("*"))
    accepted, skipped = [], []
    for path in candidates:
        if not path.is_file() or path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            continue
        match = re.search(r"label-([0-9]+)$", path.stem)
        if match and len(match.group(1)) == 1:
            accepted.append((path, match.group(1)))
        else:
            skipped.append(str(path))
    if not accepted:
        parser.error("No labelled single-digit images found; example: Paarth_01_label-0.png")
    predictor = None
    if args.checkpoint:
        from hnrs_ml.predict import DigitPredictor
        predictor = DigitPredictor(args.checkpoint)
    output = args.output.resolve()
    if output.exists():
        parser.error("Output already exists. Choose a new run name to preserve earlier evidence.")
    output.mkdir(parents=True)
    rows, manifest = [], []
    for index, (path, label) in enumerate(accepted, 1):
        sample_dir = output / f"{index:03d}_{path.stem}"
        sample_dir.mkdir()
        manifest.append({"file": str(path), "label": label, "sha256": sha256(path)})
        try:
            with Image.open(path) as opened:
                original = ImageOps.exif_transpose(opened).convert("RGB")
            variants, crop, warning = prepare_variants(original)
            save_preview(original, crop, variants, sample_dir / "comparison.png")
            for method, array in variants.items():
                Image.fromarray(np.rint(array * 255).astype(np.uint8)).save(sample_dir / f"{method}.png")
                np.save(sample_dir / f"{method}.npy", array)
                prediction = predictor.predict_number(array[None, :, :])["text"] if predictor else ""
                rows.append({"file": str(path), "label": label, "method": method,
                             "prediction": prediction, "correct": int(prediction == label) if predictor else "",
                             "error": "", "warning": warning})
            print(f"Prepared {path.name}" + (f" | WARNING: {warning}" if warning else ""))
        except (ValueError, OSError) as error:
            for method in METHODS:
                rows.append({"file": str(path), "label": label, "method": method,
                             "prediction": "", "correct": 0 if predictor else "",
                             "error": str(error), "warning": ""})
            print(f"FAILED {path.name}: {error}")
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        correct = sum(row["correct"] for row in group) if predictor else None
        summary[method] = {"total": len(group), "preprocessing_errors": sum(bool(r["error"]) for r in group),
                           "correct": correct, "accuracy": correct / len(group) if predictor else None}
    record = {"purpose": "Development comparison; not untouched final-test accuracy",
              "design": "Common autocontrast and Otsu bounding box; grayscale vs fixed-128 vs Otsu representation; same resizing/padding",
              "summary": summary, "inputs": manifest, "skipped_images": skipped,
              "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
              "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
              "script_sha256": sha256(Path(__file__)), "numpy_version": np.__version__}
    (output / "summary.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Skipped {len(skipped)} multi-digit/unlabelled images; paths recorded in summary.json")
    print(f"Evidence saved in {output}")
    if any(row["error"] for row in rows):
        print("Some inputs failed: inspect results.csv before drawing conclusions.")


if __name__ == "__main__":
    main()
