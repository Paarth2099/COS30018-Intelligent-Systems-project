"""Compare two segmenters on single-line alphanumeric DEVELOPMENT photos."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import time
import shutil
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from research.compare_segmentation import edit_distance
from .data import sha256
from .predict import CharacterPredictor
from .segmentation import METHODS, segment_characters, prepare_character_crops
from .photo_experiment import write_csv, EXTENSIONS


def parse_label(path):
    match = re.fullmatch(r"([A-Za-z]+)_(\d+)_text-([A-Za-z0-9]+)", path.stem)
    if not match or match[1] != path.parent.name:
        raise ValueError(f"Invalid string filename/writer: {path}")
    return match[1], match[3]


def load_predictor(experiment):
    selection = json.loads((experiment / "selection.json").read_text())
    name = selection["selected_model"]
    checkpoint = experiment / f"{name}.pt"
    record = next(r for r in selection["models"] if r["model"] == name)
    if sha256(checkpoint) != record["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint hash mismatch")
    return CharacterPredictor(checkpoint), checkpoint


def preview(seg, arrays, prediction, path, reference=None):
    annotated = seg.gray.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    for i, box in enumerate(seg.boxes, 1):
        draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline="red", width=2)
        draw.text((box[0], max(0, box[1] - 12)), str(i), fill="red")
    annotated.thumbnail((980, 300))
    width = max(1000, 90 * len(arrays))
    out = Image.new("RGB", (width, 470), "white")
    draw = ImageDraw.Draw(out)
    title = f"{seg.method} | predicted: {prediction}"
    if reference is not None:
        title += f" | reference: {reference}"
    draw.text((8, 5), title, fill="black")
    out.paste(annotated, (8, 25))
    for i, array in enumerate(arrays):
        tile = Image.fromarray(np.rint(array * 255).astype("uint8")).resize(
            (84, 84), Image.Resampling.NEAREST
        )
        out.paste(tile, (8 + 90 * i, 350))
        draw.text((8 + 90 * i, 440), str(i + 1), fill="black")
    out.save(path)


def save_artifacts(out, seg, arrays, prediction, reference=None):
    out.mkdir(parents=True)
    Image.fromarray(seg.mask.astype("uint8") * 255).save(out / "mask.png")
    for i, crop in enumerate(seg.crops):
        crop.save(out / f"crop_{i:02d}.png")
    np.save(out / "prepared.npy", arrays)
    for i, array in enumerate(arrays):
        Image.fromarray(np.rint(array * 255).astype("uint8")).save(
            out / f"prepared_{i:02d}.png"
        )
    details = {
        "coordinate_space": "upright resized grayscale image",
        "size": list(seg.gray.size),
        "boxes": seg.boxes,
        "component_groups": seg.groups,
        "removed_components": seg.removed_components,
        "warnings": seg.warnings,
        "method": seg.method,
    }
    (out / "segmentation.json").write_text(json.dumps(details, indent=2) + "\n")
    (out / "prediction.json").write_text(json.dumps(prediction, indent=2) + "\n")
    preview(seg, arrays, prediction["text"], out / "preview.png", reference)


def summarise(rows):
    total_chars = sum(len(r["reference"]) for r in rows)
    return {
        "images": len(rows),
        "reference_characters": total_chars,
        "count_correct": sum(r["count_correct"] for r in rows),
        "exact_correct": sum(r["exact_correct"] for r in rows),
        "exact_accuracy": sum(r["exact_correct"] for r in rows) / len(rows),
        "edit_distance": sum(r["edit_distance"] for r in rows),
        "character_error_rate": sum(r["edit_distance"] for r in rows) / total_chars,
        "errors": sum(bool(r["error"]) for r in rows),
        "mean_segmentation_ms": sum(r["segmentation_ms"] for r in rows) / len(rows),
        "mean_preparation_ms": sum(r["preparation_ms"] for r in rows) / len(rows),
        "mean_inference_ms": sum(r["inference_ms"] for r in rows) / len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/own_strings/development")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--experiment", type=Path, default=Path("experiments/emnist_byclass/run_01")
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory to retain previous evidence")
    paths = sorted(
        p
        for p in args.input.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )
    if not paths:
        parser.error("No string photos found")
    manifest = []
    for path in paths:
        writer, label = parse_label(path)
        with Image.open(path) as im:
            im.load()
        manifest.append(
            {
                "file": path.relative_to(args.input).as_posix(),
                "writer": writer,
                "reference": label,
                "sha256": sha256(path),
            }
        )
    torch.set_num_threads(4)
    predictor, checkpoint = load_predictor(args.experiment)
    predictor.predict_text(np.zeros((1, 28, 28), np.float32))
    args.output.mkdir(parents=True)
    write_csv(args.output / "manifest.csv", manifest)
    source = args.output / "source"
    source.mkdir()
    sources = {}
    for path in list(Path("hnrs_chars").glob("*.py")) + [
        Path("segment_digits.py"),
        Path("preprocess_digits.py"),
        Path("compare_segmentation.py"),
        Path("hnrs_ml/predict.py"),
    ]:
        dest = source / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        sources[path.as_posix()] = sha256(path)
    rows = []
    for path, item in zip(paths, manifest):
        with Image.open(path) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        for method in METHODS:
            row = {
                "file": item["file"],
                "writer": item["writer"],
                "reference": item["reference"],
                "method": method,
                "detected_count": 0,
                "count_correct": 0,
                "prediction": "",
                "exact_correct": 0,
                "edit_distance": len(item["reference"]),
                "segmentation_ms": 0.0,
                "preparation_ms": 0.0,
                "inference_ms": 0.0,
                "warnings": "",
                "error": "",
            }
            try:
                start = time.perf_counter()
                seg = segment_characters(image, method)
                row["segmentation_ms"] = (time.perf_counter() - start) * 1000
                row["detected_count"] = len(seg.boxes)
                row["count_correct"] = int(len(seg.boxes) == len(item["reference"]))
                row["warnings"] = "; ".join(seg.warnings)
                start = time.perf_counter()
                arrays = prepare_character_crops(seg.crops)
                row["preparation_ms"] = (time.perf_counter() - start) * 1000
                start = time.perf_counter()
                prediction = predictor.predict_text(arrays)
                row["inference_ms"] = (time.perf_counter() - start) * 1000
                row["prediction"] = prediction["text"]
                row["exact_correct"] = int(prediction["text"] == item["reference"])
                row["edit_distance"] = edit_distance(
                    item["reference"], prediction["text"]
                )
                save_artifacts(
                    args.output / item["writer"] / path.stem / method,
                    seg,
                    arrays,
                    prediction,
                    item["reference"],
                )
            except (ValueError, OSError) as error:
                row["error"] = str(error)
            rows.append(row)
    write_csv(args.output / "results.csv", rows)
    metrics = {m: summarise([r for r in rows if r["method"] == m]) for m in METHODS}
    # Mechanical provisional selection, subject to explicit visual crop review.
    winner = max(
        METHODS,
        key=lambda m: (
            metrics[m]["count_correct"],
            -metrics[m]["character_error_rate"],
            m == "projection",
        ),
    )
    summary = {
        "purpose": "Development comparison; no independent final test",
        "methods": metrics,
        "provisional_selected_method": winner,
        "selection_rule": "Higher count agreement, then lower CER, then simpler projection; visual review required",
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": sha256(checkpoint),
        "source_sha256": sources,
        "git_base": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "writers": dict(Counter(r["writer"] for r in manifest)),
        "duplicate_hash_groups": [
            [r["file"] for r in manifest if r["sha256"] == h]
            for h, c in Counter(r["sha256"] for r in manifest).items()
            if c > 1
        ],
        "timing_note": "Single local CPU run, warmed predictor; three stages timed separately; excludes loading and saving",
        "limits": [
            "Count agreement does not prove segmentation quality",
            "No true character bounding-box annotations",
            "Whole-string accuracy combines segmentation, preparation and classification errors",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    review = [
        {k: r[k] for k in ("file", "method", "reference", "detected_count")}
        | {"visual_status": "pending", "notes": ""}
        for r in rows
    ]
    write_csv(args.output / "visual_review.csv", review)
    # Five photos per page, side-by-side methods. Inspect all ordered crops.
    for writer in sorted({r["writer"] for r in manifest}):
        items = [r for r in manifest if r["writer"] == writer]
        for page in range((len(items) + 4) // 5):
            canvas = Image.new("RGB", (1200, 5 * 290), "white")
            draw = ImageDraw.Draw(canvas)
            for i, item in enumerate(items[page * 5 : page * 5 + 5]):
                for j, method in enumerate(METHODS):
                    source_path = (
                        args.output
                        / writer
                        / Path(item["file"]).stem
                        / method
                        / "preview.png"
                    )
                    if source_path.exists():
                        with Image.open(source_path) as im:
                            im.thumbnail((600, 280))
                            canvas.paste(im, (j * 600, i * 290 + 10))
                    draw.text((j * 600 + 5, i * 290), item["file"], fill="black")
            canvas.save(args.output / f"{writer}_review_{page+1}.png")
    print(json.dumps(metrics, indent=2))
    print("Provisional selection:", winner)


if __name__ == "__main__":
    main()
