"""Compare classical (non-neural) digit classifiers with the selected CNN.

Protocol, fixed before the run:
1. Use the existing MNIST 54,000/6,000 development split (hash-checked against
   the baseline CNN configuration) so every model sees identical data.
2. Choose each family's setting from a small grid by validation accuracy only.
3. Evaluate every tuned family once on the official 10,000-image test set.
4. Run the tuned models on the same prepared crops as the CNN for the 40
   development digit photos and the 60 final digit photos. The final photos
   were analysed before this study, so these are comparison results only;
   no model is promoted into the application from them.

Run from the repository root (needs requirements-research.txt and local MNIST):
    python -m research.compare_classical_models
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import platform
import time

import numpy as np
import sklearn
import torch

from gui_core import open_photo, resolve_checkpoint
from hnrs_ml.classical import FAMILIES, build_classifier, features
from hnrs_ml.data import development_split, indices_hash, load_partition
from hnrs_ml.evaluate import classification_metrics
from hnrs_ml.predict import DigitPredictor
from research.compare_photo_models import discover, prepare
from research.compare_segmentation import edit_distance
from segment_digits import prepare_crops, segment_image

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/classical_baselines"


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def photo_items():
    """Development and final digit photos with their reference text."""
    items = [dict(split="development", file=item["file"], writer=item["writer"],
                  collection=item["collection"], reference=item["reference"])
             for item in discover() if item["collection"] in ("single_digits", "numbers")]
    manifest = json.loads((ROOT / "results/final_evaluation/locked_manifest.json").read_text())
    for sample in manifest["samples"]:
        if sample["collection"] in ("single_digits", "numbers"):
            items.append(dict(split="final", file=sample["actual_path"], writer=sample["writer"],
                              collection=sample["collection"], reference=sample["verified_label"]))
    return items


def prepared_photo(item):
    """Exactly the application's number route: components, Otsu preparation."""
    image = open_photo(ROOT / item["file"])
    if item["split"] == "development":
        arrays, _, _ = prepare(image, item["collection"])
        return arrays
    return prepare_crops(segment_image(image, "components").crops, "otsu")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--jobs", type=int, default=-1, help="CPU workers for kNN and random forest")
    parser.add_argument("--train-limit", type=int, default=None,
                        help="Smoke test only: use the first N training indices")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output folder already has results; choose a new --output")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    images, targets = load_partition(args.data_dir, train=True)
    train_ids, val_ids = development_split(len(targets))
    baseline = json.loads((ROOT / "experiments/baseline/config.json").read_text())
    if (indices_hash(train_ids) != baseline["train_indices_sha256"]
            or indices_hash(val_ids) != baseline["validation_indices_sha256"]):
        raise ValueError("Split differs from the CNN baseline; comparison would not be controlled")
    if args.train_limit:
        train_ids = train_ids[:args.train_limit]
    pixels = images[:, 0].numpy()
    labels = targets.numpy()
    train_ids, val_ids = train_ids.numpy(), val_ids.numpy()
    cache = {kind: features(pixels, kind) for kind in ("pixels", "hog")}

    # 1-2. Validation grid; the official test set is not loaded yet.
    grid_rows, tuned = [], {}
    for family, spec in FAMILIES.items():
        x = cache[spec["features"]]
        best = None
        for params in spec["grid"]:
            started = time.perf_counter()
            model = build_classifier(family, params, jobs=args.jobs).fit(x[train_ids], labels[train_ids])
            fit_seconds = time.perf_counter() - started
            accuracy = float((model.predict(x[val_ids]) == labels[val_ids]).mean())
            row = dict(family=family, params=json.dumps(params), validation_accuracy=accuracy,
                       fit_seconds=round(fit_seconds, 2))
            grid_rows.append(row)
            print(json.dumps(row), flush=True)
            # Strictly greater: on a tie the earlier, simpler setting is kept.
            if best is None or accuracy > best["validation_accuracy"]:
                best = dict(params=params, validation_accuracy=accuracy, fit_seconds=fit_seconds, model=model)
        tuned[family] = best
    write_csv(args.output / "validation_grid.csv", grid_rows)
    selected = max(tuned, key=lambda name: tuned[name]["validation_accuracy"])

    # 3. One official test evaluation per tuned family.
    test_images, test_targets = load_partition(args.data_dir, train=False)
    test_pixels = test_images[:, 0].numpy()
    truth = test_targets.numpy()
    benchmark, confusion = [], {}
    for family, best in tuned.items():
        kind = FAMILIES[family]["features"]
        started = time.perf_counter()
        predicted = best["model"].predict(features(test_pixels, kind))
        seconds = time.perf_counter() - started
        metrics = classification_metrics(truth, predicted)
        confusion[family] = metrics["confusion_matrix"]
        benchmark.append(dict(
            family=family, description=FAMILIES[family]["description"], features=kind,
            params=json.dumps(best["params"]), validation_accuracy=best["validation_accuracy"],
            test_correct=metrics["correct"], test_count=metrics["count"], test_accuracy=metrics["accuracy"],
            macro_f1=metrics["macro_f1"], fit_seconds=round(best["fit_seconds"], 2),
            ms_per_digit=seconds * 1000 / len(truth),
            model_bytes=len(pickle.dumps(best["model"], protocol=pickle.HIGHEST_PROTOCOL))))
        print(json.dumps(benchmark[-1]), flush=True)
    write_csv(args.output / "benchmark.csv", benchmark)

    # 4. Paired photo comparison on identical prepared crops.
    cnn = DigitPredictor(resolve_checkpoint(ROOT, "Numbers", "cnn")[0])
    photo_rows = []
    for item in photo_items():
        arrays = prepared_photo(item)
        outputs = {"cnn": cnn.predict_number(arrays)["text"] if len(arrays) else ""}
        for family, best in tuned.items():
            if len(arrays):
                digits = best["model"].predict(features(arrays, FAMILIES[family]["features"]))
                outputs[family] = "".join(str(int(d)) for d in digits)
            else:
                outputs[family] = ""
        for model, text in outputs.items():
            photo_rows.append(dict(item, model=model, prediction=text, correct=int(text == item["reference"]),
                                   edits=edit_distance(item["reference"], text)))
    write_csv(args.output / "photo_predictions.csv", photo_rows)
    photo_summary = []
    for split in ("development", "final"):
        for collection in ("single_digits", "numbers"):
            for model in ["cnn", *tuned]:
                rows = [r for r in photo_rows if (r["split"], r["collection"], r["model"]) == (split, collection, model)]
                photo_summary.append(dict(split=split, collection=collection, model=model,
                                          correct=sum(r["correct"] for r in rows), count=len(rows),
                                          cer=sum(r["edits"] for r in rows) / sum(len(r["reference"]) for r in rows)))
    write_csv(args.output / "photo_summary.csv", photo_summary)

    summary = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        protocol=__doc__.split("Run from")[0].strip(),
        selected_classical=selected, selection_basis="Highest validation accuracy among tuned families",
        train_count=len(train_ids), validation_count=len(val_ids), smoke_test=bool(args.train_limit),
        split_sha256=dict(train=baseline["train_indices_sha256"], validation=baseline["validation_indices_sha256"]),
        benchmark=benchmark, test_confusion=confusion, photo_summary=photo_summary,
        environment=dict(python=platform.python_version(), platform=platform.platform(),
                         sklearn=sklearn.__version__, numpy=np.__version__, torch=str(torch.__version__)))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Selected classical model by validation: {selected}. Results in {args.output}")


if __name__ == "__main__":
    main()
