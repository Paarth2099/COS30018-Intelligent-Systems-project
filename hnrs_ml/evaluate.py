"""Final benchmark evaluation after validation-based selection; no model tuning."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from .data import load_partition
from .predict import DigitPredictor


def classification_metrics(truth, predicted):
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    rows = []
    for label in range(10):
        tp = int(confusion[label, label])
        support, predicted_count = int(confusion[label].sum()), int(confusion[:, label].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"digit": label, "support": support, "precision": precision, "recall": recall, "f1": f1})
    return {"count": len(truth), "correct": int(np.trace(confusion)),
            "accuracy": float(np.trace(confusion) / len(truth)),
            "macro_f1": float(np.mean([r["f1"] for r in rows])),
            "per_class": rows, "confusion_matrix": confusion.tolist()}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_failure_grid(images, truth, predicted, path):
    wrong = np.flatnonzero(truth != predicted)[:24]
    canvas = Image.new("RGB", (900, 540), "#f4f6fa")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 14), "First test errors in dataset order | true label -> model prediction", fill="#172238")
    for position, index in enumerate(wrong):
        x, y = 18 + (position % 8) * 110, 45 + (position // 8) * 156
        tile = Image.fromarray((images[index, 0].numpy() * 255).astype(np.uint8))
        canvas.paste(tile.resize((84, 84), Image.Resampling.NEAREST).convert("RGB"), (x, y))
        draw.text((x, y + 91), f"{truth[index]} -> {predicted[index]}", fill="#9c2135")
        draw.text((x, y + 110), f"Index {index}", fill="#465572")
    canvas.save(path)


def make_confusion_grid(matrix, path):
    canvas = Image.new("RGB", (660, 690), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 14), "MNIST confusion matrix | rows: true digit | columns: predicted digit", fill="#172238")
    largest = max(max(row) for row in matrix)
    for label in range(10):
        draw.text((80 + label * 55, 47), str(label), fill="#172238")
        draw.text((30, 87 + label * 55), str(label), fill="#172238")
    for row in range(10):
        for col in range(10):
            value = matrix[row][col]
            intensity = value / largest if largest else 0
            fill = (int(242 - 200 * intensity), int(247 - 155 * intensity), int(255 - 95 * intensity))
            x, y = 65 + col * 55, 70 + row * 55
            draw.rectangle((x, y, x + 53, y + 53), fill=fill)
            draw.text((x + 10, y + 20), str(value), fill="white" if intensity > .6 else "#172238")
    canvas.save(path)


def generated_string_evaluation(images, truth, predicted, confidence, predictor, output):
    # Source digits never overlap across strings, but they also appear in the
    # single-digit test results. String and digit results are not independent tests.
    rng = np.random.default_rng(2026)
    order = rng.permutation(len(truth))
    rows, cursor = [], 0
    preview = Image.new("RGB", (900, 570), "#f4f6fa")
    draw = ImageDraw.Draw(preview)
    draw.text((18, 14), "Generated test strings | known digit boundaries | not a segmentation test", fill="#172238")
    example_paths = []
    for length in (2, 3, 4, 5):
        for number in range(250):
            indices = order[cursor:cursor + length]
            cursor += length
            expected = "".join(str(truth[i]) for i in indices)
            result = "".join(str(predicted[i]) for i in indices)
            rows.append({"length": length, "example": number, "source_test_indices": " ".join(map(str, indices)),
                         "true_text": expected, "predicted_text": result, "correct": int(expected == result),
                         "minimum_softmax_score": float(confidence[indices].min())})
            if number < 2:
                row_number = (length - 2) * 2 + number
                y = 50 + row_number * 63
                for position, index in enumerate(indices):
                    tile = Image.fromarray((images[index, 0].numpy() * 255).astype(np.uint8))
                    preview.paste(tile.resize((42, 42)).convert("RGB"), (18 + position * 48, y))
                draw.text((290, y + 12), f"True: {expected}   Predicted: {result}", fill="#172238")
            if length == 3 and number == 0:
                # Exercise the public API using independently loaded saved weights.
                live_result = predictor.predict_number(images[indices])
                if live_result["text"] != result:
                    raise AssertionError("Prediction API disagrees with cached evaluation")
                crop_dir = output / "example_crops"
                crop_dir.mkdir()
                for position, index in enumerate(indices):
                    filename = f"digit_{position}.png"
                    Image.fromarray((images[index, 0].numpy() * 255).astype(np.uint8)).save(crop_dir / filename)
                    example_paths.append(str(Path("example_crops") / filename))
                (output / "example_prediction.json").write_text(json.dumps({"true_text": expected,
                    "source_test_indices": indices.tolist(), "result": live_result, "crop_files": example_paths}, indent=2) + "\n")
    write_csv(output / "generated_strings.csv", rows)
    preview.save(output / "generated_strings.png")
    return {"count": len(rows), "source_digit_count": cursor, "seed": 2026,
            "exact_string_accuracy": sum(r["correct"] for r in rows) / len(rows),
            "by_length": [{"length": length, "count": 250,
                           "exact_accuracy": sum(r["correct"] for r in rows if r["length"] == length) / 250}
                          for length in (2, 3, 4, 5)],
            "scope": "Classifier and ordered reconstruction with known crop boundaries; NOT end-to-end segmentation, photo or writer-generalisation evaluation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--experiment-dir", type=Path, default=Path("experiments/baseline"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("Threads must be positive")
    selection = json.loads((args.experiment_dir / "selection.json").read_text())
    output = args.experiment_dir / "evaluation"
    if output.exists():
        parser.error("Evaluation already exists; preserve this final test record")
    output.mkdir()
    torch.set_num_threads(args.threads)
    images, targets = load_partition(args.data_dir, train=False)
    truth, results, caches = targets.numpy(), [], {}
    for model_record in selection["models"]:
        name = model_record["model"]
        checkpoint = args.experiment_dir / f"{name}.pt"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != model_record["checkpoint_sha256"]:
            raise ValueError(f"{name}: checkpoint changed since validation selection")
        predictor = DigitPredictor(checkpoint)
        # Warm-up excluded; timing is CPU batch inference, not whole-application latency.
        predictor.predict_digits(images[:32])
        started = time.perf_counter()
        predictions = predictor.predict_digits(images)
        elapsed = time.perf_counter() - started
        predicted = np.array([p["digit"] for p in predictions])
        confidence = np.array([p["confidence"] for p in predictions])
        metrics = classification_metrics(truth, predicted)
        metrics.update({"model": name, "batch_inference_seconds": elapsed,
                        "amortised_ms_per_digit": elapsed * 1000 / len(truth), "inference_batch_size": 256,
                        "threads": args.threads, "device": "cpu"})
        results.append(metrics)
        caches[name] = (predicted, confidence)
        write_csv(output / f"{name}_predictions.csv", [{"test_index": i, "true_digit": int(truth[i]),
            "predicted_digit": int(predicted[i]), "softmax_score": float(confidence[i])} for i in range(len(truth))])
        write_csv(output / f"{name}_per_class.csv", metrics["per_class"])
        write_csv(output / f"{name}_confusion.csv", [{"true_digit": i, **{f"predicted_{j}": value for j, value in enumerate(row)}}
                  for i, row in enumerate(metrics["confusion_matrix"])])
        make_confusion_grid(metrics["confusion_matrix"], output / f"{name}_confusion.png")
        make_failure_grid(images, truth, predicted, output / f"{name}_errors.png")
    selected = selection["selected_model"]
    predicted, confidence = caches[selected]
    strings = generated_string_evaluation(images, truth, predicted, confidence,
        DigitPredictor(args.experiment_dir / f"{selected}.pt"), output)
    final = {"evaluated_utc": datetime.now(timezone.utc).isoformat(), "selected_model": selected,
             "selection_basis": "Validation only; unchanged after seeing test results",
             "single_digit": results, "generated_strings": strings,
             "limitations": ["One training seed; no repeated-run uncertainty estimate",
                             "No real group photos were supplied for this run",
                             "No integration with preprocessing, segmentation or GUI yet",
                             "Softmax scores are not calibrated probabilities of correctness",
                             "Generated strings reuse the single-digit test set and use known crop boundaries"]}
    (output / "results.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps({"selected_model": selected, "test_accuracy": {r["model"]: r["accuracy"] for r in results},
                      "generated_string_exact_accuracy": strings["exact_string_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
