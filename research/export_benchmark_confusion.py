"""Copy the recorded official-test confusion matrices into results/ for the GUI.

The Week 11 benchmark evaluation stored each model's confusion matrix under the
archived evidence folder, which is not shipped. This writes a compact copy so
the Model lab can display them without the archive. No model is re-run.

Run from the repository root:  python -m research.export_benchmark_confusion
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evidence/model_comparison/week11_run01/benchmarks"
TARGET = ROOT / "results/four_models/benchmark_confusion.json"
MODELS = ("mlp", "cnn", "resnet18", "mobilenet_v3_small")


def main():
    record = {"source": "evidence/model_comparison/week11_run01/benchmarks/<task>/<model>/metrics.json",
              "rows": "true class", "columns": "predicted class", "matrices": {}}
    for task in ("mnist", "emnist"):
        for model in MODELS:
            metrics = json.loads((SOURCE / task / model / "metrics.json").read_text())
            record["matrices"][f"{task}/{model}"] = {
                "count": metrics["count"], "correct": metrics["correct"],
                "accuracy": metrics["accuracy"], "confusion_matrix": metrics["confusion_matrix"]}
    TARGET.write_text(json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
