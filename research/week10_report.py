"""Generate the controlled model-experiment report from recorded results."""

import argparse
import json
from pathlib import Path
from research.week10_evaluate import read_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/reports/model-experiment-week10.md")
    )
    args = parser.parse_args()
    r = json.loads((args.evaluation / "results.json").read_text())
    selection = json.loads((args.experiment / "selection.json").read_text())
    plan = selection["config"]
    history = read_csv(args.experiment / "history.csv")
    photos = read_csv(args.evaluation / "photo_predictions.csv")
    weights = read_csv(args.experiment / "class_weights.csv")
    selected = selection["models"][0]
    text = [
        "# Week 10: controlled class-weighted CNN experiment",
        "",
        "**Decision: "
        + r["recommendation"]
        + ".** The application still uses its original saved CNN; no default or baseline file was replaced.",
        "",
        "## Question and rationale",
        "",
        "Does moderate class-frequency weighting improve letter recognition without sacrificing validation or real-photo performance? Week 9 found frequent O/0/o, I/l/1 and case confusions. The training set is imbalanced. Class weighting tests one plausible contributor; it cannot recover case information absent from a handwritten shape or repair defective crops.",
        "",
        "The only intentional training change is weighted cross entropy. Each class gets weight 1/sqrt(n), where n is its training-only count; weights are normalised to mean one. The square root is a moderate, preselected alternative to full inverse-frequency weighting, not a proven optimum. No exponent sweep or test-set-driven tuning was performed.",
        "",
        "PyTorch supports per-class cross-entropy weights for imbalanced training. The project runs installed PyTorch 2.6.0; the public API documentation describes this weighting mechanism. See [PyTorch CrossEntropyLoss documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html).",
        "",
        "## Controls",
        "",
        "The CNN was chosen for this experiment because it outperformed MLP on all four Week 9 development collections. The MLP and the digit classifiers are retained unchanged; this experiment targets the 62-class CNN only.",
        "",
        "- Same CNN architecture, random seed 42, train/validation indices, minibatch order, five epochs, batch size 256, Adam learning rate 0.001, zero weight decay and no augmentation. Training starts from scratch; this is not a warm start from the five-epoch baseline.",
        "- 628,142 training examples and 69,790 validation examples. Weights use training labels only.",
        "- Best epoch selected by highest unweighted validation accuracy, then lowest unweighted validation loss; retain earlier epoch on a tie.",
        "- Existing official test results were not used for selection. The official test partition and fresh final photos were not loaded in this experiment.",
        "- Exact Week 9 prepared arrays reused for all 80 character photos and 40 strings. Their image, array and baseline-source provenance is retained. No preprocessing or segmentation tuning was mixed into this comparison.",
        "",
        f"Selected candidate epoch: **{selected['best_epoch']}**. Training plus validation took {selected['training_seconds']:.1f} seconds on this computer. Runtime is descriptive, not a controlled speed comparison with the historical training run.",
        "",
        "## Training history",
        "",
        "| Epoch | Training accuracy | Validation accuracy | Unweighted validation loss |",
        "|---|---:|---:|---:|",
    ]
    for h in history:
        text.append(
            f"| {h['epoch']} | {float(h['train_accuracy']):.2%} | {float(h['validation_accuracy']):.2%} | {float(h['validation_unweighted_loss']):.4f} |"
        )
    text += [
        "",
        "Weighted training loss and unweighted baseline training loss optimise different objectives and should not be compared as equivalent quantities. The logged training loss is a sample-count-weighted average of weighted batch means. Validation loss remains unweighted.",
        "",
        "## Validation comparison",
        "",
        "| Model | Accuracy | Macro F1 | Digits | Uppercase | Lowercase |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("baseline", "weighted"):
        v = r["validation"][name]
        text.append(
            f"| {name} | {v['accuracy']:.2%} | {v['macro_f1']:.4f} | {v['groups']['digits']['accuracy']:.2%} | {v['groups']['uppercase']['accuracy']:.2%} | {v['groups']['lowercase']['accuracy']:.2%} |"
        )
    text += [
        "",
        "Macro F1 weights classes equally; accuracy weights examples equally. This validation set selected the checkpoint, so these are selection-set results, not an independent final generalisation estimate.",
        "",
        "## Frozen development-photo comparison",
        "",
        "| Collection | Model | Correct / total | Exact accuracy | CER | Inference ms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for v in r["photos"]:
        text.append(
            f"| {v['collection']} | {v['model']} | {v['correct']}/{v['count']} | {v['accuracy']:.2%} | {v['character_error_rate']:.2%} | {v['mean_inference_ms']:.3f} |"
        )
    text += [
        "",
        r["timing"],
        "",
        "| Collection | Newly correct | Newly wrong | Both correct | Both wrong |",
        "|---|---:|---:|---:|---:|",
    ]
    for c, v in r["paired"].items():
        text.append(
            f"| {c} | {v['improved']} | {v['regressed']} | {v['both_correct']} | {v['both_wrong']} |"
        )
    text += [
        "",
        "## Example gains, regressions and unresolved errors",
        "",
        "Examples are the first matching entries in dataset order for each group, not a favourable subset used for scoring. All 120 inputs remain in the aggregate. Linked images show the unchanged source/crops and the earlier MLP/CNN labels; the new weighted prediction is in this table.",
        "",
        "| Collection | Reference | Baseline CNN | Weighted CNN | Outcome | Source/crops |",
        "|---|---|---|---|---|---|",
    ]
    for collection in ("characters", "strings"):
        for outcome, condition in [
            (
                "gain",
                lambda p: p["baseline_correct"] == "0" and p["weighted_correct"] == "1",
            ),
            (
                "regression",
                lambda p: p["baseline_correct"] == "1" and p["weighted_correct"] == "0",
            ),
            (
                "unresolved",
                lambda p: p["baseline_correct"] == "0" and p["weighted_correct"] == "0",
            ),
        ]:
            for p in [
                p for p in photos if p["collection"] == collection and condition(p)
            ][:3]:
                text.append(
                    f"| {collection} | `{p['reference']}` | `{p['baseline']}` | `{p['weighted']}` | {outcome} | [Preview](../../{p['preview']}) |"
                )
    text += [
        "",
        "## Selected class diagnostics",
        "",
        "| Class | Train count | Weight | Baseline validation recall | Weighted validation recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for char in "0Oo1IlZzgq":
        w = next(w for w in weights if w["character"] == char)
        b = next(
            v
            for v in r["validation"]["baseline"]["per_class"]
            if v["character"] == char
        )
        c = next(
            v
            for v in r["validation"]["weighted"]["per_class"]
            if v["character"] == char
        )
        text.append(
            f"| `{char}` | {w['train_count']} | {float(w['loss_weight']):.3f} | {b['recall']:.2%} | {c['recall']:.2%} |"
        )
    text += [
        "",
        "## Decision and limitations",
        "",
        "The adoption rule was written before training: require no regression in validation accuracy, isolated-character exact accuracy, string exact accuracy and string CER. This conservative rule prevents selecting a change just because one attractive metric improves.",
        "",
        "| Criterion | Met? |",
        "|---|---|",
    ]
    for k, v in r["adoption_criteria"].items():
        text.append(f"| {k.replace('_',' ')} | {'Yes' if v else 'No'} |")
    text += [
        "",
        f"Result: **{r['recommendation']}**. Retain the candidate, its history and all negative results as research evidence. Do not replace the baseline checkpoint or relabel photos to improve the score.",
        "",
        "This is one candidate with one seed and a fixed short training budget. It does not establish optimal weighting or run-to-run significance. The same development handwriting has been inspected repeatedly, covers only four writers, and the isolated set contains only 20 of 62 labels. Better recall for a rare class can come with more false positives elsewhere; inspect the full per-class/confusion evidence.",
        "",
        "A model change does not repair the two known string segmentation failures or the three clear preparation defects. Independent preprocessing and segmentation experiments should use fixed comparison checkpoints. Fresh final evaluation remains for Week 11 after integration decisions.",
        "",
        "## Evidence and reproduction",
        "",
        f"- [Prospective experiment plan](../../{args.experiment.as_posix()}/plan.json)",
        f"- [Training history](../../{args.experiment.as_posix()}/history.csv)",
        f"- [Training log](../../{args.experiment.as_posix()}/training_log.txt)",
        f"- [Training-only class weights](../../{args.experiment.as_posix()}/class_weights.csv)",
        f"- [Validation-selected checkpoint record](../../{args.experiment.as_posix()}/selection.json)",
        f"- [Complete metrics and adoption decision](../../{args.evaluation.as_posix()}/results.json)",
        f"- [Verification checks](../../{args.evaluation.as_posix()}/verification.json)",
        f"- [All paired photo predictions](../../{args.evaluation.as_posix()}/photo_predictions.csv)",
        f"- [Per-writer results](../../{args.evaluation.as_posix()}/by_writer.csv)",
        f"- [Baseline validation confusion matrix](../../{args.evaluation.as_posix()}/baseline_validation_confusion.png)",
        f"- [Weighted validation confusion matrix](../../{args.evaluation.as_posix()}/weighted_validation_confusion.png)",
        "",
        "Source snapshots and validation prediction arrays are retained alongside the metrics. All 120 baseline photo predictions and the original baseline validation accuracy were reproduced before comparison.",
        "",
        "```powershell",
        "python -m research.week10_weighted --output experiments/emnist_byclass/week10_weighted_run02",
        "python -m research.week10_evaluate --candidate experiments/emnist_byclass/week10_weighted_run02 --output evidence/model_experiment/week10_run02",
        "python -m research.week10_report --experiment experiments/emnist_byclass/week10_weighted_run02 --evaluation evidence/model_experiment/week10_run02",
        "```",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
