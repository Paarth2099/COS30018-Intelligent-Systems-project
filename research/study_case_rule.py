"""Fit and evaluate the line-geometry letter-case rule (hnrs_chars/case_rule.py).

Protocol:
1. Segment and classify the 40 development string photos once per model.
2. Fit the threshold on development strings only: the midpoint of the
   separating gap between capital/zero scores and lowercase scores (or the
   lowest-error midpoint if the classes overlap).
3. Leave-one-writer-out: fit on three development writers, apply to the
   fourth, so every reported development change comes from a threshold that
   did not see that writer.
4. Adoption gate (the project's existing rule): no fall in exact matches and
   no rise in CER on either development collection, and at least one strict
   improvement.
5. Apply the all-writer threshold to the 120-photo final collection. That set
   was analysed before this study, so its numbers are post-hoc diagnostics,
   not untouched test evidence.

Run from the repository root:  python -m research.study_case_rule
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import torch

from gui_core import open_photo, resolve_checkpoint
from hnrs_chars.case_rule import (CASE_PAIRS, THRESHOLD, ZERO_LIKE, apply_line_case,
                                  line_reference, lowercase_score)
from hnrs_chars.predict import CharacterPredictor
from hnrs_chars.segmentation import prepare_character_crops, segment_characters
from research.compare_photo_models import discover
from research.compare_segmentation import edit_distance

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("mlp", "cnn", "resnet18", "mobilenet_v3_small")
SELECTED = "resnet18"


def collect():
    """Photos as dicts: split, collection, writer, file, reference."""
    items = [dict(split="development", collection=item["collection"], writer=item["writer"],
                  file=item["file"], reference=item["reference"])
             for item in discover() if item["collection"] in ("characters", "strings")]
    manifest = json.loads((ROOT / "results/final_evaluation/locked_manifest.json").read_text())
    items += [dict(split="final", collection="characters" if s["collection"] == "single_letters" else "strings",
                   writer=s["writer"], file=s["actual_path"], reference=s["verified_label"])
              for s in manifest["samples"] if s["collection"] in ("single_letters", "strings")]
    return items


def recognise(items):
    """Same grouped-component route as the GUI; boxes plus each model's text."""
    predictors = {name: CharacterPredictor(resolve_checkpoint(ROOT, "Alphanumeric", name)[0]) for name in MODELS}
    for item in items:
        segmented = segment_characters(open_photo(ROOT / item["file"]), "grouped_components")
        arrays = prepare_character_crops(segmented.crops)
        item["boxes"] = [list(box) for box in segmented.boxes]
        item["model_text"] = {name: predictor.predict_text(arrays)["text"] for name, predictor in predictors.items()}
        print(item["file"], item["reference"], item["model_text"][SELECTED], flush=True)
    return items


def training_points(items, model):
    """(score, is_lowercase, writer) for every scorable true candidate letter."""
    points = []
    for item in items:
        text, reference = item["model_text"][model], item["reference"]
        if len(text) != len(reference):
            continue  # Positions only align when segmentation found every character.
        line = line_reference(item["boxes"], text)
        if line is None:
            continue
        for box, truth in zip(item["boxes"], reference):
            if truth.lower() in CASE_PAIRS or truth in ZERO_LIKE:
                lowercase = truth in CASE_PAIRS or truth == "o"
                points.append((lowercase_score(box, line), lowercase, item["writer"]))
    return points


def fit_threshold(points):
    """Lowest-error midpoint; on ties, the one nearest the middle of the tied range."""
    scores = sorted({score for score, _, _ in points})
    if len(scores) < 2:
        return None, None
    candidates = [(a + b) / 2 for a, b in zip(scores, scores[1:])]
    errors = [sum((score >= t) != lowercase for score, lowercase, _ in points) for t in candidates]
    best = [t for t, e in zip(candidates, errors) if e == min(errors)]
    return statistics.median(best), min(errors)


def score_rows(items, model, threshold):
    rows = []
    for item in items:
        before = item["model_text"][model]
        after, details = apply_line_case(before, item["boxes"], threshold)
        rows.append(dict(split=item["split"], collection=item["collection"], writer=item["writer"],
                         file=item["file"], model=model, reference=item["reference"], model_text=before,
                         rule_text=after, threshold=round(threshold, 4), changes=len(details["changes"]),
                         correct_before=int(before == item["reference"]), correct_after=int(after == item["reference"]),
                         edits_before=edit_distance(item["reference"], before),
                         edits_after=edit_distance(item["reference"], after)))
    return rows


def totals(rows):
    characters = sum(len(r["reference"]) for r in rows)
    return dict(count=len(rows), correct_before=sum(r["correct_before"] for r in rows),
                correct_after=sum(r["correct_after"] for r in rows),
                cer_before=sum(r["edits_before"] for r in rows) / characters,
                cer_after=sum(r["edits_after"] for r in rows) / characters,
                gained=sum(r["correct_after"] > r["correct_before"] for r in rows),
                lost=sum(r["correct_after"] < r["correct_before"] for r in rows))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "results/case_rule")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    items = recognise(collect())
    development = [i for i in items if i["split"] == "development"]
    final = [i for i in items if i["split"] == "final"]
    dev_strings = [i for i in development if i["collection"] == "strings"]

    # Fit on all development writers (selected model) and record the separation.
    points = training_points(dev_strings, SELECTED)
    threshold, training_errors = fit_threshold(points)
    lower = [s for s, is_lower, _ in points if is_lower]
    upper = [s for s, is_lower, _ in points if not is_lower]

    # Leave-one-writer-out on development strings.
    held_out, folds = [], []
    for writer in sorted({i["writer"] for i in dev_strings}):
        fold_threshold, _ = fit_threshold([p for p in points if p[2] != writer])
        rows = score_rows([i for i in dev_strings if i["writer"] == writer], SELECTED, fold_threshold)
        held_out += rows
        folds.append(dict(writer=writer, threshold=fold_threshold, **totals(rows)))

    # Report the shipped (rounded) threshold, which must agree with this fit.
    if abs(threshold - THRESHOLD) > 0.005:
        raise SystemExit(f"Fitted threshold {threshold:.4f} differs from case_rule.THRESHOLD {THRESHOLD}; "
                         "review the data before changing the shipped value")
    all_rows, summary_rows = [], []
    for model in MODELS:
        rows = score_rows(items, model, THRESHOLD)
        all_rows += rows
        for split in ("development", "final"):
            for collection in ("characters", "strings"):
                part = [r for r in rows if (r["split"], r["collection"]) == (split, collection)]
                summary_rows.append(dict(split=split, collection=collection, model=model, **totals(part)))
    write_csv(args.output / "predictions.csv", all_rows)
    write_csv(args.output / "summary.csv", summary_rows)
    write_csv(args.output / "development_scores.csv",
              [dict(writer=w, score=round(s, 4), lowercase=int(l)) for s, l, w in points])

    selected_dev = {r["collection"]: r for r in summary_rows
                    if r["split"] == "development" and r["model"] == SELECTED}
    gate = {f"{c}_{k}": ok for c, r in selected_dev.items() for k, ok in
            [("exact_no_regression", r["correct_after"] >= r["correct_before"]),
             ("cer_no_regression", r["cer_after"] <= r["cer_before"])]}
    gate["strict_improvement"] = any(r["correct_after"] > r["correct_before"] or r["cer_after"] < r["cer_before"]
                                     for r in selected_dev.values())
    record = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), protocol=__doc__.split("Run from")[0].strip(),
        selected_model=SELECTED, fitted_threshold=threshold, module_threshold=THRESHOLD,
        training_points=len(points), training_errors=training_errors,
        lowercase_scores=dict(count=len(lower), minimum=min(lower), maximum=max(lower)),
        uppercase_or_zero_scores=dict(count=len(upper), minimum=min(upper), maximum=max(upper)),
        leave_one_writer_out=dict(folds=folds, combined=totals(held_out)),
        adoption_gate=dict(checks=gate, passed=all(gate.values())),
        summary=summary_rows,
        final_note="Final-collection rows are post-hoc: that collection was analysed before this rule existed.")
    (args.output / "summary.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("fitted_threshold", "lowercase_scores", "uppercase_or_zero_scores",
                                             "leave_one_writer_out", "adoption_gate")}, indent=2))
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
