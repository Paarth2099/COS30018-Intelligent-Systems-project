> Recorded study: results and original source hashes are retained. Detailed outputs are in the [research archive](../guides/research-archive.md); current commands use the `research` package.

# Week 10 segmentation investigation

**Decision: adopt bounded broken-body grouping and wider short-stem eligibility in the grouped-components method.** Both previously diagnosed failures are repaired on this development collection.

## Responsibility and question

Segmentation separates a single handwritten line into complete characters in reading order. It is geometric component analysis, not a trained segmentation model. This investigation asks whether the Week 9 broken H and detached i failures can be repaired without merging neighbouring letters or damaging existing dot groups.

## Controlled experiments

- **A: broken-body grouping.** Merge adjacent groups only when both are tall (at least 60% of the observed 80th-percentile height), horizontal box gap is at most 4% of that height, vertical overlap is at least 65% of the shorter group, one width is at most 30% and the other at least 45% of typical height, and combined width is at most typical height. A single pass merges non-overlapping pairs; narrow/narrow neighbours are excluded. Raw strokes are not drawn or filled.
- **B: short-stem eligibility.** Increase the width/height limit for a short body from 0.45 to 0.55. Retain the minimum-height checks and existing dot-assignment rule.
- **Combined:** test A+B separately after both independent comparisons meet the adoption criteria.
- **Projection comparator:** rerun the existing vertical projection method alongside the independent candidates. It does not use the grouped-component changes.

The ratios are development hypotheses informed by Week 9 diagnosis, not universal thresholds or a parameter sweep. No filename, reference label, expected count or model prediction enters the segmentation algorithm. Reference strings are used only for evaluation.

All variants use the same 40 development strings, fixed original CNN/MLP checkpoints, unchanged character preprocessing, foreground mask and crop-preparation settings. Baseline boxes, arrays and both-model predictions were reproduced for all 40 strings. Component IDs are retained exactly once and outputs remain left to right. Number regression checks use Paarth's already adopted Week 10 digit normalisation, rather than the older Week 9 recognition baseline.

## Results

| Variant | Model | Matching count | Exact strings | CER | Images with warnings | Segmentation ms |
|---|---|---:|---:|---:|---:|---:|
| baseline | cnn | 38/40 | 7/40 | 35.625% | 3 | 283.19 |
| baseline | mlp | 38/40 | 3/40 | 49.375% | 3 | 283.19 |
| projection | cnn | 32/40 | 7/40 | 40.625% | 3 | 284.23 |
| projection | mlp | 32/40 | 2/40 | 53.750% | 3 | 284.23 |
| A_body | cnn | 39/40 | 7/40 | 34.375% | 4 | 281.00 |
| A_body | mlp | 39/40 | 3/40 | 48.125% | 4 | 281.00 |
| B_stem | cnn | 39/40 | 8/40 | 34.375% | 2 | 283.63 |
| B_stem | mlp | 39/40 | 3/40 | 48.125% | 2 | 283.63 |
| combined | cnn | 40/40 | 8/40 | 33.125% | 3 | 281.75 |
| combined | mlp | 40/40 | 3/40 | 46.875% | 3 | 281.75 |

CER is total edit distance divided by the 160 reference characters. Lower is better. Timing is single-pass, fixed-order and descriptive; small differences do not establish a speed improvement. All evaluated inputs completed without processing errors.

## What changed and why it matters

| Example | Before | After | Interpretation |
|---|---|---|---|
| Haresh Hello (122) | Six crops; CNN `rlell0` | Five complete crops; CNN `Hell0` | Broken H is repaired, but final o is still recognised as zero. Segmentation success does not imply exact recognition. |
| Haresh Bill (125) | Five crops; CNN `BeIll` | Four complete crops; CNN `Bill` | Short stem now qualifies as a body and receives its detached dot. |

A changes only 122; B changes only 125. The combined variant changes only these two examples. The other 38 grouped outputs have identical geometry, component groups and prepared arrays to baseline. Both models have no newly incorrect full strings. CNN gains one exact string; MLP exact recognition remains 3/40, although its total character errors decrease.

Visual inspection of all 40 combined outputs across eight sheets found one complete crop per visible character, appropriate detached-dot grouping and no obvious neighbour contamination. Repeated l characters, Room7 neighbours, leading zeros and i/j controls remain separated. This gives **40/40 visually acceptable development separations**, compared with the earlier 38/40; it is a qualitative assessment, not an independently annotated IoU score. Count agreement alone was not used as proof. Projection count agreement is 32/40; no new full projection visual score is claimed.

The repaired H now emits an inspection warning; the old implementation failed silently on that input. The resolved i no longer produces unassigned-fragment warnings. Warnings are diagnostic prompts, not pass/fail evidence.

## Integration and verification

The production `hnrs_chars/segmentation.py` grouped-components defaults now enable both changes. The existing GUI character-recognition route calls this pipeline, so no GUI feature or model switch is required. Projection, digit segmentation, character preprocessing and checkpoint files remain unchanged.

The production verification regenerates all 40 string outputs, checking exact boxes, component groups, prepared arrays and all 80 CNN/MLP predictions against the combined experiment. The 20 multi-digit regression images preserve all boxes and arrays from Paarth's Week 10 result; CNN remains 19/20 and MLP 9/20. The full regression test log is linked below. No new manual GUI acceptance session is claimed.

## Limits and next evaluation

The changes address observed development failures and have been tested on only four writers. Tight legitimate neighbouring characters may satisfy the merge geometry on unseen inputs; unusual dots or stems may still be grouped incorrectly. These checks do not establish robustness to cursive, touching characters, punctuation or multiple lines.

Exact CNN recognition is still only 8/40 (20%) and CER is 33.125%, despite acceptable separation in this set. Case ambiguity, O/o/0 and I/l/1 confusion, preparation and classifier errors remain. Fresh final photos were not loaded and must remain separate for the final evaluation after the configuration is frozen.

## Evidence

- [Independent prospective plan (archive: evidence/segmentation_experiment/week10_run01/plan.json)](../guides/research-archive.md)
- [Independent and projection metrics (archive: evidence/segmentation_experiment/week10_run01/summary.csv)](../guides/research-archive.md)
- [Combined plan (archive: evidence/segmentation_experiment/week10_combined01/plan.json)](../guides/research-archive.md)
- [Combined metrics (archive: evidence/segmentation_experiment/week10_combined01/summary.csv)](../guides/research-archive.md)
- [All combined-run predictions (archive: evidence/segmentation_experiment/week10_combined01/predictions.csv)](../guides/research-archive.md)
- [Paired gains and regressions (archive: evidence/segmentation_experiment/week10_combined01/paired.csv)](../guides/research-archive.md)
- [40-photo qualitative review (archive: evidence/segmentation_experiment/week10_combined01/visual_review.csv)](../guides/research-archive.md)
- [Boxes and component groups (archive: evidence/segmentation_experiment/week10_combined01/geometry.json)](../guides/research-archive.md)
- [Adoption decision (archive: evidence/segmentation_experiment/week10_combined01/decision.json)](../guides/research-archive.md)
- [20-number regression results (archive: evidence/segmentation_experiment/week10_combined01/number_regression.csv)](../guides/research-archive.md)
- [Experiment checks (archive: evidence/segmentation_experiment/week10_combined01/verification.json)](../guides/research-archive.md)
- [Production checks (archive: evidence/segmentation_experiment/week10_combined01/integration_verification.json)](../guides/research-archive.md)
- [Full regression test log (archive: evidence/segmentation_experiment/week10_combined01/software_tests.txt)](../guides/research-archive.md)
- [Combined review sheet 1 (archive: evidence/segmentation_experiment/week10_combined01/combined_sheet_01.png)](../guides/research-archive.md)
- [Combined review sheet 2 (archive: evidence/segmentation_experiment/week10_combined01/combined_sheet_02.png)](../guides/research-archive.md)
- [Combined review sheet 3 (archive: evidence/segmentation_experiment/week10_combined01/combined_sheet_03.png)](../guides/research-archive.md)
- [Combined review sheet 4 (archive: evidence/segmentation_experiment/week10_combined01/combined_sheet_04.png)](../guides/research-archive.md)
- [Combined review sheet 5 (archive: evidence/segmentation_experiment/week10_combined01/combined_sheet_05.png)](../guides/research-archive.md)
- [Combined review sheet 6 (archive: evidence/segmentation_experiment/week10_combined01/combined_sheet_06.png)](../guides/research-archive.md)
- [Combined review sheet 7 (archive: evidence/segmentation_experiment/week10_combined01/combined_sheet_07.png)](../guides/research-archive.md)
- [Combined review sheet 8 (archive: evidence/segmentation_experiment/week10_combined01/combined_sheet_08.png)](../guides/research-archive.md)

Per-image folders contain masks, raw crops, prepared arrays, segmentation metadata and previews. Baseline source snapshots and hashes are preserved. To repeat the controlled comparison after production defaults change, run the archived source from the repository root:

```powershell
python evidence/segmentation_experiment/week10_run01/source/experiment_segmentation_week10.py --output evidence/segmentation_experiment/week10_repeat
python evidence/segmentation_experiment/week10_combined01/source/experiment_segmentation_week10.py --combined --output evidence/segmentation_experiment/week10_combined_repeat
python -m research.verify_segmentation_week10
python -m unittest discover -s tests -v
```
