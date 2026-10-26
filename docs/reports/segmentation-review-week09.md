> Recorded study: results and original source hashes are retained. Detailed outputs are in the [research archive](../guides/research-archive.md); current commands use the `research` package.

# Week 9 segmentation review

Visual review of all 20 multi-digit number photos and all 40 alphanumeric string photos from the existing paired-model experiment. No algorithm, model, label or GUI change was made.

## Method and result

The selected components method was recreated for numbers and grouped_components for strings. Every reconstructed bounding-box list exactly matched the frozen Week 9 result. All original image hashes were checked. Predictions and prepared arrays were reused unchanged. The review inspected annotated originals, raw crops and prepared crops, rather than inferring quality from the crop count or prediction alone.

| Collection | Photos | Matching crop count | Visually acceptable separation | Existing CNN exact recognition |
|---|---:|---:|---:|---:|
| Numbers | 20 | 20/20 | 20/20 | 18/20 |
| Alphanumeric strings | 40 | 38/40 | 38/40 | 7/40 |

All 60 outputs are sorted left to right. For the two failing strings, this orders fragments, not complete characters. A visual pass requires one complete crop per visible character with appropriate dot grouping and no obvious neighbour contamination. These qualitative judgments are not independently annotated ground truth or an IoU score.

There are two visible segmentation failures, nine examples where both models recognise the full input correctly, and 49 remaining examples with acceptable separation but at least one wrong model prediction. Those 49 are not established model-only failures: preparation and handwriting ambiguity may also contribute.

## Failure 1: broken H in Haresh Hello

[Annotated original, raw crops and prepared crops (archive: evidence/segmentation_review/week09/122/review.png)](../guides/research-archive.md)

The H becomes two crops, so Hello produces six regions. Inspection of the actual grouping inputs shows both H parts are classified as bodies: component 4 is 122x179 pixels and component 1 is 41x188 pixels, in the resized working image. Both exceed the body-height threshold of 71.6 pixels. The grouping routine attaches small fragments to bodies; it does not merge these two full-height bodies. The existing foreground closing has not connected them.

This explains why dot-grouping changes alone are unlikely to fix this case. A bounded broken-body repair or a carefully controlled connectivity change is worth testing. Merging all nearby bodies would risk joining valid neighbouring letters.

The saved warning list is empty for this failure. Therefore absence of a warning must not be reported as proof of correct separation.

## Failure 2: detached i in Haresh Bill

[Annotated original, raw crops and prepared crops (archive: evidence/segmentation_review/week09/125/review.png)](../guides/research-archive.md)

The i dot and stem remain separate, producing five crops. The working typical height is 188 pixels. The stem is 28 pixels wide and 59 high: it is below the primary body threshold of 75.2 pixels. It exceeds the short-body height threshold of 47 pixels, but fails the width condition because 28 is greater than 0.45 x 59 = 26.55. Thus both dot and stem are classified as small fragments rather than a dot with an eligible stem body. They remain unassigned in the saved grouping output.

The current code retains the fragments and emits two unassigned-component warnings. Retention is useful evidence; silently deleting the dot would hide the problem rather than solve it.

The next experiment should assess short-stem eligibility while keeping the dot-assignment rule fixed. These numbers diagnose this example, not a universal replacement threshold.

## Successful controls and non-segmentation errors

| Examples | What to preserve or investigate |
|---|---|
| Haresh ij12 (123) and jig (124) | Dots remain with the appropriate stems; neighbouring letters remain separate. CNN recognises both correctly. |
| Manula ij12 (133), jig (134), Bill (135) | Dot grouping is visually acceptable even though recognition is imperfect. |
| Saniru Bill (155) | Complete i crop and separate l crops; CNN recognises the string. |
| Manula Hello (132), Paarth Hello (142), Saniru Hello (152) | H is complete and repeated l characters are separate; do not merge them while repairing broken H. |
| Room7 (121, 131, 141, 151) | m and 7 remain separate with the selected method; use as anti-merge controls. |
| 007Ab (126, 136, 146, 156) | Leading zeros are separate and ordered. Recognition differences must not be corrected by dropping zeros. |
| Paarth 999 (025) | All three raw 9 crops are complete; weak prepared tails merit preparation/model review. Both models predict 494. |
| Saniru 240 (037) | Three complete ordered crops; MLP predicts 240 while CNN predicts 140. Wrong recognition is not evidence of a split/merge failure. |
| O0o and Il1 samples | Complete crops can still be ambiguous to the classifier. Preserve reference case and digits. |

## Controlled Week 10 experiment plan

1. **Broken-body repair:** compare one bounded connectivity/grouping change against the unchanged baseline. Target 122, while checking all other H examples and closely spaced neighbours. Do not use the filename, expected text, expected count or prediction to decide whether to merge regions.
2. **Short-stem eligibility:** compare one conservative body-eligibility change against baseline. Target 125; preserve correct dots in 123/124/133/134/135/143/144/145/153/154/155. Keep dot assignment and model/preprocessing fixed during this comparison.
3. **Combine only supported changes:** if either independent experiment helps, record a separate combined run and review every photo again. A fixed crop count with wrong dot assignment is a regression, not a success.

For each candidate use all 40 existing string development photos, the same frozen CNN and character preparation, and the same two segmentation methods where applicable for comparison. Record count agreement, visual separation, exact-string accuracy, CER, timing, warnings, new successes and regressions. Also run the 20 number photos as an integration regression check; number segmentation should remain unchanged.

Baseline string results are 38/40 visually acceptable, 7/40 exact, and 35.625% CER for grouped components. Historical projection results are 32/40 visually acceptable, 7/40 exact and 40.625% CER; this Week 9 visual review covers the selected methods only and does not claim a new projection rerun.

Prefer changes supported across the development set. If a candidate fixes one photo while harming other inputs, report the tradeoff rather than selecting only favourable examples. No guaranteed accuracy improvement is claimed. Keep fresh final-test photos unused until the configuration is frozen.

## Code locations

- `hnrs_chars/segmentation.py`: foreground connectivity repair, body/fragment classification, dot grouping and crop construction.
- `segment_digits.py`: selected number segmentation and digit crop preparation interface.
- `review_segmentation_week09.py`: recreation/verification and diagnostic image generation; it does not change the recognition implementation.

## Evidence and reproduction

- [Complete 60-photo review (archive: evidence/segmentation_review/week09/error_review.csv)](../guides/research-archive.md)
- [Diagnostic counts, warnings and hashes (archive: evidence/segmentation_review/week09/diagnostics.csv)](../guides/research-archive.md)
- [Grouping-rule measurements (archive: evidence/segmentation_review/week09/failure_rule_diagnostics.json)](../guides/research-archive.md)
- [Review summary (archive: evidence/segmentation_review/week09/review_summary.json)](../guides/research-archive.md)
- [Source/input provenance (archive: evidence/segmentation_review/week09/provenance.json)](../guides/research-archive.md)
- [Existing paired model comparison](photo-model-comparison-week09.md)

Each image has its own folder containing an annotated original, foreground mask, raw crops, grouping details and combined review preview. Ten contact sheets cover the full collection.

```powershell
python -m research.review_segmentation_week09 --output evidence/segmentation_review/week09_repeat
python evidence/segmentation_review/week09/diagnose_rules.py
```

The script verifies the frozen source and input hashes and exact agreement of bounding boxes with the existing run before writing diagnostic evidence. Visual judgments are qualitative annotations in error_review.csv; the script does not automatically judge crop quality. Changed sources require a separate experiment rather than rewriting this baseline review. The review is not blinded or independent, and no new training or recognition accuracy experiment was performed.

- [Contact sheet 1 (archive: evidence/segmentation_review/week09/sheet_01.png)](../guides/research-archive.md)
- [Contact sheet 2 (archive: evidence/segmentation_review/week09/sheet_02.png)](../guides/research-archive.md)
- [Contact sheet 3 (archive: evidence/segmentation_review/week09/sheet_03.png)](../guides/research-archive.md)
- [Contact sheet 4 (archive: evidence/segmentation_review/week09/sheet_04.png)](../guides/research-archive.md)
- [Contact sheet 5 (archive: evidence/segmentation_review/week09/sheet_05.png)](../guides/research-archive.md)
- [Contact sheet 6 (archive: evidence/segmentation_review/week09/sheet_06.png)](../guides/research-archive.md)
- [Contact sheet 7 (archive: evidence/segmentation_review/week09/sheet_07.png)](../guides/research-archive.md)
- [Contact sheet 8 (archive: evidence/segmentation_review/week09/sheet_08.png)](../guides/research-archive.md)
- [Contact sheet 9 (archive: evidence/segmentation_review/week09/sheet_09.png)](../guides/research-archive.md)
- [Contact sheet 10 (archive: evidence/segmentation_review/week09/sheet_10.png)](../guides/research-archive.md)
