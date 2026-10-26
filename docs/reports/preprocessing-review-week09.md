> Recorded study: results and original source hashes are retained. Detailed outputs are in the [research archive](../guides/research-archive.md); current commands use the `research` package.

# Week 9 preprocessing review

Review of the 20 single-digit and 80 isolated-character photographs from the paired MLP/CNN development experiment. This is diagnosis and experiment planning; no preprocessing, segmentation, model weights or GUI settings were changed.

## Coverage and method

All 100 original/prepared comparisons were inspected across ten contact sheets, including every failed/disagreed prediction and successful controls. Images and arrays are linked by their original paired-run index and SHA-256. Predictions are copied from the existing run; no new recognition score is claimed. Each image has an observation, hypothesis and next action in the review CSV.

The 20 digit inputs were prepared through the number pipeline; the 80 character inputs used the character pipeline. Their preparation differs, so findings must be investigated in the correct implementation. Sheet ink dimensions use a descriptive >0.2 threshold and are not automatic pass/fail criteria. Narrow i/l strokes can be valid.

| Review category | Images | Meaning |
|---|---:|---|
| Clear preparation defect | 3 | Visible background contamination or incorrect character scale/position |
| Thin-stroke review | 7 | Possible weakening to investigate; includes correct predictions |
| No clear preparation defect | 40 | At least one model wrong, but no clear defect visible in these previews |
| Successful control | 50 | Both models correct with no obvious defect in this review |

Both models are correct on 50/100 inputs; at least one is wrong on 50/100. These counts only describe this review population, not a new combined model score. No-clear-defect is not proof of optimal preparation.

## Priority findings

| ID | Observation | Interpretation |
|---|---|---|
| [ID 101 (archive: evidence/photo_model_comparison/week09_run01/previews/101.png)](../guides/research-archive.md), Saniru `2` | Prepared tile is dominated by background/texture instead of the isolated 2 visible in the original. | Foreground localisation includes unwanted paper/background; confirm with threshold mask and bounding box. |
| [ID 104 (archive: evidence/photo_model_comparison/week09_run01/previews/104.png)](../guides/research-archive.md), Saniru `1` | Digit 1 is small and offset; stray foreground above the character expands its occupied region. | Foreground localisation includes unwanted paper/background; confirm with threshold mask and bounding box. |
| [ID 113 (archive: evidence/photo_model_comparison/week09_run01/previews/113.png)](../guides/research-archive.md), Saniru `i` | Prepared tile contains broad paper-texture foreground instead of the isolated i and dot visible in the original. | Foreground localisation includes unwanted paper/background; confirm with threshold mask and bounding box. |

Digit 9 examples 004, 014 and 024 have weak descending strokes. Example 032 retains a visible narrow stem but CNN still fails while MLP succeeds. This comparison prevents treating every 9 error as the same preprocessing defect.

Successful thin-stroke/dot controls include 053, 054, 073 and 074. Example 081 has a faint i stem but CNN is correct. Preserve detached dots and existing successes during any foreground filtering or stroke experiment.

Round O/o/0, narrow I/l/1 and similarly written Z/z remain difficult even where their outlines are retained. The lowercase-j sample 114 already resembles uppercase J in the original. Do not alter supplied labels or claim preprocessing can recover case information absent from the shape.

## Boundary with segmentation and model work

The 100-image review concerns individual inputs; it does not assert that all multi-character crops were reviewed. Existing string evidence identifies Haresh Hello (122) with H split into two crops, and Bill (125) with i dot/stem split. Those require segmentation investigation. Matching crop count does not establish correct crop contents. Shape-preserving errors require model investigation alongside any preparation hypotheses.

## Controlled Week 10 plan

| Experiment | Single change to investigate | Main examples | Regression safeguards |
|---|---|---|---|
| A: Background estimation | Change one local background-correction parameter/method while holding the rest fixed | 101 and 113 | Inspect masks and prepared crops; retain dots in 053/054/073/074; run all 80 characters |
| B: Foreground localisation | Compare one conservative stray-component filtering rule with baseline; avoid largest-component-only filtering | 104 | Check dot/stem retention, full character crop and all 80 characters; do not use reference labels in filtering |
| C: Thin strokes | Trial one post-resize contrast or stroke-preservation change in the digit pipeline first | 004/014/024, with 020/032 as controls | Evaluate all 20 digits and 20 numbers; no automatic transfer to the character pipeline |

Run A, B and C independently against the frozen baseline first. Combine only supported changes in a separately recorded run. Keep the same saved CNN primary and MLP secondary, input files, segmentation and labels. Save source/settings/input/checkpoint hashes, every prediction, diagnostic images, new successes and regressions. Reject changes that merely trade the target failures for wider losses; report tradeoffs if no candidate improves overall behaviour.

For character changes, score all 80 individual characters and then all 40 development strings with fixed grouped segmentation to check integration. For digit changes, score the 20 individual digits and 20 number photos. Use exact accuracy, CER for multi-character inputs, preparation time, failure/warning counts and visual dot/stroke preservation. Keep isolated-character and full-string results separate.

The baseline remains digit CNN 15/20 singles and 18/20 numbers; character CNN 53/80 isolated characters and 7/40 strings (35.625% string CER). These are existing development results, not targets guaranteed to improve. Final fresh photos are reserved for Week 11 and must not guide Week 10 choices.

## Evidence and reproduction

- [Complete visual review table (archive: evidence/preprocessing_review/week09/error_review.csv)](../guides/research-archive.md)
- [Descriptive geometry and input hashes (archive: evidence/preprocessing_review/week09/diagnostics.csv)](../guides/research-archive.md)
- [Review counts (archive: evidence/preprocessing_review/week09/review_summary.json)](../guides/research-archive.md)
- [Provenance (archive: evidence/preprocessing_review/week09/provenance.json)](../guides/research-archive.md)
- [Previous paired model comparison](photo-model-comparison-week09.md)

- [Contact sheet 1 (archive: evidence/preprocessing_review/week09/sheet_01.png)](../guides/research-archive.md)
- [Contact sheet 2 (archive: evidence/preprocessing_review/week09/sheet_02.png)](../guides/research-archive.md)
- [Contact sheet 3 (archive: evidence/preprocessing_review/week09/sheet_03.png)](../guides/research-archive.md)
- [Contact sheet 4 (archive: evidence/preprocessing_review/week09/sheet_04.png)](../guides/research-archive.md)
- [Contact sheet 5 (archive: evidence/preprocessing_review/week09/sheet_05.png)](../guides/research-archive.md)
- [Contact sheet 6 (archive: evidence/preprocessing_review/week09/sheet_06.png)](../guides/research-archive.md)
- [Contact sheet 7 (archive: evidence/preprocessing_review/week09/sheet_07.png)](../guides/research-archive.md)
- [Contact sheet 8 (archive: evidence/preprocessing_review/week09/sheet_08.png)](../guides/research-archive.md)
- [Contact sheet 9 (archive: evidence/preprocessing_review/week09/sheet_09.png)](../guides/research-archive.md)
- [Contact sheet 10 (archive: evidence/preprocessing_review/week09/sheet_10.png)](../guides/research-archive.md)

To regenerate the diagnostic sheets and geometry using unchanged existing arrays:

```powershell
python -m research.review_preprocessing_week09 --output evidence/preprocessing_review/week09_repeat
```

Visual observations in error_review.csv are qualitative annotations, not automatically generated diagnoses. The script recreates contact sheets and numeric diagnostics only; changing the inputs requires a new visual review. This review is not independently annotated or blinded and does not establish causal effects. No new training or accuracy experiment was performed.
