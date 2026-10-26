> Recorded study: results and original source hashes are retained. Detailed outputs are in the [research archive](../guides/research-archive.md); current commands use the `research` package.

# Week 10 preprocessing experiments

**Decision: adopt post-resize peak normalisation for Otsu digit crops. Retain existing character preprocessing.**

## Question and controls

Can targeted preparation changes improve recognition without sacrificing existing successes? Week 9 identified three clear isolated-character preparation defects and seven thin-stroke review cases. Three candidates were specified before scoring; no parameter sweep or label-dependent filtering was used.

| Candidate | Single change | Coverage |
|---|---|---|
| A: background | Change local MaxFilter kernel from 31 to 15; retain Gaussian blur 8 and other settings | 80 characters and 40 strings |
| B: foreground | Remove a surviving component only if its area is below 10% of the largest and its bounding-box gap exceeds half the largest component height | 80 characters and 40 strings |
| C: stroke contrast | Divide each resized digit array by its nonzero maximum, preserving spatial support and relative intensities | 20 single digits and 20 numbers |

CNN is the primary model, MLP the secondary comparison. Both original digit and character checkpoints are fixed. Haresh's rejected weighted CNN is not used. Segmentation, reading order, images and supplied labels are unchanged. All 160 original prepared arrays and 320 original model predictions were reproduced before comparison. Candidate arrays, input/checkpoint/source hashes and every prediction are saved.

The prospective adoption rule requires no CNN exact-accuracy or CER regression in either relevant collection, strict improvement in at least one, no preparation failures, and visual review before integration. There is no automatic adoption from an isolated success.

## Complete comparison

| Collection | Variant | Model | Correct | Accuracy | CER | Prep ms |
|---|---|---|---:|---:|---:|---:|
| single_digits | C_peak | cnn | 16/20 | 80.00% | 20.000% | 75.12 |
| single_digits | C_peak | mlp | 11/20 | 55.00% | 45.000% | 75.12 |
| single_digits | baseline | cnn | 15/20 | 75.00% | 25.000% | 75.07 |
| single_digits | baseline | mlp | 11/20 | 55.00% | 45.000% | 75.07 |
| numbers | C_peak | cnn | 19/20 | 95.00% | 3.448% | 67.24 |
| numbers | C_peak | mlp | 9/20 | 45.00% | 24.138% | 67.24 |
| numbers | baseline | cnn | 18/20 | 90.00% | 5.172% | 67.18 |
| numbers | baseline | mlp | 9/20 | 45.00% | 24.138% | 67.18 |
| characters | A_background | cnn | 52/80 | 65.00% | 35.000% | 147.02 |
| characters | A_background | mlp | 43/80 | 53.75% | 46.250% | 147.02 |
| characters | B_foreground | cnn | 53/80 | 66.25% | 33.750% | 333.50 |
| characters | B_foreground | mlp | 42/80 | 52.50% | 47.500% | 333.50 |
| characters | baseline | cnn | 53/80 | 66.25% | 33.750% | 353.33 |
| characters | baseline | mlp | 43/80 | 53.75% | 46.250% | 353.33 |
| strings | A_background | cnn | 7/40 | 17.50% | 36.250% | 393.89 |
| strings | A_background | mlp | 4/40 | 10.00% | 48.125% | 393.89 |
| strings | B_foreground | cnn | 7/40 | 17.50% | 35.625% | 548.83 |
| strings | B_foreground | mlp | 3/40 | 7.50% | 49.375% | 548.83 |
| strings | baseline | cnn | 7/40 | 17.50% | 35.625% | 585.80 |
| strings | baseline | mlp | 3/40 | 7.50% | 49.375% | 585.80 |

CER is edit distance divided by total reference characters; lower is better. All variants completed without preparation errors. Timing is a single-pass descriptive measurement, not a speed benchmark: variants run in fixed order, baseline operations are reused for C, and regression tests ran concurrently during part of the run. No speed improvement is claimed.

## Paired gains and regressions

| Collection | Variant | Model | Gains | Regressions | Both correct | Both wrong |
|---|---|---|---:|---:|---:|---:|
| characters | A_background | cnn | 0 | 1 | 52 | 27 |
| characters | A_background | mlp | 0 | 0 | 43 | 37 |
| characters | B_foreground | cnn | 1 | 1 | 52 | 26 |
| characters | B_foreground | mlp | 0 | 1 | 42 | 37 |
| numbers | C_peak | cnn | 1 | 0 | 18 | 1 |
| numbers | C_peak | mlp | 0 | 0 | 9 | 11 |
| single_digits | C_peak | cnn | 1 | 0 | 15 | 4 |
| single_digits | C_peak | mlp | 0 | 0 | 11 | 9 |
| strings | A_background | cnn | 0 | 0 | 7 | 33 |
| strings | A_background | mlp | 1 | 0 | 3 | 36 |
| strings | B_foreground | cnn | 0 | 0 | 7 | 33 |
| strings | B_foreground | mlp | 0 | 0 | 3 | 37 |

## Visual investigation

The following 17 comparisons were inspected, covering all priority defects, the seven thin-stroke cases, detached-dot controls, and observed gains/regressions. This qualitative review is not an independent annotation study. All 160 previews remain available.

| ID / comparison | Observation |
|---|---|
| [101 (archive: evidence/preprocessing_experiment/week10_run01/previews/101.png)](../guides/research-archive.md) | Both character candidates retain broad texture contamination; original digit 2 remains unresolved. |
| [104 (archive: evidence/preprocessing_experiment/week10_run01/previews/104.png)](../guides/research-archive.md) | B removes distant noise and restores a larger centred 1; CNN changes i to 1, but MLP remains wrong (l to I). |
| [113 (archive: evidence/preprocessing_experiment/week10_run01/previews/113.png)](../guides/research-archive.md) | Both character candidates retain broad texture instead of the i; unresolved. |
| [074 (archive: evidence/preprocessing_experiment/week10_run01/previews/074.png)](../guides/research-archive.md) | B deletes the genuine detached j dot. Both models change correct lowercase j to uppercase J. Reject B. |
| [096 (archive: evidence/preprocessing_experiment/week10_run01/previews/096.png)](../guides/research-archive.md) | A changes CNN uppercase Z to lowercase z despite a retained outline; recognition regression. |
| [024 (archive: evidence/preprocessing_experiment/week10_run01/previews/024.png)](../guides/research-archive.md) | C retains the 9 geometry and restores intensity; CNN changes 7 to 9. |
| [037 (archive: evidence/preprocessing_experiment/week10_run01/previews/037.png)](../guides/research-archive.md) | C retains all three digit crops; CNN changes 140 to the correct 240. |
| [004 (archive: evidence/preprocessing_experiment/week10_run01/previews/004.png)](../guides/research-archive.md) | C brightens the existing 9 but CNN remains 7; geometry is not repaired by intensity scaling. |
| [014 (archive: evidence/preprocessing_experiment/week10_run01/previews/014.png)](../guides/research-archive.md) | C brightens the existing 9 but CNN remains 4. |
| [032 (archive: evidence/preprocessing_experiment/week10_run01/previews/032.png)](../guides/research-archive.md) | C retains the 9 outline; CNN remains wrong (4), MLP remains correct (9). |
| [020 (archive: evidence/preprocessing_experiment/week10_run01/previews/020.png)](../guides/research-archive.md) | C brightens the 0 outline without changing CNN correctness. |
| [053 (archive: evidence/preprocessing_experiment/week10_run01/previews/053.png)](../guides/research-archive.md) | Detached i dot/stem remain visible in both character candidates. |
| [054 (archive: evidence/preprocessing_experiment/week10_run01/previews/054.png)](../guides/research-archive.md) | Detached j dot and hook remain visible in both character candidates. |
| [073 (archive: evidence/preprocessing_experiment/week10_run01/previews/073.png)](../guides/research-archive.md) | Detached i dot/stem remain visible in both character candidates. |
| [077 (archive: evidence/preprocessing_experiment/week10_run01/previews/077.png)](../guides/research-archive.md) | Thin t strokes remain visible; CNN remains correct. |
| [081 (archive: evidence/preprocessing_experiment/week10_run01/previews/081.png)](../guides/research-archive.md) | Thin i and dot remain visible; CNN remains correct. |
| [083 (archive: evidence/preprocessing_experiment/week10_run01/previews/083.png)](../guides/research-archive.md) | Narrow l outline is retained; CNN remains wrong (1). |

## Interpretation and integration

A does not resolve the texture defects in 101 or 113. It loses a correct uppercase Z and increases CNN string CER. The faster observed preparation is not sufficient grounds for adoption and was not measured under benchmark controls.

B fixes the offset 1 (104) but removes Manula's genuine j dot (074). CNN character accuracy remains 53/80 while MLP falls from 43/80 to 42/80. A component-distance rule cannot safely distinguish every isolated dot from noise. B fails the strict-improvement gate and visual preservation check; it is retained only as rejected research code.

C improves CNN single digits from 15/20 to 16/20 and complete numbers from 18/20 to 19/20. It corrects Paarth's 9 (024) and Saniru's 240 (037), with no newly wrong examples. MLP exact accuracy and CER are unchanged, although a wrong prediction can change to another wrong prediction; see paired.csv. Normalisation restores intensity lost during downsampling; it neither thickens strokes nor repairs missing geometry. The other problematic 9s remain errors.

Only C is integrated into `segment_digits.prepare_crops` for the Otsu representation. The GUI number-recognition pipeline already calls this function. Fixed-threshold/grayscale alternatives, character preprocessing, segmentation and model checkpoints remain unchanged. No cross-pipeline combination is warranted because A and B were rejected.

Production verification regenerated all 40 digit inputs and exactly matched the saved C arrays and all 80 CNN/MLP predictions. The full regression suite passed 59 tests, including empty batches, intensity support, representation isolation and detached-dot preservation. This is code/pipeline verification; a new manual GUI acceptance session was not performed.

## Limits and remaining work

These are reused development photos from four writers, not independent final-test results. The gains are two photos in a small collection and do not establish statistical significance or generalisation. Character accuracy remains 53/80 and complete-string accuracy 7/40. Texture failures 101/113, the string segmentation defects and case/shape ambiguity remain unresolved. Fresh final photos must remain separate for Week 11 evaluation after integration choices.

## Evidence and reproduction

- [Prospective plan (archive: evidence/preprocessing_experiment/week10_run01/plan.json)](../guides/research-archive.md)
- [Full metric comparison (archive: evidence/preprocessing_experiment/week10_run01/summary.csv)](../guides/research-archive.md)
- [Every prediction and input hash (archive: evidence/preprocessing_experiment/week10_run01/predictions.csv)](../guides/research-archive.md)
- [Paired gains and regressions (archive: evidence/preprocessing_experiment/week10_run01/paired.csv)](../guides/research-archive.md)
- [Component boxes and warnings (archive: evidence/preprocessing_experiment/week10_run01/diagnostics.json)](../guides/research-archive.md)
- [Visual observations (archive: evidence/preprocessing_experiment/week10_run01/visual_review.json)](../guides/research-archive.md)
- [Adoption decision (archive: evidence/preprocessing_experiment/week10_run01/decision.json)](../guides/research-archive.md)
- [Baseline verification (archive: evidence/preprocessing_experiment/week10_run01/verification.json)](../guides/research-archive.md)
- [Production integration verification (archive: evidence/preprocessing_experiment/week10_run01/integration_verification.json)](../guides/research-archive.md)
- [59-test regression log (archive: evidence/preprocessing_experiment/week10_run01/software_tests_after_integration.txt)](../guides/research-archive.md)

The root experiment refuses a changed baseline. To reproduce after adoption, run the archived baseline source snapshot from the repository root. The archived runner differs from the initially executed runner only in source-root/snapshot handling; the executed runner is also retained at the evidence root with its original hash in plan.json. Reproduction source hashes are listed separately.

```powershell
python evidence/preprocessing_experiment/week10_run01/source/experiment_preprocessing_week10.py --output evidence/preprocessing_experiment/week10_repeat
python -m research.verify_preprocessing_week10
python -m unittest discover -s tests -v
```
