> Recorded study: results and original source hashes are retained. Detailed outputs are in the [research archive](../guides/research-archive.md); current commands use the `research` package.

# Week 9: real-photo MLP/CNN comparison

Paired evaluation of the existing saved classifiers on all 160 development photographs. No models, preprocessing or segmentation algorithms were changed.

## Method

The digit models use connected components and Otsu preparation. Individual characters use the existing local lighting correction, speck removal, Otsu and ink normalisation. Strings use grouped components and the same character preparation. Each photo is prepared once and exactly the same array is passed to both models. Reference labels are used only for scoring; case and leading zeros are significant.

All four checkpoints match the hashes in their original selection records. All 160 CNN predictions reproduce the previous experiment predictions. All photos remain in the comparison. A failed run stops rather than silently dropping a photo.

## Recognition results

| Collection | Model | Correct / photos | Accuracy | CER | Mean inference ms | Mean pipeline ms |
|---|---|---:|---:|---:|---:|---:|
| single_digits | mlp | 11/20 | 55.00% | 45.00% | 0.327 | 59.00 |
| single_digits | cnn | 15/20 | 75.00% | 25.00% | 0.543 | 59.22 |
| numbers | mlp | 9/20 | 45.00% | 24.14% | 0.355 | 48.75 |
| numbers | cnn | 18/20 | 90.00% | 5.17% | 0.944 | 49.33 |
| characters | mlp | 43/80 | 53.75% | 46.25% | 0.279 | 273.24 |
| characters | cnn | 53/80 | 66.25% | 33.75% | 0.494 | 273.46 |
| strings | mlp | 3/40 | 7.50% | 49.38% | 0.433 | 626.11 |
| strings | cnn | 7/40 | 17.50% | 35.62% | 1.200 | 626.88 |

Single-image accuracy means exact equality: for numbers/strings every character must match. CER is total Levenshtein edits divided by reference character count, not one minus exact-string accuracy.

## Paired outcomes

| Collection | Both correct | Only MLP correct | Only CNN correct | Both wrong |
|---|---:|---:|---:|---:|
| single_digits | 10 | 1 | 5 | 4 |
| numbers | 8 | 1 | 10 | 1 |
| characters | 40 | 3 | 13 | 24 |
| strings | 1 | 2 | 6 | 31 |

## Representative errors and disagreements

Examples below are selected deterministically by file order, not as an accuracy sample. Every input has a preview in the evidence directory.

| Collection | Reference | MLP | CNN | Review priority | Preview |
|---|---|---|---|---|---|
| single_digits | `9` | `9` | `4` | recognition_error_review_crops | [Image (archive: evidence/photo_model_comparison/week09_run01/previews/032.png)](../guides/research-archive.md) |
| single_digits | `7` | `2` | `7` | recognition_error_review_crops | [Image (archive: evidence/photo_model_comparison/week09_run01/previews/003.png)](../guides/research-archive.md) |
| single_digits | `1` | `5` | `1` | recognition_error_review_crops | [Image (archive: evidence/photo_model_comparison/week09_run01/previews/012.png)](../guides/research-archive.md) |
| single_digits | `9` | `3` | `7` | recognition_error_review_crops | [Image (archive: evidence/photo_model_comparison/week09_run01/previews/004.png)](../guides/research-archive.md) |
| single_digits | `7` | `2` | `2` | recognition_error_review_crops | [Image (archive: evidence/photo_model_comparison/week09_run01/previews/011.png)](../guides/research-archive.md) |
| numbers | `240` | `240` | `140` | recognition_error_review_crops | [Image (archive: evidence/photo_model_comparison/week09_run01/previews/037.png)](../guides/research-archive.md) |
| numbers | `14` | `17` | `14` | recognition_error_review_crops | [Image (archive: evidence/photo_model_comparison/week09_run01/previews/005.png)](../guides/research-archive.md) |
| numbers | `109` | `807` | `109` | recognition_error_review_crops | [Image (archive: evidence/photo_model_comparison/week09_run01/previews/006.png)](../guides/research-archive.md) |
| numbers | `999` | `494` | `494` | recognition_error_review_crops | [Image (archive: evidence/photo_model_comparison/week09_run01/previews/025.png)](../guides/research-archive.md) |
| characters | `O` | `O` | `0` | recognition_error_review_crops | [Image (archive: evidence/photo_model_comparison/week09_run01/previews/048.png)](../guides/research-archive.md) |
| characters | `l` | `l` | `1` | recognition_error_review_crops | [Image (archive: evidence/photo_model_comparison/week09_run01/previews/083.png)](../guides/research-archive.md) |
| characters | `1` | `I` | `1` | recognition_error_review_crops | [Image (archive: evidence/photo_model_comparison/week09_run01/previews/041.png)](../guides/research-archive.md) |
| characters | `Z` | `I` | `Z` | recognition_error_review_crops | [Image (archive: evidence/photo_model_comparison/week09_run01/previews/049.png)](../guides/research-archive.md) |
| characters | `o` | `O` | `0` | recognition_error_review_crops | [Image (archive: evidence/photo_model_comparison/week09_run01/previews/056.png)](../guides/research-archive.md) |
| characters | `q` | `1` | `Y` | recognition_error_review_crops | [Image (archive: evidence/photo_model_comparison/week09_run01/previews/057.png)](../guides/research-archive.md) |
| strings | `Ab12` | `Ab12` | `Abt2` | recognition_error_review_crops | [Image (archive: evidence/photo_model_comparison/week09_run01/previews/130.png)](../guides/research-archive.md) |
| strings | `007Ab` | `007Ab` | `D07Ab` | recognition_error_review_crops | [Image (archive: evidence/photo_model_comparison/week09_run01/previews/156.png)](../guides/research-archive.md) |
| strings | `ij12` | `ijS2` | `ij12` | recognition_error_review_crops | [Image (archive: evidence/photo_model_comparison/week09_run01/previews/123.png)](../guides/research-archive.md) |
| strings | `jig` | `ji5` | `jig` | recognition_error_review_crops | [Image (archive: evidence/photo_model_comparison/week09_run01/previews/124.png)](../guides/research-archive.md) |
| strings | `Ab12` | `AbI2` | `Abt2` | recognition_error_review_crops | [Image (archive: evidence/photo_model_comparison/week09_run01/previews/120.png)](../guides/research-archive.md) |
| strings | `Room7` | `ROOn7` | `R00M7` | recognition_error_review_crops | [Image (archive: evidence/photo_model_comparison/week09_run01/previews/121.png)](../guides/research-archive.md) |
| strings | `Hello` | `rIeIIO` | `rlell0` | count_mismatch | [Image (archive: evidence/photo_model_comparison/week09_run01/previews/122.png)](../guides/research-archive.md) |
| strings | `Bill` | `8pIII` | `BeIll` | count_mismatch | [Image (archive: evidence/photo_model_comparison/week09_run01/previews/125.png)](../guides/research-archive.md) |

## Frequent isolated-character confusions

| Model | Reference | Prediction | Count |
|---|---|---|---:|
| mlp | `g` | `9` | 3 |
| mlp | `q` | `1` | 3 |
| mlp | `O` | `0` | 3 |
| mlp | `o` | `0` | 3 |
| mlp | `1` | `I` | 2 |
| mlp | `Z` | `I` | 2 |
| mlp | `l` | `I` | 2 |
| mlp | `Z` | `2` | 2 |
| cnn | `O` | `0` | 4 |
| cnn | `o` | `0` | 4 |
| cnn | `g` | `9` | 3 |
| cnn | `z` | `Z` | 3 |
| cnn | `q` | `9` | 2 |
| cnn | `q` | `Y` | 1 |
| cnn | `Z` | `2` | 1 |
| cnn | `b` | `6` | 1 |

## Interpretation and next investigation

The CNN has higher exact accuracy in all four collections: 75% versus 55% on single digits, 90% versus 45% on numbers, 66.25% versus 53.75% on isolated characters, and 17.5% versus 7.5% on strings. String CER is also lower for CNN (35.625% versus 49.375%). This supports retaining the CNN baseline, while the low string accuracy remains a major limitation. MLP inference is faster in this local run, but shared preparation dominates total processing time.

Visual inspection of selected original/prepared previews identified these concrete investigation cases:

- Preview 113, Saniru lower-i: the prepared image is dominated by paper texture rather than a clear dot and stem. Both models fail. This is direct evidence of a preparation problem worth investigating.
- Preview 104, Saniru digit-1: the digit occupies a small, offset area in the prepared tile, with a stray mark near the top. Both models fail. Investigate localisation/normalisation; the preview does not prove that preparation alone caused the errors.
- Preview 122, Haresh Hello: H becomes two crops, producing six crops for five characters. Both models receive the same incorrect separation.
- Preview 125, Haresh Bill: the i dot and stem remain separate, producing five crops for four characters. This requires segmentation investigation.

Use the per-collection results rather than combining unrelated digit and character datasets into a single accuracy. A benchmark advantage does not guarantee a better prediction for every photograph. Retain the existing validation-selected CNN as the baseline; this experiment does not retrain or automatically change GUI defaults.

A crop-count mismatch establishes that the number of supplied regions differs from the reference length. Matching counts alone do not establish correct separation. Both classifiers failing on identical crops does not establish whether preparation or the model is responsible. Review originals and prepared crops before attributing causes.

Preprocessing investigation should inspect faint/thin strokes, oversized background crops and clipping, retaining before/after examples. Segmentation investigation should review count mismatches and misplaced dots. Model investigation should examine case and shape confusions after crop quality is assessed. Do not relabel ambiguous examples to match predictions.

## Timing and limitations

Shared preparation once per image; median of warmed repeated inference. Pipeline = preparation + inference, excludes file I/O, model load and export. Single machine, no timing significance claim.

Inference is warmed, CPU-only with four Torch threads; model execution order alternates and the median of five runs is used for each image/model by default. Shared preparation is timed once, so total pipeline figures are estimates from shared preparation plus inference, not GUI latency. No statistical speed claim is made.

These are reused development photos from four writers, not an independent final test. The 80 character photos cover only 20 of 62 labels. String examples repeat the same ten texts across writers. No causal or generalisation claim follows from this small collection. Frozen checkpoints and all original inputs are unchanged. Reserve fresh handwriting for the final evaluation.

## Reproduction and evidence

From the repository root with dependencies installed:

```powershell
python -m research.compare_photo_models --output evidence/photo_model_comparison/week09_run02
python -m research.report_photo_models --run evidence/photo_model_comparison/week09_run02 --output docs/reports/photo-model-comparison-week09.md
```

- [Summary JSON (archive: evidence/photo_model_comparison/week09_run01/summary.json)](../guides/research-archive.md)
- [All 320 model predictions (archive: evidence/photo_model_comparison/week09_run01/results.csv)](../guides/research-archive.md)
- [160 paired photo results (archive: evidence/photo_model_comparison/week09_run01/paired_results.csv)](../guides/research-archive.md)
- [Input manifest (archive: evidence/photo_model_comparison/week09_run01/manifest.csv)](../guides/research-archive.md)

The run also retains all prepared arrays, original/crop previews and source snapshots. The report generator validates checkpoint/input hashes, array contracts, pair coverage, summary counts and exact agreement with historical CNN predictions before writing this report.
