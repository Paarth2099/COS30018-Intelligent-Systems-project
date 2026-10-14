# Alphanumeric string segmentation experiment

Comparison of projection and grouped connected-component segmentation for single-line alphanumeric images.

## Outcome

**Selected method: `grouped_components`.** It is connected-component analysis with geometric grouping, not a trained neural segmentation model. It feeds complete character crops to Paarth's unchanged preparation and Haresh's frozen CNN.

| Final method | Correct crop count | Visually acceptable separation | Exact strings | Character error rate |
|---|---:|---:|---:|---:|
| projection | 32/40 | 32/40 | 7/40 (17.5%) | 65/160 (40.62%) |
| grouped_components | 38/40 | 38/40 | 7/40 (17.5%) | 57/160 (35.62%) |

All scores describe the same inspected **development set**, not a fresh final test. Exact-string accuracy is low even where segmentation is acceptable: a single wrong character makes the whole string incorrect. Do not describe crop-count agreement as recognition accuracy.

## Dataset and label controls

- 40 photographs, 10 from each writer; 160 reference characters in total.
- Strings: `Ab12`, `Room7`, `Hello`, `ij12`, `jig`, `Bill`, `007Ab`, `O0o`, `Il1`, `Zz25`.
- Paarth's seventh filename said `00Ab`, but the photographed text visibly contains `007Ab`. The filename was corrected before scoring; original image bytes were unchanged. The initial review sheet preserves the old filename.
- Intended case labels were otherwise retained, including ambiguous uppercase-looking lowercase handwriting. No label was altered to match a prediction.
- All photos were retained in every run. Reference text is used only after segmentation and prediction for evaluation.
- Byte-identical duplicate groups: 0. This does not detect re-encoded duplicates.
- The same CNN checkpoint and Paarth preprocessing source were held fixed across all runs. No retraining, spelling correction, expected-length hint or dictionary lookup was used.

## Method

Both methods use EXIF-corrected grayscale, bounded working resolution, the lighting correction used by Paarth, Otsu foreground separation and conservative speck removal. Small morphological closing repairs tiny gaps in the connectivity mask; it does not paint artificial strokes into the grayscale crops.

**Projection:** locate consecutive occupied image-column intervals. Vertically overlapping column ranges can merge neighbouring characters; horizontally offset dots can create extra intervals.

**Grouped components:** identify substantial character bodies, cluster nearby small dot fragments and attach plausible upper parts to nearby stems. Prefer horizontal alignment and a body below a dot over a tall neighbouring bounding box. Retain unassigned small parts with warnings rather than silently dropping them. Sort final groups left to right.

Each crop retains a pale margin and suppresses labelled ink from other groups. The pipeline calls `prepare_character()` on each crop and passes the stacked [N,28,28] float array to `CharacterPredictor.predict_text()`. Blank images produce an empty batch and string.

## Development history

| Run | Change | Projection count / exact / CER | Grouped count / exact / CER |
|---|---|---|---|
| 1 | Initial shared mask and dot grouping | 31/40; 7/40; 41.25% | 28/40; 4/40; 45.00% |
| 2 | Small-gap closing and revised short-body/fragment handling | 32/40; 7/40; 40.62% | 37/40; 6/40; 39.38% |
| 3 | Fragmented-dot clustering and compact-dot/body distinction | 32/40; 7/40; 40.62% | 38/40; 7/40; 36.88% |
| 4 | Prefer aligned stems over neighbouring boxes for dot assignment | 32/40; 7/40; 40.62% | 38/40; 7/40; 35.62% |

These are sequential development changes, sometimes involving multiple geometric rules. They are not independent experiments that isolate each rule's causal effect. Full outputs and source snapshots for every run are retained.

## Visual review and remaining failures

Final visual review inspected all 80 method/image combinations using annotated originals and ordered crop strips. A pass requires one crop per visible character in reading order with appropriate dot grouping, not a correct model prediction. This qualitative preview assessment is not independently annotated ground truth or a bounding-box IoU measurement.

Run 3 illustrates why counts alone are insufficient: grouped components matched 38/40 counts, but two of those attached an i dot to a neighbour. Its visual pass count was 36/40; that finding motivated run 4.

| Image | Method | Visual issue |
|---|---|---|
| Haresh/Haresh_02_text-Room7.jpg | projection | Merged m and 7 |
| Haresh/Haresh_03_text-Hello.jpg | projection | Broken H split into two crops |
| Haresh/Haresh_03_text-Hello.jpg | grouped_components | Broken H split into two crops |
| Haresh/Haresh_05_text-jig.jpg | projection | Fragmented i dot forms an extra crop |
| Haresh/Haresh_06_text-Bill.jpg | grouped_components | Detached i dot and stem remain separate |
| Haresh/Haresh_10_text-Zz25.jpg | projection | Merged 2 and 5 |
| Manula/Manula_02_text-Room7.jpg | projection | Merged m and 7 |
| Saniru/Saniru_03_text-Hello.jpeg | projection | Merged e and first l |
| Saniru/Saniru_05_text-jig.jpeg | projection | Detached i dot forms an extra crop |
| Saniru/Saniru_07_text-007Ab.jpeg | projection | Merged second zero and 7 |

Model errors remain after acceptable segmentation, including `O/o/0`, `I/l/1`, and case changes. Photo ink style, size normalisation and EMNIST-to-photo differences may contribute; this comparison does not isolate those causes. Failures and warnings remain visible in the evidence.

## Per-writer results for selected method

| Writer | Count matches | Exact strings | Character errors / reference characters |
|---|---:|---:|---:|
| Haresh | 8/10 | 4/10 | 14/40 |
| Manula | 10/10 | 1/10 | 13/40 |
| Paarth | 10/10 | 1/10 | 14/40 |
| Saniru | 10/10 | 1/10 | 16/40 |

These small samples mix handwriting, pen and photography differences; they do not rank team members.

## Metrics and timing

- Count agreement: number of predicted crops equals reference string length; no label is supplied to the segmenter.
- Exact-string accuracy: case-sensitive full equality, preserving leading zeros.
- Character error rate (CER): total Levenshtein insertions, deletions and substitutions divided by 160 reference characters. It is not clipped and can exceed 100% when there are many insertions.
- Visual separation: full-image pass/fail review of crops, independent of whether the classifier guessed correctly.

| Method | Mean segmentation | Mean preparation | Mean inference |
|---|---:|---:|---:|
| projection | 255.63 ms | 232.57 ms | 1.73 ms |
| grouped_components | 254.78 ms | 235.89 ms | 1.67 ms |

Single local CPU run with a warmed predictor; excludes loading and evidence export. Timings are not cross-machine guarantees.

## Verification and evidence

- At the segmentation experiment checkpoint, 47 software tests passed: the previous 33 plus 14 new segmentation/interface tests.
- Tests cover dots, fragmented dots, short stems, misleading neighbour boxes, small pen gaps, overlapping crop boxes, ordering, blank images, EXIF rotation and preservation of case/zeros.
- Existing digit GUI, digit algorithms, Paarth preparation and trained models remain unchanged.
- Every run has source/checkpoint/input hashes, per-image CSVs, masks, boxes, crop images, prepared arrays and predictions.
- Final default and review provenance are in `evidence/character_segmentation/selection.json`.

The component supports a single line of separated characters. Touching/cursive writing, multiline pages, word-space inference, severe shadows and very small detached parts remain limitations. The selected pipeline is integrated into the GUI. Fresh end-to-end testing and further error analysis remain necessary.

See [the segmentation guide](../guides/string-segmentation.md) for commands and the recognition interface.
