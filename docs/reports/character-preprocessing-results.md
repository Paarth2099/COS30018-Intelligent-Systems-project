> Recorded study: results and original source hashes are retained. Detailed outputs are in the [research archive](../guides/research-archive.md); current commands use the `research` package.

# Character-photo preprocessing experiment

Comparison of character-photo preparation methods using a frozen EMNIST CNN.

## Outcome

**Selected development configuration:** local lighting correction, conservative speck removal, Otsu representation, aspect-preserving resize/padding and peak ink normalisation. The original digit pipeline and Haresh checkpoints were not changed.

The final fixed-threshold and Otsu variants each recognised **53/80 characters (66.25%)**, compared with **42/80 (52.5%)** for the original Otsu baseline: 11 more correct, or 13.75 percentage points. The declared tie preference retains Otsu; this is not evidence that Otsu is superior to fixed thresholding.

These photos were inspected and reused for development decisions. This is **development accuracy**, not untouched final-test accuracy or complete-string performance.

## Inputs and controls

- 80 single-character photographs: 20 each from Paarth, Haresh, Manula and Saniru.
- 20 distinct labels, four examples each: `0 1 2 5 8 A B I O Z a b g i j l o q t z`. This does not cover all 62 model classes.
- All images were used in every run; no hard example was dropped, replaced or assigned a label based on its prediction.
- Paarth confirmed that the file originally named upper-J was intended as lowercase j. Its name was corrected before the first experiment. Two underscore separators were standardised; image bytes were not changed.
- Labels come from explicit digit/upper/lower filenames; they are used only for scoring. The preparation and prediction functions never receive the reference label.
- Same frozen validation-selected CNN checkpoint and same input SHA-256 hashes across all four runs. No model retraining or EMNIST test-set tuning.
- Duplicate byte-identical photo groups: 0. Different encodings of the same photo would not be detected by a byte hash.

## Controlled development sequence

| Run | Change from preceding run | Grayscale | Fixed | Otsu | Photos with warnings |
|---|---|---:|---:|---:|---:|
| 1 | Original common-crop baseline | 24/80 (30.00%) | 35/80 (43.75%) | 42/80 (52.50%) | 14 |
| 2 | Add local lighting correction | 34/80 (42.50%) | 36/80 (45.00%) | 42/80 (52.50%) | 3 |
| 3 | Add conservative speck filtering | 43/80 (53.75%) | 44/80 (55.00%) | 51/80 (63.75%) | 3 |
| 4 | Add post-resize ink peak normalisation | 48/80 (60.00%) | 53/80 (66.25%) | 53/80 (66.25%) | 3 |

Lighting correction improved grayscale results and reduced warning flags; it did not increase the Otsu score by itself. Speck filtering improved crop localisation, and normalisation countered faint downsampled strokes. The before/after numbers are observed changes on these same development photos, not statistical evidence of generalisation.

All three representations share the localisation mask and geometry. Fixed uses intensity <=128; Otsu uses an image-derived cutoff; grayscale keeps intensities. Resampling leaves intermediate pixel values even for thresholded representations.

## Final results by writer

| Writer | Grayscale | Fixed | Otsu |
|---|---:|---:|---:|
| Haresh | 14/20 (70.0%) | 17/20 (85.0%) | 17/20 (85.0%) |
| Manula | 13/20 (65.0%) | 13/20 (65.0%) | 13/20 (65.0%) |
| Paarth | 10/20 (50.0%) | 11/20 (55.0%) | 12/20 (60.0%) |
| Saniru | 11/20 (55.0%) | 12/20 (60.0%) | 11/20 (55.0%) |

These small samples combine writer style, pen and photography differences. They are not rankings of team members or an isolated test of handwriting quality.

## Remaining errors

No photo raised a preprocessing exception in the four runs. That only means an array was produced; it does not mean every crop was correct. The selected pipeline misclassified 27/80 photos.

| Reference | Prediction | Count |
|---|---|---:|
| `O` | `0` | 4 |
| `o` | `0` | 4 |
| `g` | `9` | 3 |
| `z` | `Z` | 3 |
| `q` | `9` | 2 |
| `q` | `Y` | 1 |
| `Z` | `2` | 1 |
| `b` | `6` | 1 |
| `q` | `8` | 1 |
| `l` | `1` | 1 |
| `1` | `l` | 1 |
| `B` | `e` | 1 |
| `2` | `e` | 1 |
| `1` | `i` | 1 |
| `i` | `x` | 1 |
| `j` | `J` | 1 |

Uppercase O and lowercase o were often read as zero; g and q were confused with digits, and lowercase z with uppercase Z. Isolated character shape can be ambiguous, but these counts alone do not establish the cause of each error.

Three photos retain background/clipping warning flags:

- `Saniru/Saniru_02_digit-2.jpeg`: true `2`, predicted `e`. Foreground touches image edge: inspect clipping, shadow or paper boundary.
- `Saniru/Saniru_05_digit-1.jpeg`: true `1`, predicted `i`. Foreground touches image edge: inspect clipping, shadow or paper boundary.
- `Saniru/Saniru_14_lower-i.jpeg`: true `i`, predicted `x`. Foreground touches image edge: inspect clipping, shadow or paper boundary.

Saniru's digit 1 has an oversized/offset crop despite speck removal; the warning and preview retain this failure. Warning flags are advisory, not a complete detector of poor preparation. Some intended lowercase forms resemble uppercase handwriting; supplied labels were retained.

## Verification and evidence

- At the preprocessing experiment checkpoint, 33 software tests passed, including nine new preprocessing tests: dots and small detached parts, lighting gradient, thin strokes, EXIF rotation, blank/low-contrast input, clipping warnings, aspect ratio, non-mutation and label validation.
- Each run includes source snapshots, checkpoint hash, input manifest, 240 method/sample results, 80 per-photo comparisons, per-writer overviews, diagnostic JSON and prepared arrays.
- Inference uses an uncalibrated softmax score; it is not a probability that the prediction is correct.
- Timing in results.csv is local CPU timing. Shared preparation computes all three variants once per photo; inference is one prepared image. File loading and evidence export are excluded.

## Limitations and system integration

The code supports one dark character on pale paper. Whole-string splitting, spaces, joined handwriting and multiline pages are not implemented here. Tiny genuine components below max(3 pixels, 0.5% of the largest component area) can still be removed; inspect i/j previews when integrating new photo conditions. Strong shadows and texture remain limitations.

The string segmenter supplies complete character crops in left-to-right order with a small pale-paper margin and detached dots grouped with their stems. The GUI displays the resulting character predictions. These 80 development images are not an independent final test.

See [the preprocessing guide](../guides/character-preprocessing.md) for code locations, commands and the preparation interface.
