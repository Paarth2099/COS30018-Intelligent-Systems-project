# Segmentation experiment results

Owner: Manula | COS30018 Option B | Development experiment `team_run_02`

## Decision

Use connected components as the provisional integration default. It matched the expected digit count on 40/40 photos and recognised 18/20 multi-digit numbers exactly. Projection merged two digits in Manula 420. Selection uses development evidence; it does not establish performance on new writers or harder inputs.

## Data and method

All 40 images from the current team photo folder were copied without changing originals into data/segmentation_photos. Each member supplied five single-digit and five multi-digit images. All 80 method-image combinations completed. No failure was excluded. Labels were read from filenames for scoring only; neither the expected count nor the reference string is passed into segmentation.

Both methods share orientation correction, grayscale conversion, maximum image dimension 1000, autocontrast, Otsu foreground extraction, and small-component filtering. The minimum component area is max(3, 1% of largest component area); minimum height is max(2, 10% of tallest component height). These are fixed heuristic settings, not tuned per image.

Connected components uses 8-neighbour connectivity, grouping row runs with union-find and sorting boxes by their left coordinate. Projection defines P(x) as the number of retained foreground pixels in column x and groups maximal consecutive intervals where P(x)>0. This requires empty columns between adjacent digits.

Each crop has a small real-background margin. Other labelled foreground is removed from a component crop by replacement with estimated local paper intensity. Paarth's unchanged prepare_variants then makes the Otsu 28x28 float input for the same unchanged saved CNN. This is a different cropping path from Paarth's earlier standalone photo experiment, so individual predictions may differ; it is not evidence of a changed CNN.

## Quantitative results

| Subset | Method | Count matches | Exact recognition | Character error rate |
|---|---|---|---|---|
| single | components | 20/20 | 15/20 (75.0%) | 5/20 (25.00%) |
| single | projection | 20/20 | 15/20 (75.0%) | 5/20 (25.00%) |
| multi | components | 20/20 | 18/20 (90.0%) | 3/58 (5.17%) |
| multi | projection | 19/20 | 17/20 (85.0%) | 5/58 (8.62%) |
| all | components | 40/40 | 33/40 (82.5%) | 8/78 (10.26%) |
| all | projection | 39/40 | 32/40 (80.0%) | 10/78 (12.82%) |

Count matching is a proxy, not a ground-truth segmentation score. There are no annotated boxes and no IoU measurement. CER is Levenshtein edit distance divided by reference characters, not 1 minus character accuracy. Exact recognition requires the entire string, including leading zeros, to match.

## Multi-digit predictions

| Image | Label | Components | Projection |
|---|---|---|---|
| Haresh_06_label-14 | 14 | 14 | 14 |
| Haresh_07_label-109 | 109 | 109 | 109 |
| Haresh_08_label-1150 | 1150 | 1150 | 1150 |
| Haresh_09_label-1020 | 1020 | 1020 | 1020 |
| Haresh_10_label-28 | 28 | 28 | 28 |
| Manula_06_label-81 | 81 | 81 | 81 |
| Manula_07_label-420 | 420 | 420 | 47 |
| Manula_08_label-615 | 615 | 615 | 615 |
| Manula_09_label-6917 | 6917 | 6917 | 6917 |
| Manula_10_label-485 | 485 | 485 | 485 |
| Paarth_06_label-999 | 999 | 494 | 494 |
| Paarth_07_label-103 | 103 | 103 | 103 |
| Paarth_08_label-69 | 69 | 69 | 69 |
| Paarth_09_label-67 | 67 | 67 | 67 |
| Paarth_10_label-26 | 26 | 26 | 26 |
| Saniru_06_label-31 | 31 | 31 | 31 |
| Saniru_07_label-1600 | 1600 | 1600 | 1600 |
| Saniru_08_label-240 | 240 | 140 | 140 |
| Saniru_09_label-8765 | 8765 | 8765 | 8765 |
| Saniru_10_label-666 | 666 | 666 | 666 |

## Failure analysis and visual inspection

- Manula 420: the 2 and 0 are separate connected regions but their occupied columns overlap. Components returns three crops and predicts 420. Projection returns two crops and predicts 47. This demonstrates the methods' different assumptions.
- Paarth 999: both methods find three regions but the CNN returns 494. The crops retain the visible digits; no definitive model-error cause is claimed.
- Saniru 240: both methods find three regions but return 140. The first digit is misrecognised despite the correct count.
- Five single-digit errors also remain: Haresh 9, Manula 7 and 9, Paarth 9, Saniru 9. Complete predictions are in results.csv.
- The 20 multi-digit component previews were inspected for ordering and major crop defects. This qualitative review is not an independent annotated-box evaluation.

## Correction between runs

Run 01 revealed neighbour ink inside the rectangular crop for the 2 in Manula 420. Run 02 removes other component labels from each crop. The visible contamination was corrected without changing aggregate predictions or count results. A regression test checks that an overlapping neighbouring component does not leak into the crop. Run 01 and its original sources remain available; run 02 snapshots its source files and hashes.

## Verification

All 13 automated tests passed in the project Python environment: five existing ML contract tests and eight segmentation tests. They cover diagonal connectivity, sorting, dtype/range/shape, blank input, noise removal, disconnected parts, touching-digit limitations, overlapping boxes, edit distance and existing leading-zero handling. Synthetic geometry tests are software checks, not handwriting benchmarks.
The standalone recognise_photo.py demo on Manula 420 returned text 420. Its output is evidence/segmentation/manula_demo_01. All 80 previews exist; the run's saved source and input hashes were checked after execution.

## Limits and next work

Inputs should be one horizontal line of separated dark digits on pale paper. Touching digits, disconnected strokes, severe shadows, paper boundaries and multiple lines remain known failure modes. All examples are development images from four team members. Test on a new reserved collection before claiming final performance. The GUI, number-image generation and approved extension are outside this implementation.
Saniru can integrate recognise_number(image, predictor, method="components") into the GUI. Manula should reproduce the 420 demonstration, explain why projection fails there, and record actual learning and contribution. This code, experiment execution and documentation were produced with Codex assistance, not independent work or invented hours.

## Sources and records

- OpenCV connected-component concepts: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html
- OpenCV Otsu explanation: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
- Primary experiment evidence: evidence/segmentation/team_run_02/results.csv and summary.json.
- The manifest contains the checkpoint, input and source hashes. Record folder: evidence/segmentation/team_run_02.
- Beginner instructions and API handover: docs/MANULA_START_HERE.md.
