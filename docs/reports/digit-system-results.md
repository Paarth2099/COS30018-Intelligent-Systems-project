# Digit-system experiment results

The selected number pipeline uses CNN classification, connected-component segmentation and Otsu digit preparation.

## MNIST comparison

Both models used a seeded split of 54,000 training and 6,000 validation images, five epochs, Adam and learning rate 0.001. Validation selected the model before evaluation on 10,000 test digits.

| Model | Validation accuracy | Test accuracy |
|---|---:|---:|
| MLP | 96.22% | 96.60% |
| CNN | 98.42% | 98.53% |

The selected CNN reconstructed 957/1,000 constructed strings exactly. Digit boundaries were known and source images came from the same MNIST test set, so this does not measure independent photo or segmentation performance.

Evidence: [model selection](../../experiments/baseline/selection.json), [benchmark results](../../experiments/baseline/evaluation/results.json), [constructed-string preview](../../experiments/baseline/evaluation/generated_strings.png).

## Preprocessing

Twenty single-digit development photos were compared using the same CNN, common Otsu localisation and the same resizing/padding.

| Representation | Correct predictions |
|---|---:|
| Grayscale | 9/20 (45%) |
| Fixed threshold | 15/20 (75%) |
| Otsu threshold | 15/20 (75%) |

Fixed and Otsu tied. These are pixel-representation comparisons after a shared crop, not three independent localisation algorithms.

Evidence: [preprocessing summary](../../evidence/preprocessing/team_single_digits_run_01/summary.json).

## Complete photo pipeline

The collection contains 20 single-digit and 20 multi-digit photos. Both segmentation methods share foreground preparation and the same CNN.

| Method | Correct single digits | Exact multi-digit numbers | Matching crop count, all photos |
|---|---:|---:|---:|
| Connected components | 15/20 | 18/20 | 40/40 |
| Projection | 15/20 | 17/20 | 39/40 |

Connected components recognised 33/40 photos exactly. Count agreement does not prove pixel-level segmentation quality. These images were used for development and do not establish unseen-handwriting performance.

Evidence: [summary](../../evidence/segmentation/team_run_02/summary.json), [per-image results](../../evidence/segmentation/team_run_02/results.csv), [selected method](../../evidence/segmentation/selection.json).

## Limitations

Touching strokes, shadows, clipped handwriting and overlapping horizontal ranges can cause failures. The benchmark uses one training seed and fixed hyperparameters. Model scores are uncalibrated. Further real-photo model comparison and independent evaluation remain outstanding.
