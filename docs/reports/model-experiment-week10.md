# Week 10: controlled class-weighted CNN experiment

**Decision: retain baseline.** The application still uses its original saved CNN; no default or baseline file was replaced.

## Question and rationale

Does moderate class-frequency weighting improve letter recognition without sacrificing validation or real-photo performance? Week 9 found frequent O/0/o, I/l/1 and case confusions. The training set is imbalanced. Class weighting tests one plausible contributor; it cannot recover case information absent from a handwritten shape or repair defective crops.

The only intentional training change is weighted cross entropy. Each class gets weight 1/sqrt(n), where n is its training-only count; weights are normalised to mean one. The square root is a moderate, preselected alternative to full inverse-frequency weighting, not a proven optimum. No exponent sweep or test-set-driven tuning was performed.

PyTorch supports per-class cross-entropy weights for imbalanced training. The project runs installed PyTorch 2.6.0; the public API documentation describes this weighting mechanism. See [PyTorch CrossEntropyLoss documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html).

## Controls

The CNN was chosen for this experiment because it outperformed MLP on all four Week 9 development collections. The MLP and the digit classifiers are retained unchanged; this experiment targets the 62-class CNN only.

- Same CNN architecture, random seed 42, train/validation indices, minibatch order, five epochs, batch size 256, Adam learning rate 0.001, zero weight decay and no augmentation. Training starts from scratch; this is not a warm start from the five-epoch baseline.
- 628,142 training examples and 69,790 validation examples. Weights use training labels only.
- Best epoch selected by highest unweighted validation accuracy, then lowest unweighted validation loss; retain earlier epoch on a tie.
- Existing official test results were not used for selection. The official test partition and fresh final photos were not loaded in this experiment.
- Exact Week 9 prepared arrays reused for all 80 character photos and 40 strings. Their image, array and baseline-source provenance is retained. No preprocessing or segmentation tuning was mixed into this comparison.

Selected candidate epoch: **5**. Training plus validation took 737.0 seconds on this computer. Runtime is descriptive, not a controlled speed comparison with the historical training run.

## Training history

| Epoch | Training accuracy | Validation accuracy | Unweighted validation loss |
|---|---:|---:|---:|
| 1 | 76.27% | 81.43% | 0.5510 |
| 2 | 82.64% | 83.21% | 0.4835 |
| 3 | 83.62% | 83.29% | 0.4680 |
| 4 | 84.21% | 83.93% | 0.4560 |
| 5 | 84.61% | 84.19% | 0.4413 |

Weighted training loss and unweighted baseline training loss optimise different objectives and should not be compared as equivalent quantities. The logged training loss is a sample-count-weighted average of weighted batch means. Validation loss remains unweighted.

## Validation comparison

| Model | Accuracy | Macro F1 | Digits | Uppercase | Lowercase |
|---|---:|---:|---:|---:|---:|
| baseline | 85.51% | 0.7212 | 95.10% | 81.25% | 70.22% |
| weighted | 84.19% | 0.7330 | 91.72% | 78.17% | 75.30% |

Macro F1 weights classes equally; accuracy weights examples equally. This validation set selected the checkpoint, so these are selection-set results, not an independent final generalisation estimate.

## Frozen development-photo comparison

| Collection | Model | Correct / total | Exact accuracy | CER | Inference ms |
|---|---|---:|---:|---:|---:|
| characters | baseline | 53/80 | 66.25% | 33.75% | 0.898 |
| characters | weighted | 58/80 | 72.50% | 27.50% | 0.896 |
| strings | baseline | 7/40 | 17.50% | 35.62% | 2.008 |
| strings | weighted | 5/40 | 12.50% | 42.50% | 2.015 |

Median of five warmed CPU inference calls per photo, four threads, alternating order. Preparation reused and excluded.

| Collection | Newly correct | Newly wrong | Both correct | Both wrong |
|---|---:|---:|---:|---:|
| characters | 7 | 2 | 51 | 20 |
| strings | 2 | 4 | 3 | 31 |

## Example gains, regressions and unresolved errors

Examples are the first matching entries in dataset order for each group, not a favourable subset used for scoring. All 120 inputs remain in the aggregate. Linked images show the unchanged source/crops and the earlier MLP/CNN labels; the new weighted prediction is in this table.

| Collection | Reference | Baseline CNN | Weighted CNN | Outcome | Source/crops |
|---|---|---|---|---|---|
| characters | `g` | `9` | `g` | gain | [Preview](../../evidence/photo_model_comparison/week09_run01/previews/072.png) |
| characters | `q` | `8` | `q` | gain | [Preview](../../evidence/photo_model_comparison/week09_run01/previews/079.png) |
| characters | `g` | `9` | `g` | gain | [Preview](../../evidence/photo_model_comparison/week09_run01/previews/080.png) |
| characters | `1` | `1` | `l` | regression | [Preview](../../evidence/photo_model_comparison/week09_run01/previews/061.png) |
| characters | `B` | `B` | `8` | regression | [Preview](../../evidence/photo_model_comparison/week09_run01/previews/066.png) |
| characters | `O` | `0` | `0` | unresolved | [Preview](../../evidence/photo_model_comparison/week09_run01/previews/048.png) |
| characters | `o` | `0` | `0` | unresolved | [Preview](../../evidence/photo_model_comparison/week09_run01/previews/056.png) |
| characters | `q` | `Y` | `4` | unresolved | [Preview](../../evidence/photo_model_comparison/week09_run01/previews/057.png) |
| strings | `jig` | `ji9` | `jig` | gain | [Preview](../../evidence/photo_model_comparison/week09_run01/previews/134.png) |
| strings | `jig` | `ji9` | `jig` | gain | [Preview](../../evidence/photo_model_comparison/week09_run01/previews/144.png) |
| strings | `ij12` | `ij12` | `iji2` | regression | [Preview](../../evidence/photo_model_comparison/week09_run01/previews/123.png) |
| strings | `Zz25` | `Zz25` | `zz25` | regression | [Preview](../../evidence/photo_model_comparison/week09_run01/previews/129.png) |
| strings | `007Ab` | `007Ab` | `007A8` | regression | [Preview](../../evidence/photo_model_comparison/week09_run01/previews/136.png) |
| strings | `Ab12` | `Abt2` | `Abg2` | unresolved | [Preview](../../evidence/photo_model_comparison/week09_run01/previews/120.png) |
| strings | `Room7` | `R00M7` | `R00m7` | unresolved | [Preview](../../evidence/photo_model_comparison/week09_run01/previews/121.png) |
| strings | `Hello` | `rlell0` | `rIeIl0` | unresolved | [Preview](../../evidence/photo_model_comparison/week09_run01/previews/122.png) |

## Selected class diagnostics

| Class | Train count | Weight | Baseline validation recall | Weighted validation recall |
|---|---:|---:|---:|---:|
| `0` | 31127 | 0.405 | 83.40% | 77.91% |
| `O` | 22485 | 0.477 | 52.08% | 56.97% |
| `o` | 2474 | 1.438 | 0.00% | 2.18% |
| `1` | 34537 | 0.385 | 91.01% | 80.61% |
| `I` | 10751 | 0.690 | 53.97% | 61.34% |
| `l` | 13786 | 0.609 | 14.03% | 24.54% |
| `Z` | 2431 | 1.451 | 69.63% | 66.67% |
| `z` | 2453 | 1.444 | 49.26% | 68.01% |
| `g` | 3318 | 1.242 | 48.24% | 55.28% |
| `q` | 2695 | 1.378 | 35.79% | 46.82% |

## Decision and limitations

The adoption rule was written before training: require no regression in validation accuracy, isolated-character exact accuracy, string exact accuracy and string CER. This conservative rule prevents selecting a change just because one attractive metric improves.

| Criterion | Met? |
|---|---|
| validation accuracy no regression | No |
| character accuracy no regression | Yes |
| string accuracy no regression | No |
| string cer no regression | No |

Result: **retain baseline**. Retain the candidate, its history and all negative results as research evidence. Do not replace the baseline checkpoint or relabel photos to improve the score.

This is one candidate with one seed and a fixed short training budget. It does not establish optimal weighting or run-to-run significance. The same development handwriting has been inspected repeatedly, covers only four writers, and the isolated set contains only 20 of 62 labels. Better recall for a rare class can come with more false positives elsewhere; inspect the full per-class/confusion evidence.

A model change does not repair the two known string segmentation failures or the three clear preparation defects. Independent preprocessing and segmentation experiments should use fixed comparison checkpoints. Fresh final evaluation remains for Week 11 after integration decisions.

## Evidence and reproduction

- [Prospective experiment plan](../../experiments/emnist_byclass/week10_weighted_run01/plan.json)
- [Training history](../../experiments/emnist_byclass/week10_weighted_run01/history.csv)
- [Training log](../../experiments/emnist_byclass/week10_weighted_run01/training_log.txt)
- [Training-only class weights](../../experiments/emnist_byclass/week10_weighted_run01/class_weights.csv)
- [Validation-selected checkpoint record](../../experiments/emnist_byclass/week10_weighted_run01/selection.json)
- [Complete metrics and adoption decision](../../evidence/model_experiment/week10_run01/results.json)
- [Verification checks](../../evidence/model_experiment/week10_run01/verification.json)
- [All paired photo predictions](../../evidence/model_experiment/week10_run01/photo_predictions.csv)
- [Per-writer results](../../evidence/model_experiment/week10_run01/by_writer.csv)
- [Baseline validation confusion matrix](../../evidence/model_experiment/week10_run01/baseline_validation_confusion.png)
- [Weighted validation confusion matrix](../../evidence/model_experiment/week10_run01/weighted_validation_confusion.png)

Source snapshots and validation prediction arrays are retained alongside the metrics. All 120 baseline photo predictions and the original baseline validation accuracy were reproduced before comparison.

```powershell
python -m hnrs_chars.week10_weighted --output experiments/emnist_byclass/week10_weighted_run02
python -m hnrs_chars.week10_evaluate --candidate experiments/emnist_byclass/week10_weighted_run02 --output evidence/model_experiment/week10_run02
python -m hnrs_chars.week10_report --experiment experiments/emnist_byclass/week10_weighted_run02 --evaluation evidence/model_experiment/week10_run02
```
