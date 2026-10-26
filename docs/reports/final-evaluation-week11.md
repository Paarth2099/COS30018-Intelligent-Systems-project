> Recorded study: results and original source hashes are retained. Detailed outputs are in the [research archive](../guides/research-archive.md); current commands use the `research` package.

# Week 11: final fresh-photo evaluation

Haresh's final model evaluation uses 120 new photographs from the four team writers. The user confirmed the assigned labels, tight camera framing (not post-capture cropping), and that none had previously been tested. No training, tuning or default-model changes were performed on these final results.

The selected defaults were fixed before inference: CNN for numbers and ResNet18 for alphanumeric inputs. Saniru's second-computer acceptance was reported complete by the user, who explicitly authorised proceeding while screenshots/environment records await upload. This report does not claim independent verification of that acceptance.

## Selected-system results

| Collection | Selected model | Correct / total | Exact accuracy | Character error rate |
| --- | --- | --- | --- | --- |
| single_digits | cnn | 37/40 | 92.50% | 7.50% |
| numbers | cnn | 16/20 | 80.00% | 8.82% |
| single_letters | resnet18 | 29/40 | 72.50% | 27.50% |
| strings | resnet18 | 4/20 | 20.00% | 27.50% |

Each collection has its own denominator; no combined overall accuracy is presented. Letter case and leading zeros are scored exactly. Single-input accuracy requires one correct character, so extra/missing outputs fail. CER is total edit distance divided by total reference characters; lower is better and it can exceed 100%. Errors remain in the denominator with empty-output scoring.

## Paired four-model comparison

Models receive exactly the same saved ordered crops within each collection. All single-letter photos use the GUI's grouped-component route, so segmentation can fail even for a single character. This differs from the older development isolated-character comparison's direct preparation route and prevents treating those percentages as directly controlled before/after measurements.

| Collection | Model | Correct / total | Accuracy | CER |
| --- | --- | --- | --- | --- |
| single_digits | mlp | 31/40 | 77.50% | 22.50% |
| single_digits | cnn | 37/40 | 92.50% | 7.50% |
| single_digits | resnet18 | 37/40 | 92.50% | 7.50% |
| single_digits | mobilenet_v3_small | 31/40 | 77.50% | 22.50% |
| numbers | mlp | 7/20 | 35.00% | 29.41% |
| numbers | cnn | 16/20 | 80.00% | 8.82% |
| numbers | resnet18 | 18/20 | 90.00% | 4.41% |
| numbers | mobilenet_v3_small | 10/20 | 50.00% | 22.06% |
| single_letters | mlp | 27/40 | 67.50% | 32.50% |
| single_letters | cnn | 27/40 | 67.50% | 32.50% |
| single_letters | resnet18 | 29/40 | 72.50% | 27.50% |
| single_letters | mobilenet_v3_small | 27/40 | 67.50% | 32.50% |
| strings | mlp | 2/20 | 10.00% | 42.50% |
| strings | cnn | 6/20 | 30.00% | 25.00% |
| strings | resnet18 | 4/20 | 20.00% | 27.50% |
| strings | mobilenet_v3_small | 6/20 | 30.00% | 31.25% |

The comparison describes performance under the predeclared experiment. A model that outperforms the selected default here is not automatically promoted using this test set.

## Segmentation alternative

Projection is compared with components (numbers) or grouped components (strings), holding the selected model fixed. These are whole-pipeline recognition scores, not manual segmentation accuracy. Manula must still review complete character crops, dot grouping, reading order and split/merge failures. Crop counts alone cannot establish segmentation success.

| Collection | Method | Model | Correct / total | CER |
| --- | --- | --- | --- | --- |
| numbers | components | cnn | 16/20 | 8.82% |
| numbers | projection | cnn | 16/20 | 8.82% |
| strings | grouped_components | resnet18 | 4/20 | 27.50% |
| strings | projection | resnet18 | 4/20 | 31.25% |

## Auxiliary: digits with the 62-class alphanumeric models

These reuse the 40 single-digit photographs; they are not 40 additional independent samples or part of the primary 40-letter score.

| Model | Correct / 40 | Accuracy | CER |
| --- | --- | --- | --- |
| mlp | 34 | 85.00% | 15.00% |
| cnn | 36 | 90.00% | 10.00% |
| resnet18 | 39 | 97.50% | 2.50% |
| mobilenet_v3_small | 31 | 77.50% | 22.50% |

## Per-writer selected-system results

| Writer | Collection | Correct / total | CER |
| --- | --- | --- | --- |
| Haresh | single_digits | 9/10 | 10.00% |
| Haresh | numbers | 5/5 | 0.00% |
| Haresh | single_letters | 9/10 | 10.00% |
| Haresh | strings | 1/5 | 25.00% |
| Paarth | single_digits | 8/10 | 20.00% |
| Paarth | numbers | 1/5 | 35.29% |
| Paarth | single_letters | 7/10 | 30.00% |
| Paarth | strings | 2/5 | 20.00% |
| Manula | single_digits | 10/10 | 0.00% |
| Manula | numbers | 5/5 | 0.00% |
| Manula | single_letters | 6/10 | 40.00% |
| Manula | strings | 0/5 | 40.00% |
| Saniru | single_digits | 10/10 | 0.00% |
| Saniru | numbers | 5/5 | 0.00% |
| Saniru | single_letters | 7/10 | 30.00% |
| Saniru | strings | 1/5 | 25.00% |

## Timing

CPU inference with four threads; models and preparation routes warmed on development photos. Each configuration has five repeated model-only and end-to-end calls per photo, with model order rotated. End-to-end means preprocessing, segmentation and inference; image loading, model loading, exports and GUI interaction are excluded. Repeats are timing observations, not new samples. Collection median and p95 below are calculated from each photo's median duration.

| Collection | Model | Median end-to-end ms | p95 end-to-end ms |
| --- | --- | --- | --- |
| single_digits | cnn | 43.09 | 132.54 |
| numbers | cnn | 57.84 | 99.78 |
| single_letters | resnet18 | 502.43 | 1080.23 |
| strings | resnet18 | 1031.82 | 1316.32 |

Environment: Windows-11-10.0.26200-SP0; Python 3.12.10; PyTorch 2.6.0+cpu. Full raw timing repeats and model-only medians are retained in predictions.json.

## All selected-system errors

These are all failed selected-system samples, not a favourable subset. Crop counts and predictions are observations, not a diagnosis of the underlying cause. Paarth and Manula should use the saved crops/annotations for their respective reviews.

| Sample | Reference | Prediction | Crops | Evidence |
| --- | --- | --- | --- | --- |
| Haresh_single_digits_10 | `9` | `4` | 1 | [annotation (archive: evidence/final_evaluation/run_01/samples/Haresh_single_digits_10/mnist_components/segmentation.png)](../guides/research-archive.md) |
| Haresh_single_letters_08 | `c` | `C` | 1 | [annotation (archive: evidence/final_evaluation/run_01/samples/Haresh_single_letters_08/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Haresh_strings_01 | `UvW1` | `PvW1` | 4 | [annotation (archive: evidence/final_evaluation/run_01/samples/Haresh_strings_01/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Haresh_strings_02 | `XyZ2` | `XYZ2` | 4 | [annotation (archive: evidence/final_evaluation/run_01/samples/Haresh_strings_02/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Haresh_strings_03 | `uVw3` | `UVw3` | 4 | [annotation (archive: evidence/final_evaluation/run_01/samples/Haresh_strings_03/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Haresh_strings_04 | `xYz4` | `BDz4` | 4 | [annotation (archive: evidence/final_evaluation/run_01/samples/Haresh_strings_04/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Paarth_single_digits_05 | `4` | `7` | 1 | [annotation (archive: evidence/final_evaluation/run_01/samples/Paarth_single_digits_05/mnist_components/segmentation.png)](../guides/research-archive.md) |
| Paarth_single_digits_10 | `9` | `7` | 1 | [annotation (archive: evidence/final_evaluation/run_01/samples/Paarth_single_digits_10/mnist_components/segmentation.png)](../guides/research-archive.md) |
| Paarth_numbers_02 | `815` | `803` | 3 | [annotation (archive: evidence/final_evaluation/run_01/samples/Paarth_numbers_02/mnist_components/segmentation.png)](../guides/research-archive.md) |
| Paarth_numbers_03 | `462` | `6162` | 4 | [annotation (archive: evidence/final_evaluation/run_01/samples/Paarth_numbers_03/mnist_components/segmentation.png)](../guides/research-archive.md) |
| Paarth_numbers_04 | `9071` | `7071` | 4 | [annotation (archive: evidence/final_evaluation/run_01/samples/Paarth_numbers_04/mnist_components/segmentation.png)](../guides/research-archive.md) |
| Paarth_numbers_05 | `2386` | `2385` | 4 | [annotation (archive: evidence/final_evaluation/run_01/samples/Paarth_numbers_05/mnist_components/segmentation.png)](../guides/research-archive.md) |
| Paarth_single_letters_06 | `f` | `F` | 1 | [annotation (archive: evidence/final_evaluation/run_01/samples/Paarth_single_letters_06/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Paarth_single_letters_07 | `g` | `9` | 1 | [annotation (archive: evidence/final_evaluation/run_01/samples/Paarth_single_letters_07/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Paarth_single_letters_09 | `i` | `6` | 1 | [annotation (archive: evidence/final_evaluation/run_01/samples/Paarth_single_letters_09/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Paarth_strings_02 | `dEf7` | `dEF7` | 4 | [annotation (archive: evidence/final_evaluation/run_01/samples/Paarth_strings_02/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Paarth_strings_04 | `jKl8` | `jkl8` | 4 | [annotation (archive: evidence/final_evaluation/run_01/samples/Paarth_strings_04/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Paarth_strings_05 | `O0l1` | `00L1` | 4 | [annotation (archive: evidence/final_evaluation/run_01/samples/Paarth_strings_05/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Manula_single_letters_05 | `O` | `0` | 1 | [annotation (archive: evidence/final_evaluation/run_01/samples/Manula_single_letters_05/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Manula_single_letters_07 | `l` | `1` | 1 | [annotation (archive: evidence/final_evaluation/run_01/samples/Manula_single_letters_07/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Manula_single_letters_08 | `m` | `M` | 1 | [annotation (archive: evidence/final_evaluation/run_01/samples/Manula_single_letters_08/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Manula_single_letters_10 | `o` | `0` | 1 | [annotation (archive: evidence/final_evaluation/run_01/samples/Manula_single_letters_10/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Manula_strings_01 | `MnO3` | `Mn53` | 4 | [annotation (archive: evidence/final_evaluation/run_01/samples/Manula_strings_01/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Manula_strings_02 | `pQr6` | `PQrb` | 4 | [annotation (archive: evidence/final_evaluation/run_01/samples/Manula_strings_02/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Manula_strings_03 | `StU5` | `5tU5` | 4 | [annotation (archive: evidence/final_evaluation/run_01/samples/Manula_strings_03/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Manula_strings_04 | `vWx9` | `VWX9` | 4 | [annotation (archive: evidence/final_evaluation/run_01/samples/Manula_strings_04/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Manula_strings_05 | `Ji10` | `jil0` | 4 | [annotation (archive: evidence/final_evaluation/run_01/samples/Manula_strings_05/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Saniru_single_letters_06 | `p` | `P` | 1 | [annotation (archive: evidence/final_evaluation/run_01/samples/Saniru_single_letters_06/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Saniru_single_letters_08 | `r` | `T` | 1 | [annotation (archive: evidence/final_evaluation/run_01/samples/Saniru_single_letters_08/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Saniru_single_letters_09 | `s` | `S` | 1 | [annotation (archive: evidence/final_evaluation/run_01/samples/Saniru_single_letters_09/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Saniru_strings_01 | `YzA4` | `YZA4` | 4 | [annotation (archive: evidence/final_evaluation/run_01/samples/Saniru_strings_01/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Saniru_strings_02 | `bCd8` | `b6d8` | 4 | [annotation (archive: evidence/final_evaluation/run_01/samples/Saniru_strings_02/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Saniru_strings_03 | `EfG2` | `EF62` | 4 | [annotation (archive: evidence/final_evaluation/run_01/samples/Saniru_strings_03/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |
| Saniru_strings_04 | `hIj6` | `hIJ6` | 4 | [annotation (archive: evidence/final_evaluation/run_01/samples/Saniru_strings_04/emnist_grouped_components/segmentation.png)](../guides/research-archive.md) |

Runtime errors across all comparison rows: 0.

## Limitations and evidence boundaries

- Four known team writers, only 120 photographs and deliberately selected content. This is new-input evaluation from known writers, not unseen-writer generalisation or a representative population study. Class coverage is sparse; per-class confusion counts are descriptive.
- Tight framing is user-reported camera framing. Newness/untested status and exact intended case are user attestations, not independently observed capture events. Individual writing dates remain unspecified.
- Human-assigned case can be visually ambiguous. Preserve the confirmed ground truth and disclose this limitation rather than relabel from model output.
- The planned light-stroke and close-spacing conditions were not verified as controlled interventions; no causal robustness claim follows from their labels.
- No hyperparameter search, repeated-seed retraining or new official benchmark evaluation was performed here. The existing Week 11 report retains the benchmark/training evidence.
- A separate teammate label inspection was not claimed; the recorded evidence is assistant visual review plus Paarth's collective confirmation.
- Manual segmentation and preprocessing error attribution, acceptance evidence upload, full team report and video remain outstanding. This is Haresh's component report, not the complete submission.

## Reproducibility and validation

Sixteen saved development predictions were reproduced before first final-photo inference. The locked manifest and frozen configuration were written before scoring. The code state is the recorded base Git commit plus exact archived runtime/evaluator source hashes; evaluation files were uncommitted and no new commit is invented. All 120 input hashes, eight checkpoint hashes and source hashes remained unchanged after the run.

There are 680 scored rows: 480 primary four-model comparisons, 40 projection comparisons and 160 auxiliary alphanumeric-digit comparisons. Independent verification recalculated every edit distance and exact match, collection metrics and timing medians, checked array shape/range, and verified archived sources and input/checkpoint hashes.

- [Frozen configuration](../../results/final_evaluation/frozen_configuration.json)
- [Locked manifest](../../results/final_evaluation/locked_manifest.json)
- [Development reproduction (archive: evidence/final_evaluation/run_01/development_reproduction.json)](../guides/research-archive.md)
- [All predictions and timing repeats (archive: evidence/final_evaluation/run_01/predictions.json)](../guides/research-archive.md)
- [Summary](../../results/final_evaluation/summary.json)
- [Per-writer metrics](../../results/final_evaluation/by_writer.json)
- [Confusion counts](../../results/final_evaluation/confusion_counts.json)
- [Paired model changes](../../results/final_evaluation/paired_changes.json)
- [Run integrity (archive: evidence/final_evaluation/run_01/verification.json)](../guides/research-archive.md)
- [Independent verification](../../results/final_evaluation/independent_verification.json)

`evaluate_final_photos.py` refuses to overwrite run_01. Do not rerun final inference for tuning. To check saved evidence without recognition, run `python -m research.verify_final_results`. To regenerate this report from saved results, run `python -m research.report_final_photos`.
