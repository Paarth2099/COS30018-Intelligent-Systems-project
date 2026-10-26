> Recorded study: results and original source hashes are retained. Detailed outputs are in the [research archive](../guides/research-archive.md); current commands use the `research` package.

# Week 11: four-architecture comparison

**Scope:** Haresh's model investigation following tutor feedback. Four architectures are now implemented and evaluated for both digit and alphanumeric recognition. This report does not claim completion of the whole project, fresh final evaluation or GUI integration of the new models.

## Outcome

- **MNIST:** validation candidate `resnet18`; development-gated recommendation **`cnn`**.
- **EMNIST:** validation candidate `resnet18`; development-gated recommendation **`resnet18`**.

The recommendation follows the recorded validation ranking and development no-regression gates. Official-test results were excluded from these decisions. The GUI continues to use its existing MLP/CNN checkpoints until an explicit integration and acceptance step.

ResNet-18 improves single-digit photo recognition from 16/20 to 19/20, but exact multi-digit recognition falls from 19/20 to 17/20. The number system therefore retains the baseline CNN under the recorded no-regression rule. The better MNIST benchmark alone does not justify replacement.

For alphanumerics, ResNet-18 improves isolated characters from 53/80 to 54/80 and complete strings from 8/40 to 12/40. String CER changes from 33.12% to 26.88%. This supports integration as the alphanumeric candidate, but most complete strings are still wrong; it is not evidence of reliable unrestricted text recognition.

## Motivation and models

The tutor requested two more architectures in addition to MLP and the baseline CNN. ResNet-18 adds residual blocks, and MobileNetV3-Small uses a compact convolutional design. Both are CNN architectures; the original custom network is therefore named baseline CNN throughout this comparison. The Week 10 class-weighted CNN is a training variant, not a fifth architecture.

The new models use Torchvision 0.21 with no pretrained weights. Only the first convolution is adapted to one channel and the classifier to 10 or 62 classes. Default widths, strides and pooling remain intact. Inputs stay 28x28, scaled to [0,1]. This is an adapted handwriting experiment, not a reproduction of published ImageNet performance.

## Experimental controls and provenance

- MNIST: 54,000 training and 6,000 validation examples. EMNIST ByClass: 628,142 training and 69,790 validation examples. Split hashes match the existing baselines.
- Five epochs, seed 42, batch 256, Adam learning rate 0.001, no weight decay, no augmentation, unweighted cross entropy. All original baseline checkpoints are preserved.
- Checkpoints selected by highest validation accuracy, then lower validation loss; earlier epoch retained on exact ties. EMNIST MobileNet can therefore retain epoch 4 even though five epochs completed.
- New runs trained on the RTX 3070 using float32 and deterministic settings, without AMP/TF32. Historical baselines trained on CPU; device and OS differ. Do not interpret historical training times as a controlled speed comparison.
- The imported ZIP, all four checkpoint hashes, source snapshots, five-row histories and best-epoch choices were checked. The PC run had no Git commit identifier; source hashes match the local files.
- Current Week 10 preprocessing and segmentation prepare each development photo once. All four models receive the same ordered arrays. These are previously used development images, not new held-out photos.
- Evaluation uses cpu with 4 threads on one computer. Photo latency is the mean of per-photo medians over five warmed calls, with rotating model order and preparation excluded.

## Validation and model size

| Dataset | Architecture | Best epoch | Validation accuracy | Parameters | Checkpoint MiB |
| --- | --- | --- | --- | --- | --- |
| mnist | mlp | 5 | 96.22% | 109,386 | 0.42 |
| mnist | cnn | 5 | 98.42% | 105,866 | 0.41 |
| mnist | resnet18 | 5 | 98.70% | 11,175,370 | 42.70 |
| mnist | mobilenet_v3_small | 5 | 97.22% | 1,527,818 | 5.95 |
| emnist | mlp | 5 | 83.19% | 112,766 | 0.43 |
| emnist | cnn | 5 | 85.51% | 109,246 | 0.42 |
| emnist | resnet18 | 5 | 87.16% | 11,202,046 | 42.81 |
| emnist | mobilenet_v3_small | 4 | 85.24% | 1,581,118 | 6.15 |

## Official test benchmarks

These results evaluate frozen checkpoints. They were not used for tuning or promotion. Macro F1 gives equal weight to each class; overall accuracy weights each example equally.

| Dataset | Architecture | Correct / total | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- |
| mnist | mlp | 9660 / 10000 | 96.60% | 0.9657 |
| mnist | cnn | 9853 / 10000 | 98.53% | 0.9853 |
| mnist | resnet18 | 9864 / 10000 | 98.64% | 0.9862 |
| mnist | mobilenet_v3_small | 9716 / 10000 | 97.16% | 0.9713 |
| emnist | mlp | 96973 / 116323 | 83.37% | 0.6901 |
| emnist | cnn | 99563 / 116323 | 85.59% | 0.7215 |
| emnist | resnet18 | 101315 / 116323 | 87.10% | 0.7518 |
| emnist | mobilenet_v3_small | 98994 / 116323 | 85.10% | 0.7086 |

## Existing handwriting development photos

Exact accuracy requires the entire predicted string to match, including case and leading zeros. Character error rate (CER) is total edit distance divided by total reference characters. A correct crop count does not imply correct recognition.

| Collection | Architecture | Exact correct / total | Exact accuracy | CER | Photo inference ms |
| --- | --- | --- | --- | --- | --- |
| single_digits | mlp | 11 / 20 | 55.00% | 45.00% | 0.437 |
| single_digits | cnn | 16 / 20 | 80.00% | 20.00% | 0.682 |
| single_digits | resnet18 | 19 / 20 | 95.00% | 5.00% | 9.571 |
| single_digits | mobilenet_v3_small | 12 / 20 | 60.00% | 40.00% | 8.538 |
| numbers | mlp | 9 / 20 | 45.00% | 24.14% | 0.466 |
| numbers | cnn | 19 / 20 | 95.00% | 3.45% | 1.169 |
| numbers | resnet18 | 17 / 20 | 85.00% | 8.62% | 14.708 |
| numbers | mobilenet_v3_small | 10 / 20 | 50.00% | 22.41% | 15.262 |
| characters | mlp | 43 / 80 | 53.75% | 46.25% | 0.483 |
| characters | cnn | 53 / 80 | 66.25% | 33.75% | 0.775 |
| characters | resnet18 | 54 / 80 | 67.50% | 32.50% | 10.533 |
| characters | mobilenet_v3_small | 45 / 80 | 56.25% | 43.75% | 10.016 |
| strings | mlp | 3 / 40 | 7.50% | 46.88% | 0.515 |
| strings | cnn | 8 / 40 | 20.00% | 33.12% | 1.360 |
| strings | resnet18 | 12 / 40 | 30.00% | 26.88% | 16.702 |
| strings | mobilenet_v3_small | 4 / 40 | 10.00% | 48.75% | 17.783 |

## Paired changes against the baseline CNN

| Collection | Architecture | Newly correct | Newly wrong | Both correct | Both wrong |
| --- | --- | --- | --- | --- | --- |
| single_digits | mlp | 1 | 6 | 10 | 3 |
| single_digits | resnet18 | 3 | 0 | 16 | 1 |
| single_digits | mobilenet_v3_small | 1 | 5 | 11 | 3 |
| numbers | mlp | 0 | 10 | 9 | 1 |
| numbers | resnet18 | 0 | 2 | 17 | 1 |
| numbers | mobilenet_v3_small | 0 | 9 | 10 | 1 |
| characters | mlp | 3 | 13 | 40 | 24 |
| characters | resnet18 | 2 | 1 | 52 | 25 |
| characters | mobilenet_v3_small | 3 | 11 | 42 | 24 |
| strings | mlp | 2 | 7 | 1 | 30 |
| strings | resnet18 | 5 | 1 | 7 | 27 |
| strings | mobilenet_v3_small | 1 | 5 | 3 | 31 |

## Representative changes

Examples below are the first two gains and first two losses in dataset order for ResNet-18 in each task, rather than hand-picked favourable cases. All images remain in aggregate results.

| Task | Change | Reference | Baseline CNN | ResNet-18 | Evidence |
| --- | --- | --- | --- | --- | --- |
| mnist | gain | 7 | 2 | 7 | [Image 11 (archive: evidence/model_comparison/week11_run01/previews/011.png)](../guides/research-archive.md) |
| mnist | gain | 9 | 4 | 9 | [Image 14 (archive: evidence/model_comparison/week11_run01/previews/014.png)](../guides/research-archive.md) |
| mnist | loss | 485 | 485 | 455 | [Image 19 (archive: evidence/model_comparison/week11_run01/previews/019.png)](../guides/research-archive.md) |
| mnist | loss | 69 | 69 | 64 | [Image 27 (archive: evidence/model_comparison/week11_run01/previews/027.png)](../guides/research-archive.md) |
| emnist | gain | q | 8 | q | [Image 79 (archive: evidence/model_comparison/week11_run01/previews/079.png)](../guides/research-archive.md) |
| emnist | gain | q | 9 | q | [Image 117 (archive: evidence/model_comparison/week11_run01/previews/117.png)](../guides/research-archive.md) |
| emnist | loss | z | z | 2 | [Image 59 (archive: evidence/model_comparison/week11_run01/previews/059.png)](../guides/research-archive.md) |
| emnist | loss | 007Ab | 007Ab | 007AB | [Image 136 (archive: evidence/model_comparison/week11_run01/previews/136.png)](../guides/research-archive.md) |

## Selection gates

For each task the candidate is the highest-validation model. Promotion additionally requires no reduction in exact accuracy and no increase in CER versus the baseline CNN on either development collection for that task. If any gate fails, retain the baseline while documenting the candidate and its limitations. This deliberately avoids choosing a different model after inspecting official-test scores.

| Task | Gate | Passed |
| --- | --- | --- |
| mnist | single_digits_exact_no_regression | True |
| mnist | single_digits_cer_no_regression | True |
| mnist | numbers_exact_no_regression | False |
| mnist | numbers_cer_no_regression | False |
| emnist | characters_exact_no_regression | True |
| emnist | characters_cer_no_regression | True |
| emnist | strings_exact_no_regression | True |
| emnist | strings_cer_no_regression | True |

## Limitations and remaining work

- One seed and five epochs per architecture. These results establish this fixed-budget comparison, not the maximum attainable accuracy of each architecture.
- GPU versus historical CPU training introduces a hardware/numerical difference. Deterministic settings do not guarantee identical results across hardware or PyTorch builds.
- Model size and default downsampling differ greatly. The default large-image strides may lose information on small handwriting images; this is a plausible limitation, not a separately tested explanation.
- Four team writers and reused development photos limit generalisation. Case ambiguity, similar symbols and segmentation errors remain possible. Do not describe these photos as an independent final test.
- Benchmark recognition operates on individual characters; photographed numbers/strings exercise the complete preparation and segmentation pipeline. Report these results separately.
- Complete selected-model GUI integration/acceptance, freeze configuration, then evaluate fresh handwriting without repeated tuning on that final set. Complete installation checks, team report and video separately.

## Evidence and reproduction

- [Import verification (archive: evidence/week11_gpu_setup/import_verification.json)](../guides/research-archive.md)
- [Plan and validation candidates (archive: evidence/model_comparison/week11_run01/plan.json)](../guides/research-archive.md)
- [Model metadata](../../results/four_models/models.csv)
- [Benchmark summary](../../results/four_models/benchmark_summary.csv)
- [Photo summary](../../results/four_models/photo_summary.csv)
- [Every photo prediction (archive: evidence/model_comparison/week11_run01/photo_predictions.csv)](../guides/research-archive.md)
- [Paired predictions (archive: evidence/model_comparison/week11_run01/paired_photos.csv)](../guides/research-archive.md)
- [Per-writer results](../../results/four_models/by_writer.csv)
- [Recommendation](../../results/four_models/recommendation.json)
- [Verification (archive: evidence/model_comparison/week11_run01/verification.json)](../guides/research-archive.md)
- [Independent recalculation and historical reproduction](../../results/four_models/independent_verification.json)
- Training records: `experiments/week11_gpu_run01/{mnist,emnist}/{resnet18,mobilenet_v3_small}/`.
- Each benchmark model directory contains predictions, per-class metrics, confusion matrix and first errors. Prepared photo arrays and 160 previews are retained with source snapshots.

```powershell
python -m research.evaluate_models_week11 --output evidence/model_comparison/week11_repeat01
python -m research.report_models_week11 --evaluation evidence/model_comparison/week11_repeat01 --report docs/reports/model-comparison-week11-repeat.md
```

## References

- [Torchvision 0.21 ResNet-18](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.resnet18.html)
- [Torchvision 0.21 MobileNetV3-Small](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.mobilenet_v3_small.html)
