> Recorded study: results and original source hashes are retained. Detailed outputs are in the [research archive](../guides/research-archive.md); current commands use the `research` package.

# EMNIST ByClass baseline experiment results

Controlled comparison of MLP and CNN character classifiers.
These results measure isolated character classification. Photo preprocessing, string segmentation and GUI evaluation are reported separately.

## Method

The saved run uses 628,142 training and 69,790 validation images from the original EMNIST training partition. The official 116,323-image test partition was evaluated only after validation selected a model.
The verified 62 labels preserve digits, uppercase and lowercase separately. Raw EMNIST image axes were transposed once to upright; samples were visually inspected before training.
Both architectures used 5 epochs, seed 42, Adam at learning rate 0.001, batch size 256, unweighted cross entropy and no augmentation. The split and minibatch order are shared. See [the model guide](../guides/character-models.md) for architectures and reproduction commands.

## Model comparison

| Model | Parameters | Best epoch | Validation accuracy | Official test correct | Test accuracy | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 112,766 | 5 | 83.19% | 96,973/116,323 | 83.37% | 0.6901 |
| CNN | 109,246 | 5 | 85.51% | 99,563/116,323 | 85.59% | 0.7215 |

**Selected model: CNN**, chosen from validation performance before testing. Test scores did not select the model.

Macro F1 gives every class equal weight; overall accuracy gives every example equal weight. ByClass is imbalanced, so the two measures describe different aspects of performance.

## Character groups

| Model | Digits | Uppercase | Lowercase |
|---|---:|---:|---:|
| MLP | 93.97% (n=57,918) | 77.05% (n=31,346) | 67.99% (n=27,059) |
| CNN | 95.05% (n=57,918) | 81.09% (n=31,346) | 70.58% (n=27,059) |

## Timing

| Model | Training plus validation | Warmed test prediction |
|---|---:|---:|
| MLP | 40.70 s | 0.91 s / 116,323 characters |
| CNN | 576.69 s | 9.82 s / 116,323 characters |

These are single-run local CPU timings. Prediction includes pixel conversion, batched inference, softmax and Python result construction; it excludes dataset loading, segmentation, photo preprocessing and GUI work. Timing is not a portable performance guarantee.

## Known boundary string reconstruction

For each model, 1,000 strings were constructed from 3,500 distinct official-test characters: 250 strings at each length. Both models use identical indices. Boundaries were already known, so these results do not test character segmentation, words or complete photo recognition. They reuse a subset of the character test set and are not independent evidence.

| Model | 2 characters | 3 characters | 4 characters | 5 characters | Overall |
|---|---:|---:|---:|---:|---:|
| MLP | 170/250 (68.0%) | 155/250 (62.0%) | 129/250 (51.6%) | 104/250 (41.6%) | 558/1000 |
| CNN | 178/250 (71.2%) | 162/250 (64.8%) | 143/250 (57.2%) | 114/250 (45.6%) | 597/1000 |

## Retained errors and interface example

MLP frequent confusions: `O` to `0` (2123), `l` to `1` (1796), `I` to `1` (767), `0` to `O` (687), `1` to `I` (443).
CNN frequent confusions: `O` to `0` (1984), `l` to `1` (1730), `0` to `O` (845), `I` to `1` (793), `V` to `v` (435).

Similar handwriting shapes and class imbalance are plausible contributors; confusion counts alone do not establish a causal explanation. Lowercase/uppercase labels remain distinct even where isolated handwriting looks alike.

The fixed interface example has reference text `007Ab12`. Samples were selected by label and dataset order, not by whether the model predicted them correctly.

- MLP predicted `007Ab12`.
- CNN predicted `007Ab12`.

The public predictor was checked against the batched test predictions for those same crops. It never receives the reference text. Leading zeros and case are preserved in the output format, not guaranteed to be recognised correctly.

## Verification and limits

- At the character-model experiment checkpoint, 24 software tests passed: 18 existing tests and six new character tests. The original digit source and model files were unchanged.
- Dataset mapping, class counts, split isolation, checkpoint hashes, source snapshots and per-image predictions are retained.
- One training seed and one configuration per model do not establish optimal settings or run-to-run variability.
- Confidence values are uncalibrated model scores, not correctness guarantees.
- This benchmark does not evaluate photographed strings or segmentation. Those components and GUI integration have separate development evaluations.
- Keep later design decisions on validation/development inputs. Reserve new handwriting for final system evaluation.

## Evidence

- `evidence/emnist/preparation/`: sample grids, orientation comparison, class counts, dataset hashes and split indices.
- `experiments/emnist_byclass/run_01/`: both checkpoints, configuration, history, source snapshot and validation selection.
- `evidence/emnist/evaluation_run_01/`: results, per-image CSVs, per-class metrics, confusion matrices, first-error grids, string results and interface example.
- `evidence/emnist/unit_tests.txt` and `evidence/emnist/training_log.txt`.
- [Character model guide](../guides/character-models.md): references, commands and prediction interface.
