# Character models

The alphanumeric pipeline uses separate 62-class EMNIST MLP and CNN checkpoints. Both models are available in the desktop application. The original 10-class digit models remain separate.

See [installation](installation.md), [measured results](../reports/emnist-results.md) and [the GUI guide](gui.md). Commands below run from the repository root.

## What was implemented

| File | Purpose |
|---|---|
| `hnrs_chars/data.py` | Official download and checksum, 62-class mapping, upright orientation, stratified split and memory-efficient batch conversion |
| `hnrs_chars/prepare.py` | Labelled sample grid, raw/upright comparison, class counts, file hashes and fixed split indices |
| `hnrs_chars/models.py` | Independent 62-class MLP and CNN architectures |
| `hnrs_chars/train.py` | Same split and batch order for both models; saved best checkpoints selected by validation |
| `hnrs_chars/evaluate.py` | Official test metrics, confusions, errors, timing, known-boundary string evaluation and interface example |
| `hnrs_chars/predict.py` | Load a character checkpoint and return ordered, case-sensitive predictions |
| `tests/test_characters.py` | Split isolation, checkpoint compatibility, case, order, leading zeros, pixel validation and metric checks |

## Dataset checks

EMNIST **ByClass** contains 697,932 original training images and 116,323 official
test images. It has 62 imbalanced classes, mapped as `0-9`, `A-Z`, then `a-z`.
The supplied mapping file is checked against the code and torchvision mapping.
This is not EMNIST Balanced (47 classes) or Letters (26 classes).

The official archive is checked against torchvision's published MD5. The loader
extracts only ByClass files. SHA-256 hashes identify the extracted dataset files.

The binary images need one transpose of their row/column axes to appear upright.
`orientation_check.png` compares uncorrected and corrected training examples;
`all_classes_upright.png` shows two training examples for every class. Both were
visually inspected before training. The asymmetric examples include `2`, `F`,
`L`, `R`, `b` and `j`. Visual inspection does not certify every dataset label.

The transform is **only for raw EMNIST**. Do not transpose ordinary upright
photo crops passed into `CharacterPredictor`.

Using seed 42, approximately 10% of each training class was reserved for
validation: **628,142 training and 69,790 validation images**. The sets are
disjoint, together cover the original training partition and retain all classes.
The official test images are not loaded by preparation or training.

## Training design

Both models receive upright 28x28 grayscale character images, light strokes on
black, scaled by dividing by 255. No thresholding or augmentation is applied to
the EMNIST training images. Photographic preprocessing requires a separate
development comparison; it must not be assumed equivalent to this benchmark.

- MLP: flatten, 128-unit ReLU layer, 64-unit ReLU layer, 62 output logits.
- CNN: 16-channel 3x3 convolution, ReLU, 2x2 pooling; 32-channel convolution,
  ReLU and pooling; 64-unit ReLU layer and 62 output logits.
- Five epochs, Adam, learning rate 0.001, batch size 256, unweighted cross entropy.
- CPU, four threads, deterministic operations, seed 42 and identical minibatch
  order for both models. The full training split is used in every epoch.
- Validation accuracy selects each checkpoint; validation loss breaks ties.
  The same validation rule selects the model for integration before testing.

This is a controlled baseline, not an exhaustive search or proof of optimal
hyperparameters. There is only one training seed. Class imbalance and similar
upper/lowercase forms are important limitations. Macro-F1 and per-class results
are reported alongside overall accuracy so common classes do not hide all rare
class failures.

## Rerunning experiments

Existing evidence folders are never overwritten. These reproduction commands use new output directories; choose another run name if they already exist.

```powershell
.\.venv\Scripts\python.exe -m hnrs_chars.prepare --download --output evidence/emnist/preparation_repeat
.\.venv\Scripts\python.exe -m hnrs_chars.train --preparation evidence/emnist/preparation_repeat --output experiments/emnist_byclass/repeat
.\.venv\Scripts\python.exe -m hnrs_chars.evaluate --experiment experiments/emnist_byclass/repeat --output evidence/emnist/evaluation_repeat
```

The original recorded run is in `evidence/emnist/preparation/`,
`experiments/emnist_byclass/run_01/` and `evidence/emnist/evaluation_run_01/`. Results and the selected model are recorded
in JSON rather than inferred from filenames. Checkpoint hashes are verified
before evaluation.

Once the test results have been inspected, do not tune repeatedly against that
test set and present the improved score as untouched final-test performance.
Use validation/development inputs for decisions and reserve fresh handwriting
for the later complete-system evaluation.

## Prediction interface

```python
from hnrs_chars.predict import CharacterPredictor

# Choose the checkpoint named by run_01/selection.json.
predictor = CharacterPredictor("experiments/emnist_byclass/run_01/cnn.pt")
result = predictor.predict_text(prepared_crops)
print(result["text"])
```

`prepared_crops` must be float arrays shaped `[N,28,28]` or `[N,1,28,28]`, range
0..1, upright light handwriting on black, already ordered left to right. An empty
batch returns an empty string. Integer pixels, wrong shapes, nonfinite values
and out-of-range values are rejected. A digit-only checkpoint is rejected too.

The result contains `text`, `characters`, `model` and `task`. Each character entry
has `class_id`, `character` and an uncalibrated `confidence` score. Case and leading
zeros are retained. The predictor does not crop, segment, infer word spaces,
correct spelling or substitute a requested reference string as its answer.

The GUI renders the `characters` result schema in Alphanumeric mode and the
`digits` schema in Numbers mode. Character segmentation groups detached parts
before preparation; the classifier receives complete crops in reading order.

The constructed-string benchmark uses already separated test characters. It
measures classification and ordering with known boundaries, rather than
segmentation or end-to-end photo recognition. These samples reuse the official
character test set and are not independent evaluation data.

## Sources

- NIST, EMNIST dataset: https://www.nist.gov/itl/products-and-services/emnist-dataset
- Cohen et al., EMNIST paper: https://arxiv.org/abs/1702.05373
- Torchvision EMNIST loader and archive checksum: https://github.com/pytorch/vision/blob/v0.21.0/torchvision/datasets/mnist.py
- TensorFlow Datasets EMNIST catalogue (counts and transposed-image explanation): https://www.tensorflow.org/datasets/catalog/emnist

See [the measured results](../reports/emnist-results.md) for the completed benchmark.
