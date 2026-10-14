# System architecture

The system has separate digit and alphanumeric pipelines behind a shared desktop interface.

```text
Loaded photo OR generated handwriting image
  -> upright grayscale working image
  -> foreground detection and segmentation
  -> individual crops in reading order
  -> 28x28 preparation
  -> selected MLP/CNN
  -> prediction string, crop previews and evidence export
```

## Modules

| Module | Responsibility |
|---|---|
| `app.py` | Tkinter controls, worker jobs and result rendering |
| `gui_core.py` | Acquisition, generation, mode-specific model cache and export |
| `preprocess_digits.py` | Digit crop and grayscale/fixed/Otsu representations |
| `segment_digits.py` | Connected components, projection and number-recognition API |
| `hnrs_ml/` | MNIST models, training, evaluation and prediction |
| `hnrs_chars/preprocessing.py` | Complete-character preparation retaining detached parts |
| `hnrs_chars/segmentation.py` | Character grouping and string-recognition API |
| `hnrs_chars/data.py`, `train.py`, `evaluate.py`, `predict.py` | EMNIST mapping, models and evaluation |

## Data contracts

Digit classifiers have ten outputs; character classifiers have 62. Prepared inputs are upright floating-point arrays in [0,1], shaped [N,28,28] or [N,1,28,28], with light ink on black. Only raw EMNIST receives the dataset-specific transpose.

`recognise_number()` and `recognise_text()` return segmentation details, prepared arrays and predictions. The GUI renders `digits` entries in Numbers mode and `characters` entries in Alphanumeric mode. Final text retains case and leading zeros.

The character segmenter groups detached dots and fragmented strokes before sorting crops left to right. Boxes refer to the returned upright, resized grayscale image. Preparation corrects local lighting, filters small specks, preserves aspect ratio, pads to 28x28 and normalises ink brightness.

## Interface and provenance

Loading, generation and recognition run in a worker thread; a queue delivers results to the main Tk thread. Controls are disabled during a job. Recognition-setting changes clear stale results before export.

Generation labels select handwriting tiles but are not supplied to recognition. Exports keep requested text separate from predictions and record settings, source/checkpoint hashes, masks, crops, boxes and warnings.

## Evaluation boundaries

Validation data selects checkpoints before official test evaluation. Inspected real-photo collections measure development performance. Constructed strings with known boundaries test classification and ordering, not segmentation. Fresh end-to-end evaluation remains separate.
