# System architecture

The system has separate digit and alphanumeric pipelines behind a shared desktop interface.

```text
Loaded photo OR generated handwriting image
  -> upright grayscale working image
  -> foreground detection and segmentation
  -> individual crops in reading order
  -> 28x28 preparation
  -> selected MLP / baseline CNN / ResNet-18 / MobileNetV3-Small
  -> optional line-context letter-case adjustment (alphanumeric photos)
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

Mode switches select the evaluated defaults: baseline CNN for Numbers and ResNet-18 for Alphanumeric. All four architectures remain selectable. `resolve_checkpoint()` chooses the task-specific experiment and validates its saved hash before loading. The eight mode/model combinations use separate caches; missing or corrupt checkpoints raise errors rather than substituting another model. GPU-trained checkpoints are portable to the GUI's CPU inference path.

Generation labels select handwriting tiles but are not supplied to recognition. Exports keep requested text separate from predictions and record settings, source/checkpoint hashes, masks, crops, boxes and warnings.

## Evaluation boundaries

Validation data selects checkpoints before official test evaluation. Inspected real-photo collections measure development performance. Constructed strings with known boundaries test classification and ordering, not segmentation. Fresh end-to-end evaluation remains separate.

## Recorded evaluation and repository organisation

The frozen 120-photo evaluation is complete; see [selected results](../results/README.md). The final report and presentation remain separate deliverables. Research commands live in `research/`, wrappers in `scripts/`, and detailed immutable evidence is in the [archive](guides/research-archive.md).

## Model lab and additional studies

`gui_lab.py` owns the training, comparison and curves windows; `gui_charts.py` draws their charts using Tk. `gui_training.py` validates settings, loads the fixed training/validation split, runs training outside the UI thread and saves the best completed epoch. Training does not use the official test partition. Local runs go under `experiments/gui_runs/`. The recognition service registers these checkpoints separately and verifies their hashes before use; shipped checkpoints are preserved.

`hnrs_chars/case_rule.py` uses the original segmentation-box geometry to adjust case after classification. It retains the classifier output as `model_text` and records changes in the exported prediction. The GUI enables this by default for loaded alphanumeric photos and disables it for generated tiles. The lower-level API defaults to `line_case=False`, preserving the frozen evaluation route.

The five classical baselines are research comparisons, not extra GUI recognition choices. Their implementation is `hnrs_ml/classical.py`, their driver is `research/compare_classical_models.py`, and saved summaries are in `results/classical_baselines/`. Scikit-learn is optional and needed only for that study and its tests.

The original final evaluation remains 4/20 exact mixed strings. The later line-case rule reached 20/40 in writer-held-out development evaluation and 6/20 on the already-inspected final strings. The latter is post-hoc evidence, not a new independent test. See the [case-rule report](reports/case-rule-study.md).
