# Handwritten Number and Character Recognition

COS30018 Intelligent Systems, Option B. A desktop application for recognising handwritten numbers, with the approved extension for single-line alphanumeric handwriting.

## Run the application

Use Python 3.12. From this folder in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Alternatively, use ` .\run_gui.ps1 ` after installation. The eight saved checkpoints are included; training datasets are not needed to recognise a supplied photograph. See [installation](docs/guides/installation.md) and the [GUI guide](docs/guides/gui.md).

The ZIP includes [280 handwriting photos](data/README.md). For generation, select `data/segmentation_photos` in Numbers mode or `data/own_characters/development` in Alphanumeric mode.

## Features and model selection

- Load photographs or generate images from labelled handwriting samples.
- Segment characters in reading order, inspect prepared crops and export results.
- Both modes offer MLP, CNN, ResNet-18 and MobileNetV3-Small.
- Selected defaults: CNN for numbers; ResNet-18 for alphanumeric text.
- Preserve letter case and leading zeros.

Inputs should contain one line of separately written dark characters on pale paper. Joined cursive, punctuation and multiline documents are outside the approved scope. Ambiguous letters and mixed strings remain difficult.

## Recorded final evaluation

The frozen system was evaluated on 120 newly written photographs from four group writers. These results are separate from the development collection used to refine the system.

| Collection | Selected system correct | Metric |
|---|---:|---|
| Individual digits | 37/40 | Classification accuracy |
| Multi-digit numbers | 16/20 | Exact number accuracy |
| Individual letters/digits | 29/40 | Classification accuracy |
| Mixed strings | 4/20 | Exact case-sensitive string accuracy |

See [machine-readable results](results/README.md) and [full research reports](docs/README.md). Later augmentation candidates did not pass their adoption criteria; the selected application checkpoints were retained. The final set has since been analysed and must not be treated as untouched evidence for further changes.

## Project structure

| Path | Purpose |
|---|---|
| `app.py`, `gui_core.py` | Desktop interface and application services |
| `preprocess_digits.py`, `segment_digits.py` | Digit preparation and segmentation |
| `hnrs_ml/`, `hnrs_chars/` | Models and recognition pipelines |
| `research/` | Experiment, evaluation and reporting commands |
| `scripts/` | PowerShell launchers and project checks |
| `tests/` | Automated tests and desktop workflow checks |
| `experiments/` | Eight application checkpoints and their configurations |
| `results/` | Selected metrics, verification and dataset split records |
| `docs/` | Guides, reports and selected illustrations |
| `data/` | 280 collected handwriting photos and their checksum manifest; MNIST/EMNIST excluded |
| `evidence/` | Detailed research outputs, excluded from Git |

The complete original experiment tree, crops, source snapshots and photos are preserved in the [research archive](docs/guides/research-archive.md). They are not deleted from the working computer. Historical Git commits are retained.

## Verification and research

```powershell
.\.venv\Scripts\python.exe scripts/check_project.py
.\.venv\Scripts\python.exe -m research.train_models_week11 --help
```

The project check runs the unit suite and verifies all eight checkpoint paths. The real GUI smoke test (`.\run_gui.ps1 -Verify`) uses the included development photos and additionally needs an interactive Windows desktop. See [research commands](research/README.md) before reproducing experiments.

## Submission status

The software and experiment evidence do not replace the final team report, presentation video or individual records. Cross-computer acceptance evidence must be supplied by the member who performed those checks. Recognition limitations are documented rather than presented as resolved.
