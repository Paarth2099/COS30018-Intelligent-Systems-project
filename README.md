# Handwritten Number Recognition System

COS30018 Intelligent Systems — Option B. A Python desktop application for handwritten numbers, with an approved extension for single-line alphanumeric handwriting.

The system combines image preparation, segmentation and MLP/CNN classification. Both modes support loading photographs and generating images from labelled handwriting samples, with ordered crop previews and exportable results.

## Quick start

Use Python 3.12. Run from the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run_gui.ps1
```

With an existing environment, only the last command is needed. See [installation and datasets](docs/guides/installation.md).

## Modes

| Mode | Classes | Models | Segmentation |
|---|---|---|---|
| Numbers | 0–9 | MNIST MLP and CNN | Connected components or projection |
| Alphanumeric | 0–9, A–Z, a–z | EMNIST ByClass MLP and CNN | Grouped components or projection |

Outputs retain case and leading zeros. Supported photographs contain one line of separate dark characters on pale paper. Joined cursive and multiline documents are outside the supported scope.

## Documentation

- [Documentation index](docs/README.md)
- [Installation](docs/guides/installation.md) and [GUI user guide](docs/guides/gui.md)
- [System architecture](docs/architecture.md)
- [Digit-system results](docs/reports/digit-system-results.md)
- [Character-model results](docs/reports/emnist-results.md)
- [Character-preprocessing results](docs/reports/character-preprocessing-results.md)
- [String-segmentation results](docs/reports/string-segmentation-results.md)
- [GUI verification](docs/reports/gui-verification.md)

## Results and scope

| Evaluation | Result | Scope |
|---|---|---|
| MNIST test set | MLP 96.60%; CNN 98.53% | 10,000 isolated test digits |
| Original number photos, selected pipeline | 15/20 single digits; 18/20 exact multi-digit numbers | Development photos |
| EMNIST ByClass test set | MLP 83.37%; CNN 85.59% | 116,323 isolated test characters |
| Character photos, selected preparation + CNN | 53/80 correct | Development photos |
| String photos, grouped components | 38/40 acceptable separations; 7/40 exact strings | Development photos; qualitative crop review |

Classification, segmentation quality and complete-string recognition measure different outcomes. Development photos were inspected during refinement and are not an independent final test. Recognition error analysis and fresh-photo evaluation remain outstanding.

## Repository layout

```text
docs/          Guides, architecture, experiment reports and data notes
hnrs_ml/       Digit model training, evaluation and prediction
hnrs_chars/    Character models, preprocessing, segmentation and experiments
tests/         Automated tests and desktop workflow checks
experiments/   Checkpoints, training configurations and source snapshots
evidence/      Metrics, crops, previews and verification records
data/          Local datasets and original photos (excluded from Git)
app.py         Desktop interface
gui_core.py    Acquisition, recognition dispatch and export
run_*.ps1      Windows launchers and experiment commands
```

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\run_gui.ps1 -Verify
```

The software suite contains 52 tests. Desktop checks require the models and photo folders described in the installation guide and briefly open the application. Each run creates a new evidence folder. Historical reports identify the test count at their own development stage.

Raw datasets and photos are shared separately. Classifying a supplied photo does not require retraining the saved models.
