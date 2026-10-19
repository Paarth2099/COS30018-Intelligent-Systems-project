# Alphanumeric string segmentation

## Overview

Split a single-line photograph such as `Room7` into complete characters, keep dots with their stems, and return crops in left-to-right order. This is geometric connected-component analysis, not a trained segmentation model. The pipeline reuses the project's connected-region utilities, character preparation and frozen EMNIST CNN.

Implemented and compared vertical projection and grouped connected components on all 40 development photos. Grouped components is the selected default. The Week 10 version adds bounded broken-body merging and short-stem eligibility, reaching 40/40 visually acceptable separations and 8/40 exact CNN strings, with 33.125% character error rate. See the [Week 10 investigation](../reports/segmentation-experiment-week10.md). These are reused development results, not final generalisation results.

The earlier baseline had 38/40 visually acceptable separations and 7/40 exact CNN strings. Week 10 repairs Haresh's broken H in `Hello` and detached i in `Bill`. `Hello` is still predicted as `Hell0`: acceptable segmentation is not perfect recognition. The projection comparator remains unchanged.

## Run and inspect

Open this `project-github` folder in VS Code. In its PowerShell terminal:

```powershell
.\run_segmentation.ps1 -Action Demo
```

This runs Manula's `Ab12` photo and prints a newly created evidence folder. Open its `preview.png`: the top shows character boxes; the bottom shows the 28x28 prepared characters in reading order. `demo.json` and `prediction.json` contain actual model output. The demo does not read the expected text from its filename.

Run a different photo, including the repaired H example:

```powershell
.\run_segmentation.ps1 -Action Demo -Image 'data/own_strings/development/Haresh/Haresh_03_text-Hello.jpg'
```

Run tests or repeat both methods on the full development collection:

```powershell
.\run_segmentation.ps1 -Action Test
.\run_segmentation.ps1 -Action Compare
```

Each demo/comparison creates a new timestamped output directory. The frozen reported experiment is `evidence/character_segmentation/run_04`; reruns do not overwrite it. The report builder is specifically for the four saved development runs, not an automatic selection tool for future datasets.

Create a local Python 3.12 environment using [the installation guide](installation.md). There are no new package requirements. Raw photos under `data/` are Git-ignored: distribute the matching `own_strings/development` directory through the team's OneDrive. The manifest contains input hashes to verify the same files were used.

## Source modules

| File | Purpose |
|---|---|
| `hnrs_chars/segmentation.py` | Foreground mask, small-gap repair, two segmentation methods, dot grouping, ordered crops and public pipeline |
| `hnrs_chars/string_experiment.py` | Label validation, both-method experiment, metrics, evidence and review sheets |
| `hnrs_chars/string_predict.py` | Photo-to-string demo without reference labels |
| `tests/test_character_segmentation.py` | 14 new behavioural and regression tests |
| `hnrs_chars/string_report.py` | Measured report and recorded selection from reviewed evidence |

The pipeline uses `hnrs_chars/preprocessing.py` for individual crops and `hnrs_chars/predict.py` for classification.

## Recognition interface

Load the predictor once, then call the pipeline for each photo:

```python
from pathlib import Path
from PIL import Image
from hnrs_chars.string_experiment import load_predictor
from hnrs_chars.segmentation import recognise_text

predictor, checkpoint = load_predictor(Path('experiments/emnist_byclass/run_01'))
with Image.open(photo_path) as photo:
    result = recognise_text(photo, predictor)
print(result['prediction']['text'])
```

- `result['segmentation']`: dataclass with `gray`, `mask`, `boxes`, `crops`, `warnings`, `groups`, `removed_components` and `method`.
- Box coordinates refer to the EXIF-corrected, resized `gray` image, not necessarily the original photo. Use that image when drawing boxes.
- `result['prepared']`: float32 array `[N, 28, 28]`, light ink on black; blank input produces an empty batch.
- `result['prediction']`: existing character-predictor dictionary with `text`, `characters`, `model` and `task`. Retain strings, case and leading zeros. Do not convert the text to a number or use the digit-only `digits` field.
- The interface displays warnings and processes recognition in a worker thread. Numbers mode remains available alongside Alphanumeric mode.

The desktop GUI uses this interface in Alphanumeric mode, including mode selection, text results and ordered crop previews. See [the GUI guide](gui.md).

## Evidence and limitations

See [the measured report](../reports/string-segmentation-results.md). All four runs retain input/checkpoint/source hashes, CSV results, masks, component groups, crops, arrays and predictions. `evidence/character_segmentation/run_04/visual_review.csv` records qualitative preview inspection of all 80 method/image combinations; it is not independently annotated ground truth or a bounding-box IoU evaluation. The [initial input sheets](../../evidence/character_segmentation/input_review/) and [final review sheets](../../evidence/character_segmentation/run_04/) are retained with the experiment evidence.

Paarth's seventh source filename was corrected from `00Ab` to the visibly written `007Ab` before evaluation. Image bytes were preserved. Ambiguous case labels were otherwise retained as provided.

The algorithm expects a single line of separated alphanumeric handwriting. Touching/cursive writing, multiline pages, punctuation, inferred word spaces, severe shadows and very small detached parts remain outside its reliable scope. The frozen CNN also confuses similar shapes and case. Further evaluation on untouched photos is required to measure generalisation.
