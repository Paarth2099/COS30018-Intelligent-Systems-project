# Manula's extension: string segmentation

## What this section does

Split a single-line photograph such as `Room7` into complete characters, keep dots with their stems, and return crops in left-to-right order. This is geometric connected-component analysis, not a trained segmentation model. The new code reuses the original project's component labelling utilities, Paarth's character preparation and Haresh's frozen EMNIST CNN.

Implemented and compared vertical projection and grouped connected components on all 40 development photos. Grouped components is the selected default: 38/40 visually acceptable separations versus 32/40 for projection. The complete CNN pipeline correctly reads only 7/40 strings for either method. Selected-method character error rate is 35.625%. These are development results after inspecting and refining the methods on these same images, not final generalisation results.

The two remaining grouped-method segmentation failures are Haresh's `Hello` (broken H split) and `Bill` (i dot separated from stem). Other recognition errors occur even with acceptable crops. Do not report 95% recognition accuracy.

## Run and inspect

Open this `project-github` folder in VS Code. In its PowerShell terminal:

```powershell
.\run_segmentation.ps1 -Action Demo
```

This runs Manula's `Ab12` photo and prints a newly created evidence folder. Open its `preview.png`: the top shows character boxes; the bottom shows the 28x28 prepared characters in reading order. `demo.json` and `prediction.json` contain actual model output. The demo does not read the expected text from its filename.

Run a different photo, including a known failure:

```powershell
.\run_segmentation.ps1 -Action Demo -Image 'data/own_strings/development/Haresh/Haresh_03_text-Hello.jpg'
```

Run tests or repeat both methods on the full development collection:

```powershell
.\run_segmentation.ps1 -Action Test
.\run_segmentation.ps1 -Action Compare
```

Each demo/comparison creates a new timestamped output directory. The frozen reported experiment is `evidence/character_segmentation/run_04`; reruns do not overwrite it. The report builder is specifically for the four saved development runs, not an automatic selection tool for future datasets.

The runner uses this checkout's `.venv` when present and otherwise the existing downloaded baseline environment on Paarth's laptop. On another machine, create `.venv` with Python 3.12 and install `requirements.txt`. There are no new package requirements. Raw photos under `data/` are Git-ignored: distribute the matching `own_strings/development` directory through the team's OneDrive. The manifest contains input hashes to verify the same files were used.

## Where to explain the code

| File | Purpose |
|---|---|
| `hnrs_chars/segmentation.py` | Foreground mask, small-gap repair, two segmentation methods, dot grouping, ordered crops and public pipeline |
| `hnrs_chars/string_experiment.py` | Label validation, both-method experiment, metrics, evidence and review sheets |
| `hnrs_chars/string_predict.py` | Photo-to-string demo without reference labels |
| `tests/test_character_segmentation.py` | 14 new behavioural and regression tests |
| `hnrs_chars/string_report.py` | Measured report and recorded selection from reviewed evidence |

The preparation function in `hnrs_chars/preprocessing.py` and the classifier in `hnrs_chars/predict.py` are reused unchanged; they are Paarth's and Haresh's earlier components.

## Interface for Saniru

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
- Surface warnings in the interface. Keep processing off the GUI event thread and preserve the existing digit mode.

Saniru's next task is GUI integration for the approved character extension, with mode selection, string results and crop previews. This task has provided the callable pipeline and CLI; it has not added that GUI mode.

## Evidence and limitations

See [the measured report](MANULA_SEGMENTATION_RESULTS.md). All four runs retain input/checkpoint/source hashes, CSV results, masks, component groups, crops, arrays and predictions. `run_04/visual_review.csv` records assistant inspection of all 80 method/image combinations. It is not independent human annotation or bounding-box IoU evaluation. The initial input sheets are in `input_review/` and final review sheets are in `run_04/`.

Paarth's seventh source filename was corrected from `00Ab` to the visibly written `007Ab` before evaluation. Image bytes were preserved. Ambiguous case labels were otherwise retained as provided.

The algorithm expects a single line of separated alphanumeric handwriting. Touching/cursive writing, multiline pages, punctuation, inferred word spaces, severe shadows and very small detached parts remain outside its reliable scope. The frozen CNN also confuses similar shapes and case. The team should investigate those errors and evaluate on fresh, untouched photos after finalising development choices.

For a tutor demonstration, show one successful separation, projection merging `Room7`, grouped components separating it, and a retained failure. Explain count agreement, visual separation, exact-string accuracy and character error rate separately. Review and understand the code before claiming individual learning or contributions. Implementation, execution and visual review here used Codex assistance; no personal hours have been assigned.
