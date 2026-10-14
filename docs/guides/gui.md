# Desktop application user guide

## What was implemented

The desktop application now has Numbers and Alphanumeric modes, each supporting photo loading, generation from labelled handwriting images, CNN/MLP selection, crop previews and evidence export. The interface retains the existing visual design and digit workflow. Recognition and image acquisition run in a worker thread; Tk widgets update on the main thread.

The application integrates character segmentation, preprocessing and the saved EMNIST models through `app.py` and `gui_core.py`.

## Start the application

Open the `project-github` folder in VS Code. In PowerShell:

```powershell
.\run_gui.ps1
```

The launcher uses the repository's local virtual environment when available. Environment setup is described below. If PowerShell blocks scripts, enable them for this terminal session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\run_gui.ps1
```

On another Windows computer, install Python 3.12 with Tcl/Tk and create the local environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run_gui.ps1
```

No extra Python packages were added. Obtain the saved `experiments/baseline` and `experiments/emnist_byclass/run_01` model folders with the project. Raw photos are excluded from Git; copy the following shared OneDrive folders into the corresponding local directories:

- `data/segmentation_photos`: original labelled digit/number photos.
- `data/own_characters/development`: labelled single-character extension photos.
- `data/own_strings/development`: handwritten string development photos.

## Recognise a photo

1. Select **Alphanumeric** under Recognition mode.
2. Leave **cnn**, **grouped_components** and **otsu** selected initially.
3. Choose **Open photo → Choose image** and select a single-line handwritten string.
4. Select **Recognise text**. Inspect the text, character boxes and ordered 28x28 crops.
5. Select **Export result** to save a new evidence directory.

Otsu preparation is fixed in Alphanumeric mode because that is the selected character preparation pipeline. Projection and MLP remain available for comparisons. Changing the mode, model or separation setting clears the previous prediction and disables export until recognition runs again. Switching mode retains the displayed input photo but resets generation defaults and separation settings for that mode.

Use dark, separately written characters on pale paper. Joined cursive, multiline documents, punctuation and inferred word spaces are not supported. Blank input gives an empty result and a warning. Invalid files and missing/corrupt model files produce a visible error. The trained character checkpoint hash is checked before recognition.

## Generate a mixed string

1. Select **Alphanumeric**, then **Generate**.
2. Choose a folder containing individually labelled character photos; nested writer folders are supported.
3. Enter 1–12 ASCII letters/digits, for example `Ab12` or `00Ai`.
4. Select **Generate image**, then **Recognise text**.

Accepted labels include `Name_01_upper-A.jpg`, `Name_02_lower-a.jpg`, `Name_03_digit-0.jpg`, or a single-character filename such as `A.png`. Upper/lowercase labels must match their stated case. Whole-string photos are excluded from the character catalogue. The generator reports missing characters instead of replacing them; the current photo collection does not cover every possible letter/digit. Choose a larger source folder or collect a missing character if necessary.

Generation uses a fixed seed of 42 for repeatable source selection. It records chosen files and their SHA-256 hashes. The **Requested** text is a generation instruction, not a prediction. Recognition sees only the generated pixels. Case and leading zeros stay as text throughout display and export. Generated samples reuse prepared character photos and must not be treated as independent handwritten-string evaluation.

Switch to **Numbers** for the existing digit workflow. Digit generation still uses filenames such as `Name_01_label-7.jpg` or `7.png`.

## Export contents

Each export creates a new directory, preserving earlier evidence:

- Input image and SHA-256 hash.
- Annotated image (`detected_characters.png` or `detected_digits.png`).
- Foreground mask, original character crops, prepared crop images and NumPy array.
- `result.json`: actual predicted text and per-character scores, source provenance, requested text where applicable, active mode/model/separation/preparation, checkpoint path/hash, boxes and warnings.

Boxes refer to the resized upright grayscale working image, not necessarily the original photo dimensions. Per-character model scores are uncalibrated and must not be presented as verified probabilities of correctness.

## Source modules

| Code | Responsibility |
|---|---|
| `app.py`: `mode_changed` | Mode controls, labels and defaults; clears stale results |
| `app.py`: `start_job`, `poll` | Worker processing and safe main-thread display |
| `app.py`: `draw_digits` | Shared ordered-crop rendering for digit and character prediction schemas |
| `gui_core.py`: `character_catalog`, `generate_text` | Case-sensitive source selection and reproducible mixed-string composition |
| `gui_core.py`: `RecognitionService` | Separate mode/model caches, checkpoint validation and dispatch to existing pipelines |
| `gui_core.py`: `export_result` | Source/settings/result evidence without replacing earlier exports |

## Verification and limitations

Software tests cover acquisition, generation, provenance, case/zeros, malformed/missing inputs, corrupt checkpoints and the existing recognition algorithms. The separate desktop smoke script exercises real Tk callbacks and saved models while replacing file-dialog selection with known local paths; it saves screenshots and a verification record under `evidence/gui_extension/`.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\run_gui.ps1 -Verify
```

The first command assumes a local `.venv`; the environment selected by `run_gui.ps1` can also be used. The smoke test requires the development photo folders listed above and briefly opens the GUI.

Read [the GUI verification report](../reports/gui-verification.md) for recorded results. Automated checks cover one local environment; cross-machine acceptance testing remains outstanding.

GUI integration does not improve recognition accuracy. The earlier 40-photo development experiment reported 38/40 acceptable grouped segmentations but only 7/40 exactly correct strings. Investigate classifier/preparation errors and test fresh photos before final performance claims. Preserve those earlier results instead of replacing them with a successful demo.
