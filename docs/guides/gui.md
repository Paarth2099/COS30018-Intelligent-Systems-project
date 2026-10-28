# Desktop application user guide

## What was implemented

The desktop application has Numbers and Alphanumeric modes, each supporting photo loading, generation from labelled handwriting images, four-model selection, crop previews and evidence export. Available models are `cnn` (baseline CNN), `mlp`, `resnet18` and `mobilenet_v3_small`. The interface retains the existing visual design and digit workflow. Recognition and image acquisition run in a worker thread; Tk widgets update on the main thread.

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

No extra Python packages were added. Obtain the saved `experiments/baseline`, `experiments/emnist_byclass/run_01` and `experiments/week11_gpu_run01` model folders with the project, including their `selection.json` files. The GUI performs inference on CPU, including models trained on a GPU; a GPU is not required to run it. The collected photographs are included in the downloaded ZIP at the following locations:

- `data/segmentation_photos`: original labelled digit/number photos.
- `data/own_characters/development`: labelled single-character extension photos.
- `data/own_strings/development`: handwritten string development photos.

## Recognise a photo

### Window size and scrolling

The window starts within the available screen area. At smaller sizes or higher display scaling, use the right-hand scrollbar or mouse wheel to reach recognition controls and the crop breakdown. Use the bottom scrollbar or **Shift + mouse wheel** to move sideways. The status line stays visible below the scrolling workspace.

Keyboard focus automatically brings active controls into view as you tab between them. The crop breakdown also has its own horizontal scrollbar for long results. On this Windows computer, the layout was checked at normal startup size, 800 × 500 and 640 × 400 Tk units with 150% effective display scaling. Other display configurations still need acceptance testing.

### Recognition steps

1. Select **Alphanumeric** under Recognition mode.
2. Leave **resnet18**, **grouped_components** and **otsu** selected initially.
3. Choose **Open photo → Choose image** and select a single-line handwritten string.
4. Select **Recognise text**. Inspect the text, character boxes and ordered 28x28 crops.
5. Select **Export result** to save a new evidence directory.

Otsu preparation is fixed in Alphanumeric mode because that is the selected character preparation pipeline. Projection and all four models remain available for comparisons. Changing the mode, model or separation setting clears the previous prediction and disables export until recognition runs again. Switching mode retains the displayed input photo but resets model, generation and separation defaults: **Numbers uses baseline CNN; Alphanumeric uses ResNet-18**. These defaults follow the Week 11 model comparison; a benchmark improvement alone did not justify replacing the number model.

### Letter case from line context

In Alphanumeric mode with a loaded photo, **Letter case from line context** is ticked by default. After classification it compares each case-ambiguous letter (c, m, o, s, u, v, w, x, z, and 0/O/o) with the height of the line's unambiguous characters. Short, low letters become lowercase; tall ones become capitals. For example, ResNet-18 reads `R00M7` and the rule outputs `Room7`. The result line states how many characters changed and what the classifier read. Changed crops are labelled `final←classifier` (for example `o←0`). The rule needs at least one reference character (a digit 1-9, an unambiguous capital, or b/d/h/k/l/t). It is disabled for generated images, whose tiles all have the same height. See [the case-rule study](../reports/case-rule-study.md).

Use dark, separately written characters on pale paper. Joined cursive, multiline documents, punctuation and inferred word spaces are not supported. Blank input gives an empty result and a warning. Invalid files and missing/corrupt model files produce a visible error. Every selected checkpoint hash is checked before recognition; a missing new model does not silently fall back to CNN.

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

## Model lab: train, compare and inspect models

Select **Model lab: train & compare...** at the bottom of the left panel. A second window opens with three tabs.

**Train a model**
1. Choose the dataset (MNIST digits or EMNIST characters), architecture (`mlp`, `cnn`, `resnet18`, `mobilenet_v3_small`), optimizer (Adam or SGD with momentum 0.9), learning rate, batch size, epochs, number of training and validation samples, and random seed.
2. Select **Start training**. Accuracy and loss curves update live: per-batch loss, and train and validation values after each epoch. **Stop** ends training after the current batch and keeps the best completed epoch.
3. When training finishes, the validation confusion matrix and the most frequent mistakes are shown. The run is saved under `experiments/gui_runs/<time>_<task>_<model>/` (`run.json`, `history.csv`, checkpoint).
4. Select **Use in recogniser**. The model appears in the main window's model list as `lab-<model>-<time>`, and is used like the shipped models (its hash is checked before every use).

Training uses the project's fixed train/validation split and never loads the official test set. The datasets are not shipped; tick **Download dataset if missing** to fetch MNIST (about 60 MB) or EMNIST (about 560 MB). An MLP or CNN on 10,000 MNIST images trains in seconds to about a minute on a laptop CPU. ResNet-18 and full EMNIST take much longer.

**Compare models** shows the recorded official-test accuracy, macro F1, size and speed of all four networks and, for MNIST, the five [classical baselines](../reports/classical-baselines.md). It also shows the 120-photo final evaluation per model. The application default is highlighted. Nothing is re-run.

Screenshots: [training](../assets/model_lab_training.png), [comparison](../assets/model_lab_compare.png), [curves and confusion](../assets/model_lab_curves.png), [line-case result](../assets/line_case_room7.png).

**Curves and errors** shows, for any shipped checkpoint, its recorded training and validation curves and its official-test confusion matrix (10x10 with counts, or the 62x62 heatmap for EMNIST), with the most frequent mistakes listed.

## Source modules

| Code | Responsibility |
|---|---|
| `app.py`: `mode_changed` | Mode controls, labels and defaults; clears stale results |
| `app.py`: `start_job`, `poll` | Worker processing and safe main-thread display |
| `app.py`: `draw_digits` | Shared ordered-crop rendering for digit and character prediction schemas |
| `gui_core.py`: `character_catalog`, `generate_text` | Case-sensitive source selection and reproducible mixed-string composition |
| `gui_core.py`: `RecognitionService` | Separate mode/model caches, checkpoint validation and dispatch to existing pipelines |
| `gui_core.py`: `export_result` | Source/settings/result evidence without replacing earlier exports |
| `gui_core.py`: `RecognitionService.register_custom` | Makes Model-lab checkpoints selectable, with hash checks |
| `gui_lab.py`: `ModelLab` | Model-lab window: training form, live charts, comparison and curves tabs |
| `gui_training.py`: `train` | Tk-independent training loop, best-epoch checkpoint, validation confusion matrix |
| `gui_charts.py` | Line, bar and confusion-heatmap charts drawn on Tk canvases (no plotting dependency) |
| `hnrs_chars/case_rule.py` | Letter case from line geometry |

## Verification and limitations

Software tests cover acquisition, generation, provenance, case/zeros, malformed/missing inputs, corrupt checkpoints and the existing recognition algorithms. The separate desktop smoke script exercises real Tk callbacks and saved models while replacing file-dialog selection with known local paths; current runs save screenshots and a verification record under `evidence/gui_week11/`.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\run_gui.ps1 -Verify
.\.venv\Scripts\python.exe tests/gui_lab_smoke.py
```

`tests/gui_lab_smoke.py` opens the Model lab, checks every tab, trains an MLP on 3,000 MNIST images, uses it in the main window and checks the line-case result on `Room7`. It needs `data/MNIST/raw` and saves screenshots under `evidence/gui/lab_smoke_*`.

The first command assumes a local `.venv`; the environment selected by `run_gui.ps1` can also be used. The smoke test requires the development photo folders listed above and briefly opens the GUI.

Read [the Week 11 integration report](../reports/gui-integration-week11.md) for current verification, and [the earlier GUI verification report](../reports/gui-verification.md) for historical results. Automated checks cover one local environment; updated cross-machine acceptance testing remains outstanding.

GUI integration reproduces the selected pipeline; it is not new accuracy evidence. The Week 11 development comparison recorded 12/40 exactly correct strings for ResNet-18 versus 8/40 for the baseline CNN. Most strings remain imperfect. Test fresh photos before final performance claims and preserve the earlier experiments.
