# Installation and datasets

## Environment

Use Python 3.12 with Tkinter, PyTorch, torchvision, NumPy and Pillow. Package versions are pinned in [requirements.txt](../../requirements.txt). A CPU is sufficient.

From the repository root on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run_gui.ps1
```

Activation is optional. The launcher prefers the repository's `.venv` and supports the original local baseline environment as a fallback. A fresh checkout should use its own environment.

If PowerShell blocks scripts:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\run_gui.ps1
```

The direct command is `.\.venv\Scripts\python.exe app.py`. On macOS/Linux, create a Python 3.12 environment, install the same requirements and run `python app.py`; Tkinter must be available. The recorded extension desktop verification covers Windows only.

## Saved models

- [Digit checkpoints](../../experiments/baseline/): CNN, MLP and selection records.
- [Character checkpoints](../../experiments/emnist_byclass/run_01/): CNN, MLP and selection records.

The GUI uses the saved weights. Raw MNIST/EMNIST data is required for training and benchmark reproduction, not for classifying a supplied photo.

## Local photo folders

`data/` is excluded from Git. Copy the shared original photos to these locations before running development comparisons or the desktop smoke test:

| Folder | Example filename |
|---|---|
| `data/segmentation_photos/<writer>/` | `Manula_07_label-420.jpg` |
| `data/own_characters/development/<writer>/` | `Manula_06_upper-A.jpg` |
| `data/own_strings/development/<writer>/` | `Manula_02_text-Room7.jpg` |

Photo loading also accepts other folders. Generation needs labelled individual samples and reports missing requested characters. Case matters in Alphanumeric mode.

Manifests under `evidence/` record source labels and hashes. Existing development photos are not an untouched final test. A distributed working package needs suitable input examples or explicit access instructions.

## Commands

```powershell
.\run_gui.ps1
.\run_gui.ps1 -Verify
.\run_characters.ps1 -Action Demo
.\run_preprocessing.ps1 -Action Demo
.\run_segmentation.ps1 -Action Demo
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

GUI verification opens a real window and creates a new folder under `evidence/gui_extension/`. Preprocessing and segmentation demos also preserve earlier outputs by using new folders.

## Rebuilding reports

```powershell
.\.venv\Scripts\python.exe -m hnrs_chars.report
.\.venv\Scripts\python.exe -m hnrs_chars.photo_report
.\.venv\Scripts\python.exe -m hnrs_chars.string_report
```

These commands use saved results without retraining and write to `docs/reports/`. The photo and string builders summarise the four recorded development runs. New experiments require new output folders and their own analysis.
