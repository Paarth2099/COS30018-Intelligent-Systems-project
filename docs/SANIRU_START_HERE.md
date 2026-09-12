# Saniru's GUI and image acquisition

Your component provides the controls people use to interact with the HNRS.
It connects two required image-input routes to the existing recognition pipeline:
load an image file, or create a number image from a folder of individual digits.

The GUI and image acquisition carry 6 marks in the supplied Option B rubric.
Model/settings selection, intermediate-image views and result export make the
application easier to demonstrate and investigate. This completes an initial
local interface, not the extension or final project submission.

## Start the application

Open the inner project folder containing app.py and requirements.txt in VS Code.
Use the existing Python 3.12 virtual environment. In PowerShell:

```powershell
.\.venv\Scripts\python.exe app.py
```

Keep the terminal open while the window is running. No retraining, API key,
internet service or new pip dependency is needed for the local demonstration.
Saved models must exist in experiments/baseline/cnn.pt and mlp.pt.
The minimum window size is 1020 by 860 pixels; use a display with room for that
window and its title bar. Smaller-screen layout is not supported in this version.

For a fresh Windows machine, create its own environment rather than copying .venv:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe app.py
```

Tkinter is included in the standard Windows Python installation; if missing,
modify that installation to include Tcl/Tk support. The app uses Tk's native PNG
display to avoid a Pillow ImageTk DLL path-length failure found in this checkout.
The final app has been exercised on this Windows machine; other installations
still need a clean setup check.

## Route 1 Load an existing photo

1. Select Open a photo, then Choose image.
2. Open data/segmentation_photos/Manula/Manula_07_label-420.jpg.
3. Keep cnn, components and otsu selected.
4. Select Recognise. The recorded result is 420.
5. Inspect the input, numbered digit regions, prepared crops and model scores.
6. Select Export result and choose a destination such as a shared OneDrive
   experiment folder. A new timestamped subfolder preserves previous exports.

The filename need not contain a label. The prediction is computed from the image,
not from its filename. The GUI preserves leading zeros by displaying text.

## Route 2 Generate a number from digit images

1. Select Generate a number.
2. Browse to data/segmentation_photos, or another folder of digit photos.
3. Enter a number with 1 to 12 digits, such as 007, and select Generate image.
4. The app finds labelled individual-digit files recursively, prepares those
   handwriting samples, and joins them with space between the digits.
5. Select Recognise to read the generated image through the same pipeline.
6. Export result to save input.png, predictions and provenance.

Accepted source names include Paarth_01_label-0.jpg or 0.png. Multi-digit filenames
are excluded from the tile catalog. If a requested digit is missing or unreadable,
the app shows a message. Source photos are not overwritten. Sample selection uses
a fixed seed so repeated demonstrations with the same folder are reproducible.

The requested number is a generation instruction, not a guaranteed recognition
result. A generated 007 image was predicted as 007 by the CNN and 002 by the MLP
in the recorded window test. These examples are functional checks, not benchmark
accuracy estimates. Generated images are assembled from prepared samples and may
be easier than photographs of complete handwritten numbers.

## Settings and results

- Model: cnn or mlp, loaded from Haresh's saved weights and cached after first use.
- Segmentation: components or projection from Manula's implementation.
- Preprocessing: otsu, fixed or grayscale from Paarth's implementation.
- Changing a setting clears the old result; select Recognise again.
- Recognition and generation run on a worker thread. Only the main thread updates
  Tk widgets. Input controls are disabled while a job runs.
- Blank input displays No digits detected. Warnings and failures are shown.
- Per-digit percentages are uncalibrated model scores, not verified probabilities.

## Exported evidence

Each export creates a new gui_TIMESTAMP folder with input.png,
detected_digits.png, foreground_mask.png, crop images, prepared 28x28 images,
prepared_crops.npy and result.json. The JSON includes settings, model hash,
predicted text, scores, boxes, warnings and input provenance. Generated-input
metadata includes the requested text and source image hashes separately from
the recogniser's predictions. Box coordinates refer to the resized grayscale
image shown in detected_digits.png.

## Code ownership and connections

- app.py: window, controls, dialogs, worker/main-thread communication, previews.
- gui_core.py: file loading, single-digit catalog, number-image generation, export.
- segment_digits.recognise_number: Manula's pipeline entry point, used by the GUI.
- preprocess_digits.prepare_variants: Paarth's preparation, reused by both routes.
- hnrs_ml.predict.DigitPredictor: Haresh's saved-model prediction interface.

Saniru's code does not train or alter the models. The application supports digits
only; it does not claim to implement the pending alphanumeric extension.

## Verification and next action

18 automated tests passed, including five acquisition/export tests. The separate
tests/gui_smoke.py exercise opens the real Tk window, drives button callbacks,
substitutes dialog choices, runs the real models, and saves screenshots and a
verification JSON under evidence/gui. It tests both input routes, exports,
model/segmentation switching, result invalidation, cancellation, invalid requested
text and a blank image. The file-picker interaction itself is mocked in this
automated check; manually choose a file and export folder for your own demo.

Run both routes yourself and retain your screenshots. Explain how input reaches
the existing modules, and record actual work, learning and time in your worklog.
Code, automated execution and documentation were produced with Codex assistance
at Saniru's request. Do not present these as unperformed independent work.

The next team tasks are checking the application on another member's machine,
evaluating on new held-out handwriting, handling known recognition/segmentation
limitations, the tutor-approved extension, and the final report/video.

## References

- Python 3.12 Tkinter documentation: https://docs.python.org/3.12/library/tkinter.html
- Project pipeline interface: segment_digits.py and docs/MANULA_START_HERE.md.
- Model contract: docs/INTEGRATION.md.
