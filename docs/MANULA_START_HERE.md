# Manula's segmentation component

## What your part does

For a photo of 420, your component finds the three digit regions, extracts one
crop for each digit, and returns them in left-to-right order. Paarth's existing
function converts each crop into a 28 by 28 model input; Haresh's saved CNN
recognises each one. Saniru can call the combined function from the GUI.

Task 2 requires researching and comparing segmentation methods, implementing a
selected method, and explaining its behaviour. It is worth 5 team marks. The
specific algorithms and division of work come from the team's plan, not a rule
requiring these exact two algorithms.

## Two methods implemented

**Connected components:** neighbouring dark pixels form a region. This
implementation uses 8-connectivity, meaning diagonal neighbours count. Small
noise components are filtered out and the remaining boxes are sorted from left
to right. It can separate spatially disconnected digits even when their boxes
overlap horizontally. A crop removes ink belonging to other labelled components.

**Vertical projection:** after the same foreground preparation and noise removal,
the code counts whether each column has ink. Consecutive occupied columns form
one candidate digit region. It works when a clear vertical gap separates digits,
but may merge separate digits whose strokes occupy overlapping columns.

Both use the same grayscale conversion, autocontrast, Otsu threshold and
component-based noise filtering. This compares the grouping stage, not two
completely independent pipelines. The projection alternative also pays the cost
of the shared component filtering, which matters when interpreting timing.

The implementation uses NumPy and Pillow already in requirements.txt. OpenCV's
official documentation provides background on 4/8-connectivity and component
statistics: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html

For threshold selection, see OpenCV's official Otsu tutorial:
https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

These explain concepts; the component code is a local row-run/union-find
implementation, not an invocation or copy of OpenCV's implementation.

## Run your first demonstration

Open the inner project folder in VS Code. In PowerShell, run:

```powershell
.\.venv\Scripts\python.exe recognise_photo.py --input "data/segmentation_photos/Manula/Manula_07_label-420.jpg" --output evidence/segmentation/my_manula_demo_01
```

The recorded connected-component demo recognises this photo as `420`. Open the
new folder's `preview.png` to see the boxes and prepared crops. The script also
saves the crops, a NumPy input array and prediction.json. Output folders must be
new to avoid overwriting evidence. Use a new suffix for each subsequent run.

Try the alternative on the same image:

```powershell
.\.venv\Scripts\python.exe recognise_photo.py --input "data/segmentation_photos/Manula/Manula_07_label-420.jpg" --output evidence/segmentation/my_projection_demo_01 --method projection
```

The comparison currently merges the 2 and 0 and predicts `47`. This is a useful
recorded limitation, not a result to delete. A recogniser does not receive the
filename's label. Labels are read only by the batch evaluator for scoring.

## Reproduce the team comparison

```powershell
.\.venv\Scripts\python.exe compare_segmentation.py --input data/segmentation_photos --output evidence/segmentation/my_team_run_01 --checkpoint experiments/baseline/cnn.pt --include-single
```

This processes 40 source photos: 20 single-digit and 20 multi-digit images,
five of each type per member. Both methods run on every image. Omit
`--include-single` to evaluate only the multi-digit subset. The checkpoint is
fixed and is not retrained. The evaluator saves masks, raw crops, prepared crops,
boxes, previews, predictions, a CSV, source-code snapshots and hashes.

The final verified development comparison is in
`evidence/segmentation/team_run_02`. The initial run is preserved as `team_run_01`.
The first run revealed neighbour-ink contamination inside overlapping component
boxes. Run 02 removes that contamination; its regression test uses synthetic
overlapping boxes. No images or labels were changed to repair the issue.

## Explain the measurements correctly

- Count match: the number of detected crops equals the number of digits in the
  reference label. This is a useful proxy, not proof that all crops are correct.
- Exact number match: every digit in the predicted string matches, in order.
- Character error rate: Levenshtein insertions, deletions and substitutions
  divided by the number of reference characters; it can exceed 100%.
- Segmentation time: one local measurement per image/method, including shared
  mask generation and crop extraction. It is not a rigorous speed benchmark.

The method was chosen on this small development set. It needs new photos and
harder cases before final assessment. There are no manually annotated digit boxes,
so no bounding-box overlap accuracy is claimed. A correct count or prediction
does not replace visual inspection.

## Code to understand

1. `segment_digits.py`: foreground preparation, `connected_regions`,
   `segment_image`, `prepare_crops`, and the combined `recognise_number` function.
2. `compare_segmentation.py`: label parsing after segmentation, evidence saving
   and scoring. `occupied_intervals` is in segment_digits.py.
3. `recognise_photo.py`: a user-facing command that accepts any image filename.
4. `tests/test_segmentation.py`: geometry, order, blank-input, noise,
   neighbour-isolation and known-limit regression tests.

`segment_image` returns a SegmentationResult containing `gray`, `mask`, `boxes`,
`crops`, warnings and a removed-component count. Boxes use exclusive right/bottom
coordinates in the resized grayscale image, not the original camera image.
`prepare_crops` calls Paarth's unchanged `prepare_variants` and returns a float32
array of shape `[N,28,28]` in `[0,1]`. Blank input returns an empty batch.

## GUI handover to Saniru

```python
from PIL import Image
from hnrs_ml.predict import DigitPredictor
from segment_digits import recognise_number

# Create this once, using a path relative to the application root.
predictor = DigitPredictor("experiments/baseline/cnn.pt")
with Image.open("input_photo.jpg") as image:
    result = recognise_number(image, predictor, method="components")

text = result["prediction"]["text"]
boxes = result["segmentation"].boxes
crops = result["segmentation"].crops
```

Display `text` as a string to retain leading zeros. Empty text should produce
"No digits detected". Surface warnings and input errors. Confidence values are
uncalibrated model scores. The GUI and both image-acquisition routes still need
Saniru's implementation.

## Limitations and your evidence

Supported input is one horizontal line of separated dark handwritten digits on
pale paper. Touching digits can merge; broken digits can split; severe shadows,
paper boundaries, textured backgrounds and multi-line input can fail. Relative
noise filters may remove small legitimate strokes. This component does not
implement the pending alphanumeric extension.

Run the demonstration yourself, inspect its output and explain the two methods.
Record actual work, learning, problems and time in your worklog. Code, tests,
execution and documentation were produced with Codex assistance at Manula's
request; do not claim unperformed independent development or invented hours.
Review these changes with the team before putting your contribution into GitHub.
