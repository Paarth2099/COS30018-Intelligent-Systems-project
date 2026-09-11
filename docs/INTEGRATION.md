# Prediction interface for Paarth, Manula and Saniru

## Input agreement

Pass a floating-point NumPy array or PyTorch tensor with shape `[N,1,28,28]` or
`[N,28,28]`. Each item must contain exactly one prepared digit. Values are 0..1,
with light strokes on a dark background. Crops should be centred and padded after
aspect-ratio-preserving resizing, and supplied in left-to-right order.

`uint8` 0..255 arrays are deliberately rejected so missing scaling is noticed.
The classifier does not resize, invert, crop or segment photos. It cannot detect
all invalid content: a correctly shaped blank crop still gets a digit prediction.

## Python usage

```python
from pathlib import Path
from hnrs_ml.predict import DigitPredictor

# For a root-level GUI script. Adapt the root if your GUI is in a subdirectory.
ROOT = Path(__file__).resolve().parent
predictor = DigitPredictor(ROOT / "experiments/baseline/cnn.pt")

# ordered_crops is provided by your preprocessing/segmentation functions.
result = predictor.predict_number(ordered_crops)
recognised_text = result["text"]
per_digit_results = result["digits"]
```

Load the predictor once when the application starts and reuse it. To compare
models from the GUI, load `mlp.pt` as a separate predictor; do not retrain on each
button click. Keep inference away from a GUI's event loop if it affects response.

The result has this structure (illustrative values, not a recorded experiment):

```json
{"text":"007","digits":[{"digit":0,"confidence":0.98},{"digit":0,"confidence":0.97},{"digit":7,"confidence":0.95}],"model":"cnn"}
```

Preserve the text string. Converting to an integer loses leading zeros. Confidence
is an uncalibrated softmax score; label it as a model score in the interface.

## Errors and empty input

- An empty batch with shape `[0,1,28,28]` returns empty text and an empty digit list.
  The GUI should show "No digits detected" in that case.
- Wrong dimensions, integer inputs, NaN/infinity and out-of-range values raise
  `ValueError`. Surface a useful message and fix preprocessing at its source.
- `predict_digits(crops, batch_size=256)` returns individual predictions in order.
- The CLI accepts only already prepared 28x28 files and converts their grayscale
  pixel values to the required range. Supply files in left-to-right order.

## Integration acceptance checks

1. All teammates can run the saved example and get the expected recorded output.
2. A single prepared crop returns one result and a number returns one result per
   crop, in the same order.
3. The GUI displays input, crops and predicted text for each required image route.
4. Real-photo labels are recorded independently; digit accuracy and exact-number
   accuracy are measured, and segmentation errors are tracked separately.
5. Development photos are used to improve preprocessing. Keep separate final
   photos for the end-to-end evaluation. Do not report generated known-crop
   accuracy as real-photo recognition performance.

The model code has been checked locally; these team integration checks remain open.
