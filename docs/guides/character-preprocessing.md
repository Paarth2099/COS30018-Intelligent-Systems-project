# Character-photo preprocessing

The preparation pipeline is integrated with string segmentation and the desktop GUI. The comparison covers 80 individual-character development photos; see [the measured results](../reports/character-preprocessing-results.md).

## Processing pipeline

The new code corrects EXIF orientation, estimates uneven paper lighting, converts to grayscale, localises the complete character, filters tiny specks, resizes without stretching, adds black padding and normalises the final ink brightness. It returns upright float32 28x28 images in [0,1], compatible with the character predictor.

The selected default is Otsu with local lighting correction, conservative speck filtering and peak normalisation. Fixed thresholding tied its 53/80 development score. Neither method is guaranteed to win on new photos.

The extension reuses the existing Otsu threshold and connected-region utilities. Component detection removes small noise within one character. Whole-string separation is implemented independently in `hnrs_chars/segmentation.py` and calls this preparation function on each ordered crop.

| File | Purpose |
|---|---|
| `hnrs_chars/preprocessing.py` | `prepare_character_variants` with diagnostics and `prepare_character` for pipeline integration |
| `hnrs_chars/photo_experiment.py` | Validate labels and compare all three methods using the frozen selected CNN |
| `hnrs_chars/photo_predict.py` | Demonstrate a single photo without passing any reference label to the model |
| `hnrs_chars/photo_report.py` | Generate the report from the four saved development runs |
| `tests/test_character_preprocessing.py` | Nine behavioural regression tests |
| `run_preprocessing.ps1` | Run the demo, comparison or tests from the repository root |

## Commands

Open the **project-github** folder in VS Code. In its terminal:

```powershell
.\run_preprocessing.ps1 -Action Demo
```

This uses `data/own_characters/development/Paarth/Paarth_02_lower-i.jpg`, an existing development example, and saves the prediction and comparison preview in a new timestamped folder under `evidence/character_preprocessing/`. It should not be presented as an unseen test.

For any other single-character photo:

```powershell
.\run_preprocessing.ps1 -Action Demo -Image "data/own_characters/development/Saniru/Saniru_14_lower-i.jpeg"
```

That example intentionally demonstrates a retained failure. Read the actual predicted character and warnings. Labels do not control the answer.

```powershell
.\run_preprocessing.ps1 -Action Test
.\run_preprocessing.ps1 -Action Compare
```

Compare creates a new timestamped run using all development photos and current defaults. It never overwrites an earlier run. It does not retrain the CNN.

Create a local `.venv` and install `requirements.txt` as described in [installation](installation.md). Copy the development photos into the same relative data folder. Raw photos are ignored by Git; the committed previews, prepared arrays and results provide evidence but do not replace the raw inputs for reproduction.

## Preparation interface

```python
import numpy as np
from PIL import Image
from hnrs_chars.preprocessing import prepare_character
from hnrs_chars.predict import CharacterPredictor

# Supply complete character crops in reading order, including dots and a small
# pale-paper margin. No whole-word image and no already-normalised EMNIST tile.
with Image.open("data/own_characters/development/Paarth/Paarth_02_lower-i.jpg") as photo:
    prepared = prepare_character(photo)
predictor = CharacterPredictor("experiments/emnist_byclass/run_01/cnn.pt")
result = predictor.predict_text(prepared[None, :, :])
print(result["text"])

# For a batch of segmented character crops:
# batch = np.stack([prepare_character(crop) for crop in ordered_crops]) if ordered_crops else np.empty((0,28,28), np.float32)
# result = predictor.predict_text(batch)
```

`prepare_character(photo, method="otsu")` returns one array. It raises ValueError for blank/very low contrast inputs, empty prepared output or unknown methods. For warnings, crop bounds and the three variants, use:

```python
from hnrs_chars.preprocessing import prepare_character_variants
variants, common_crop, diagnostics = prepare_character_variants(photo)
warnings = diagnostics["warnings"]
```

The example `photo` must still be open when calling these functions. No filename, true label, expected number of characters or model prediction is used to choose its crop. Background should be pale and ink dark; ordinary photos are not transposed as raw EMNIST images are.

All retained components form the crop. Unlike the old digit segmentation filter, there is no minimum component-height filter. The area cutoff is max(3 pixels, 0.5% of the largest component), applied at the resized working resolution. Small detached dots survive the regression examples and most photo examples; genuinely tiny dots below that cutoff remain a limitation. The algorithm does not invent missing dots, change case, thicken strokes or correct spelling.

## Reproduce the four comparisons

With an activated environment, use new output folder names:

```powershell
python -m hnrs_chars.photo_experiment --output evidence/character_preprocessing/repeat_01 --illumination none --keep-specks --no-ink-normalisation
python -m hnrs_chars.photo_experiment --output evidence/character_preprocessing/repeat_02 --keep-specks --no-ink-normalisation
python -m hnrs_chars.photo_experiment --output evidence/character_preprocessing/repeat_03 --no-ink-normalisation
python -m hnrs_chars.photo_experiment --output evidence/character_preprocessing/repeat_04
```

Each run's source snapshot records the code actually used at that time. The final code exposes the earlier settings for repeat experiments. Three representations share each run's lighting, speck removal, localisation, resizing, padding and normalisation, so the within-run comparison changes only the representation. Cross-run changes were sequential development interventions, not independent held-out experiments.
