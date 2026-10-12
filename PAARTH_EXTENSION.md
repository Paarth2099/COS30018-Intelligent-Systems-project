# Paarth: character-photo preprocessing handover

Your extension component is implemented and compared on all 80 supplied development photos. Read [the measured report](PAARTH_PREPROCESSING_RESULTS.md) before describing its accuracy. The full alphanumeric extension still needs Manula's string segmentation, Saniru's GUI integration and fresh system evaluation.

## What changed

The new code corrects EXIF orientation, estimates uneven paper lighting, converts to grayscale, localises the complete character, filters tiny specks, resizes without stretching, adds black padding and normalises the final ink brightness. It returns upright float32 28x28 images in [0,1], compatible with Haresh's character predictor.

The selected default is Otsu with local lighting correction, conservative speck filtering and peak normalisation. Fixed thresholding tied its 53/80 development score. Neither method is guaranteed to win on new photos.

Existing digit preparation, segmentation, GUI and all trained checkpoints were left unchanged. The extension reuses the existing Otsu threshold helper and Manula's connected-region utility; those original algorithms are not new Paarth contributions. Component detection here removes small noise within ONE character; it does not implement the upcoming string-segmentation task.

| File | Purpose |
|---|---|
| `hnrs_chars/preprocessing.py` | `prepare_character_variants` with diagnostics and `prepare_character` for team integration |
| `hnrs_chars/photo_experiment.py` | Validate labels and compare all three methods using the frozen selected CNN |
| `hnrs_chars/photo_predict.py` | Demonstrate a single photo without passing any reference label to the model |
| `hnrs_chars/photo_report.py` | Generate the report from the four saved development runs |
| `tests/test_character_preprocessing.py` | Nine behavioural regression tests |
| `run_preprocessing.ps1` | Run the demo, comparison or tests on this computer |

## Demonstrate your work

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

The runner prefers a local `.venv`, otherwise it uses the original downloaded project's verified environment on this laptop. On another computer, create `.venv` and install `requirements.txt`, then copy the source development photos from OneDrive into the same relative data folder. Raw photos are ignored by Git; the committed previews, prepared arrays and results provide evidence but do not replace the raw inputs for reproduction.

## Code contract for Manula

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

# Once Manula has produced ordered_crops:
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

## Evidence for your tutor

1. Show `PAARTH_PREPROCESSING_RESULTS.md` and explain why the 80 images are development data.
2. Show `evidence/character_preprocessing/run_04/Paarth_overview.png` and the other writer overviews.
3. Show a dotted-character success and a retained shadow/texture failure.
4. Explain the shared crop, three representations and why preserving multiple components matters for i/j.
5. Show the input manifest, result CSVs, source/checkpoint hashes and passing test log.
6. Run the demo and explain the actual output, including any error.

Suggested wording after you have reviewed and run it yourself:

> My section prepares handwritten character photos for the 62-class model. We compared grayscale, fixed thresholding and Otsu on 80 development photos. I reviewed the image preparation and results, including lighting, noise and thin-stroke issues. With the final preparation, fixed and Otsu both achieved 53 out of 80, compared with 42 for the original Otsu baseline. We retained the failed examples and still need fresh end-to-end testing. The implementation and experiment execution used Codex assistance.

Use your worklog for your actual collection, review, demonstration and learning activities. Do not count assistant runtime as personal hours or claim independent implementation you did not perform.

## Handover and Git

No Git identity was changed and no commit or push was made for this section. Switch back to Paarth before committing. The report, code, tests and evidence are ready for review; the source photos remain under ignored `data/` and should stay shared through OneDrive.

Next, Manula needs to group dotted characters correctly when splitting separate-character strings and call this preparation function on each complete crop. Saniru then adds the character interface to the GUI. The team's remaining fresh-photo evaluation should include all intended character groups, multiple writers and lighting conditions, with exact character and exact string accuracy recorded separately.
