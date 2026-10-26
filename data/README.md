# Collected handwriting photographs

These original photographs are included with the GitHub ZIP. No resizing or pixel changes were made when publishing them.

| Folder | Images | Purpose |
|---|---:|---|
| `segmentation_photos/` | 40 | Development digits and multi-digit numbers |
| `own_characters/development/` | 80 | Development single letters and digits |
| `own_strings/development/` | 40 | Development handwritten strings |
| `final_evaluation/` | 120 | Previously evaluated, frozen final collection |

## Use in the GUI

For number generation, select `data/segmentation_photos`. For alphanumeric generation, select `data/own_characters/development`. Writer subfolders are searched automatically. Generation needs an individual sample for every requested character; this small collection does not cover all 62 classes. Missing samples produce an explanatory error.

Any of these photos can also be loaded individually. Preserve filename labels, case and leading zeros. Multi-character photographs are not individual-character generator tiles.

`photo_manifest.json` lists every included image and its SHA-256. The final labels and frozen configuration are in `results/final_evaluation/`. The final collection has already been analysed: it is not an untouched test set for future changes. Do not mix it into development or training.

MNIST/EMNIST downloads, scratch validation images and detailed experiment outputs are excluded. The application uses included model checkpoints and does not need those datasets. See the main installation guide for Python dependencies and the research archive guide for detailed experiment reproduction.
