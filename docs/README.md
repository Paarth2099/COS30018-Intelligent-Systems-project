# Project documentation

Commands and plain source paths are relative to the repository root unless stated otherwise.

## Guides

| Document | Contents |
|---|---|
| [Installation and datasets](guides/installation.md) | Environment, models, photos and launch commands |
| [GUI user guide](guides/gui.md) | Modes, input routes, controls and export |
| [System architecture](architecture.md) | Processing stages and module interfaces |
| [Character models](guides/character-models.md) | EMNIST preparation, training and prediction API |
| [Character preprocessing](guides/character-preprocessing.md) | Methods, diagnostics and comparison commands |
| [String segmentation](guides/string-segmentation.md) | Grouping, ordered crops and recognition API |

## Experiment reports

| Report | Scope |
|---|---|
| [Digit-system results](reports/digit-system-results.md) | MNIST and number-photo development comparisons |
| [Character-model results](reports/emnist-results.md) | MLP/CNN EMNIST benchmark |
| [Character-preprocessing results](reports/character-preprocessing-results.md) | 80 individual-character photos |
| [String-segmentation results](reports/string-segmentation-results.md) | 40 string photos and two methods |
| [GUI verification](reports/gui-verification.md) | Software and desktop workflow checks |

## Data notes

- [Character-photo review](data/character-photo-review.md)
- [String-photo review](data/string-photo-review.md)

Metrics, snapshots, previews and checkpoints remain under `evidence/` and `experiments/`. These component reports document recorded experiments; they do not replace the final team project report.
