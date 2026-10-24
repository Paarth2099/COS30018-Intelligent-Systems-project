# Project documentation

Commands and plain source paths are relative to the repository root unless stated otherwise.

## Guides

| Document | Contents |
|---|---|
| [Installation and datasets](guides/installation.md) | Environment, models, photos and launch commands |
| [GUI user guide](guides/gui.md) | Modes, input routes, controls and export |
| [System architecture](architecture.md) | Processing stages and module interfaces |
| [Character models](guides/character-models.md) | EMNIST preparation, training and prediction API |
| [Week 11 GPU training](guides/week11-gpu-training.md) | ResNet-18 and MobileNetV3-Small setup, GPU checks and experiment protocol |
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
| [Week 9 photo-model comparison](reports/photo-model-comparison-week09.md) | Paired MLP/CNN comparison on all 160 development photos |
| [Week 9 preprocessing review](reports/preprocessing-review-week09.md) | Visual diagnosis of 100 individual inputs and controlled Week 10 experiment plan |
| [Week 9 segmentation review](reports/segmentation-review-week09.md) | Visual diagnosis of 60 multi-character inputs, grouping failures and experiment plan |
| [Week 9 local GUI review](reports/gui-review-week09.md) | Functional checks, display-scaling defect and pending acceptance tests |
| [Week 10 controlled model experiment](reports/model-experiment-week10.md) | Class-weighted CNN training, validation and paired photo comparison; baseline retained |
| [Week 10 preprocessing experiments](reports/preprocessing-experiment-week10.md) | Three controlled candidates, rejected character changes and adopted digit contrast normalisation |
| [Week 10 segmentation experiments](reports/segmentation-experiment-week10.md) | Independent and combined broken-body/short-stem repairs, visual review and integration checks |
| [Week 10 GUI improvements](reports/gui-improvements-week10.md) | Screen-fit repair, scrolling, updated recognition integration and local verification |
| [Week 11 four-model comparison](reports/model-comparison-week11.md) | Imported GPU training, four-architecture benchmark/photo evaluation and model recommendations |
| [Week 11 GUI integration](reports/gui-integration-week11.md) | Four-model selector, evaluated defaults, checkpoint integrity and desktop verification |

## Data notes

- [Character-photo review](data/character-photo-review.md)
- [String-photo review](data/string-photo-review.md)

Metrics, snapshots, previews and checkpoints remain under `evidence/` and `experiments/`. These component reports document recorded experiments; they do not replace the final team project report.
