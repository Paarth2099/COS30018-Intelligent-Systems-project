# Project documentation

Start with the [current project overview](../README.md), [selected results](../results/README.md), [research commands](../research/README.md), and [complete evidence archive](guides/research-archive.md).

The reports below document successive studies; earlier defaults and pending tasks describe the state at the time of each study. The current README and final-evaluation summaries describe the selected system.

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

- [Final evaluation protocol](guides/final-evaluation-week11.md) â€” planned procedure, candidate configuration and scoring gates.
- [Final photo assignments](guides/final-photo-assignments.md) â€” exact 30-photo collection for each member.

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

- [Week 11 final collection review](reports/final-collection-review-week11.md) â€” 120-photo integrity and visual review, with pending label/provenance confirmations.

- [Week 11 final fresh-photo evaluation](reports/final-evaluation-week11.md) â€” frozen 120-photo run, four-model comparison, timing and independently verified metrics.

- [Week 11 final preprocessing review](reports/preprocessing-final-review-week11.md) â€” all 120 selected-route inputs, visible damage cases and limits of attribution.

- [Week 11 final segmentation review](reports/segmentation-final-review-week11.md) â€” all 40 multi-character inputs, 80 method outputs and visual split/merge/dot review.

## Data notes

- [Character-photo review](data/character-photo-review.md)
- [String-photo review](data/string-photo-review.md)

Metrics, snapshots, previews and checkpoints remain under `evidence/` and `experiments/`. These component reports document recorded experiments; they do not replace the final team project report.

## Post-evaluation improvement study

- [Non-GUI study and decisions](reports/improvement-study-week11.md)
- [RTX transfer and run instructions](guides/improvement-gpu-transfer.md)
- [Fresh v2 photo assignments](guides/fresh-evaluation-v2-photos.md)

- [Full GPU augmentation results and selection decision](reports/gpu-augmentation-results-week11.md) - all eight runs verified; production retained; follow-up photography on hold.
