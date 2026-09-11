# COS30018 Intelligent Systems project

Haresh's machine-learning component for Option B: handwritten number recognition.
This branch supplies the digit classifiers and prediction interface; preprocessing,
segmentation, image acquisition and the team GUI still need integration.

## Verified baseline

| Model | Validation accuracy | MNIST test accuracy |
|---|---:|---:|
| MLP | 96.22% | 96.60% |
| CNN | 98.42% | 98.53% |

The CNN was selected using validation results before test evaluation. These are
single-seed, five-epoch benchmark results, not real-photo application accuracy.
The model also reconstructed 957 of 1,000 generated test strings correctly when
given their known digit crops. This does not evaluate segmentation.

## Start here

- [Beginner walkthrough and setup](docs/START_HERE.md)
- [Model comparison and limitations](docs/ML_REPORT.md)
- [Integration instructions for teammates](docs/INTEGRATION.md)
- [Evidence and individual worklog guidance](docs/CONTRIBUTION_RECORD.md)
- [Actual experiment settings](experiments/baseline/config.json)
- [Actual final evaluation](experiments/baseline/evaluation/results.json)

Use Python 3.12. From the repository directory on macOS:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m hnrs_ml.predict --checkpoint experiments/baseline/cnn.pt --crops experiments/baseline/evaluation/example_crops/digit_0.png experiments/baseline/evaluation/example_crops/digit_1.png experiments/baseline/evaluation/example_crops/digit_2.png
```

The supplied demo really predicts **766** for an image labelled **966**. It is a
retained failure example, useful for explaining limitations. The preview below
also includes successful examples. No data download or retraining is needed for
this demo because trained models and the three small crops are included.

![Generated examples with known digit boundaries](experiments/baseline/evaluation/generated_strings.png)

To inspect MNIST, run `python inspect_mnist.py`. To reproduce training and final
evaluation in a new output directory:

```sh
python -m hnrs_ml.train --download --output-dir experiments/reproduction
python -m hnrs_ml.evaluate --experiment-dir experiments/reproduction
```

Training and evaluation refuse to overwrite an existing experiment record.
Use validation data for future improvements. Once test results have been examined,
do not repeatedly tune against them and describe that as an untouched final test.

The code, experiments and documentation were produced with Codex assistance at
Haresh's request. Haresh should review and understand the implementation and
record his actual learning, changes and time under the unit's AI-use rules.
