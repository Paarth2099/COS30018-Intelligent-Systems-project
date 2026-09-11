# Haresh: what your part does

Your group is making a program that reads handwritten numbers. For a photo of
583, Paarth prepares the image, Manula separates the three digits, your model
recognises each digit, and Saniru displays the reconstructed text through the GUI.

Your responsibility is the machine-learning part. The PDF requires model research,
comparisons and meaningful predictions using MNIST. The exact MLP/CNN choice and
the allocation to you come from your group discussion. PyTorch is the framework
chosen for this implementation; the earlier repository contained no framework.

The last message in the exported discussion delayed training until after setup.
Following your new request to do the ML work, the baseline models have now also
been trained and evaluated. Your teammates can still follow their agreed order.

## What has been completed

1. A fixed Python dependency list and MNIST inspection script.
2. An MLP and CNN with the same digit input/output format.
3. Five training epochs per model on the same 54,000 training examples.
4. Selection using the same 6,000 validation images; both best checkpoints were
   from epoch 5 and the CNN was selected.
5. Evaluation of both checkpoints on the 10,000 official MNIST test examples.
6. Reconstruction of 1,000 generated strings from known test-digit crops.
7. Saved weights, prediction code, failure images, confusion matrices, CSV/JSON
   evidence, and tests for integration risks.

## Words you need to understand

| Term | Meaning here |
|---|---|
| MNIST | Labelled images of handwritten digits, each 28 by 28 pixels |
| Label | The known correct digit, such as 7 |
| Training | Adjusting model weights from examples and their labels |
| Epoch | One pass through all 54,000 training examples |
| Batch | Up to 256 examples processed together during training |
| MLP | A neural network that takes the 784 pixels as a flat list |
| CNN | A neural network using small filters across the image to learn spatial features |
| Validation | Separate development images used to choose the model and checkpoint |
| Test | Images excluded from training and model selection, used for reporting |
| Checkpoint | A saved set of learned weights and its settings |
| Accuracy | The fraction of labels predicted correctly |
| Confidence | Here, a softmax score; it is not a verified chance of being correct |

## Run the project on a Mac

Open Terminal in the repository folder and use Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

A virtual environment keeps the project's libraries together. Each teammate
creates their own. This baseline was verified with Python 3.12.14 on Apple Silicon
macOS, using CPU execution. No GPU is required. The pinned framework versions
were not selected for Python 3.14; if Python 3.12 is missing, install/configure
Python 3.12 before following these steps.

## Run the project on Windows (Command Prompt)

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For later commands, use `.venv\Scripts\python.exe` in place of `python`.
Windows instructions are provided but have not been tested on a Windows computer.

## See a prediction without training again

```sh
python -m hnrs_ml.predict --checkpoint experiments/baseline/cnn.pt --crops experiments/baseline/evaluation/example_crops/digit_0.png experiments/baseline/evaluation/example_crops/digit_1.png experiments/baseline/evaluation/example_crops/digit_2.png
```

The output text is `766`, while the correct label is `966`. This is the first
three-digit sequence from the preselected evaluation order, retained even though
the first digit is wrong. It demonstrates that 98.53% digit accuracy does not mean
every number will be correct. See `generated_strings.png` in the same evaluation
folder for successful and failed examples.

## Inspect the data

```sh
python inspect_mnist.py
```

This downloads MNIST if it is missing and saves a labelled grid under `evidence/`.
The grid labels are supplied answers, not model predictions. Raw training data
has shape `[60000,28,28]`. The model receives batches `[N,1,28,28]`, where N is the
number of images and 1 is the grayscale channel. Pixel values are scaled to 0..1.

## Understand the code in this order

1. `inspect_mnist.py`: loads and checks the original dataset.
2. `hnrs_ml/data.py`: scales pixels and establishes a reproducible split.
3. `hnrs_ml/models.py`: defines the two architectures.
4. `hnrs_ml/train.py`: trains, measures validation performance and saves weights.
5. `hnrs_ml/predict.py`: loads weights and recognises prepared crops in order.
6. `hnrs_ml/evaluate.py`: measures final performance without changing weights.
7. `tests/test_contract.py`: checks splitting, saved weights, invalid inputs,
   leading zeros, order, batching and a hand-calculated metric example.

## What still depends on your team

- Paarth and Manula must provide correct, ordered 28x28 digit crops.
- Saniru must connect the predictor to the GUI and both required input methods.
- The group must evaluate real handwriting photos, segmentation and complete
  application behaviour. The photos mentioned in the HTML were not supplied here.
- The extension still needs confirmed tutor approval. The HTML says the email
  was sent; it does not establish approval. This package recognises digits only.
- You must review the code, demonstrate understanding, and record your actual
  contribution in your worklog and sprint reports. A test run cannot establish
  hours worked or learning you have not yet done.

This is a complete runnable ML baseline, not the entire group's final submission.
