# Machine-learning component: baseline methods, results and limitations

This section documents an executed baseline comparison for the COS30018 Option B
project. It can be adapted into the team report after review and integration. It
covers the digit classifier, not a finished handwritten-number application.

## Question and candidate techniques

The initial question was whether a small convolutional neural network (CNN) would
recognise MNIST digits more accurately than a multilayer perceptron (MLP) under
the same data split and training budget.

The MLP receives 784 pixel values and passes them through fully connected hidden
layers. It is a simple reference with few architectural components. It does not
explicitly share local filters across the image. The CNN uses learned 3x3 filters
and pooling before its classifier, providing an architectural bias toward local
image patterns. This motivated testing it, but the final choice was based on
validation evidence rather than assuming a CNN must always be better.

| Component | MLP | CNN |
|---|---|---|
| Input | One 28x28 grayscale digit | Same |
| Architecture | Flatten -> 128 ReLU -> 64 ReLU -> 10 logits | 16-channel 3x3 convolution + ReLU + 2x2 max pool -> 32-channel 3x3 convolution + ReLU + 2x2 max pool -> flatten -> 64 ReLU -> 10 logits |
| Padding | Not applicable | 1 pixel on each convolution |
| Dropout/augmentation | None | None |
| Output | Ten class logits | Ten class logits |

This is a controlled baseline, not an exhaustive model or hyperparameter search.
The architectures have similar parameter counts but different computational cost.
Both share the optimizer and epoch budget; that does not establish that either
architecture has been individually tuned to its best possible performance.

## Dataset and method

MNIST provides 60,000 original training images and 10,000 official test images.
A deterministic random permutation with seed 42 split the original training set
into 54,000 training and 6,000 validation examples. The exact indices are saved in
`split_indices.npz`, with hashes in `config.json`. The training script never loads
the official test partition. Pixels are divided by 255, with no augmentation or
additional thresholding. This is appropriate for the MNIST baseline; it does not
choose the best preprocessing for photographs.

Both models were trained for five epochs with Adam, learning rate 0.001, batch
size 256 and cross-entropy loss. Initialization seeds and minibatch-order seeds
were fixed; a separate minibatch generator gives both architectures the same
example order. CPU execution used four threads. The environment and versions are
saved in `config.json` and `evidence/verified-environment.txt`.

After every epoch, validation accuracy and loss were measured. The highest
validation accuracy selected each checkpoint, with lower validation loss used
as the tie-breaker. The same rule selected the model for integration. Both best
checkpoints occurred at epoch 5. The CNN was selected before the final test run.
Checkpoint hashes ensure evaluation loads the same selected weights.

Deterministic settings improve reproducibility in this environment; exact results
and timings are not guaranteed across hardware, library versions or platforms.

## Single-digit results

| Model | Parameters | Best epoch | Validation accuracy | Test correct / accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|---:|
| MLP | 109,386 | 5 | 96.22% | 9,660/10,000 (96.60%) | 0.9657 |
| CNN | 105,866 | 5 | 98.42% | 9,853/10,000 (98.53%) | 0.9853 |

The CNN improves test accuracy by 1.93 percentage points in this run, reducing
mistakes from 340 to 147 out of 10,000. Its validation accuracy was also higher,
which is the actual reason it was selected. Macro-F1 is the unweighted mean of
per-digit F1 scores; per-class precision, recall and support are saved separately.

Recorded training-plus-validation wall time was approximately 0.75 seconds for
the MLP and 42.65 seconds for the CNN in this execution environment. A warmed,
single CPU batch-inference pass over all 10,000 images took approximately 0.059
and 0.333 seconds respectively. These are raw local measurements from one run,
not portable speed guarantees or GUI latency benchmarks. Startup, image loading,
preprocessing and segmentation are excluded from inference timing. Repeat timing
under controlled conditions on the deployment computer before making speed claims.

## Generated multi-digit reconstruction

The selected CNN's test predictions were assembled into 1,000 strings: 250 each
of lengths 2, 3, 4 and 5. The source order was fixed with seed 2026. A total of
3,500 test digits were used without reusing a source digit across strings. These
digits are a subset of the same 10,000-image benchmark, so the string results are
not statistically independent of the single-digit results.

| Digits per string | Strings | Exact matches | Exact-string accuracy |
|---:|---:|---:|---:|
| 2 | 250 | 243 | 97.20% |
| 3 | 250 | 243 | 97.20% |
| 4 | 250 | 237 | 94.80% |
| 5 | 250 | 234 | 93.60% |

Overall, 957 of 1,000 strings matched exactly (95.70%). This measures recognition
and ordered reconstruction when digit boundaries are already known. The script
creates a visual preview, but does not feed a merged image through a segmentation
algorithm. It therefore does not satisfy the entire team's end-to-end multi-digit
evaluation requirement by itself. The public predictor was also exercised on one
three-digit batch and checked against the cached benchmark predictions.

The first stored three-digit example has true label `966` and predicted text
`766`. This failure was retained without selecting a nicer example. The displayed
`01` and `0040` examples illustrate why number outputs should remain strings.

## Error analysis

The first 24 CNN test errors are shown in `cnn_errors.png`, in original dataset
order. The largest off-diagonal confusion counts include:

| True digit | Predicted digit | Count |
|---:|---:|---:|
| 4 | 9 | 8 |
| 6 | 0 | 7 |
| 2 | 8 | 7 |
| 9 | 7 | 6 |
| 5 | 3 | 6 |

Visual inspection shows ambiguous and unusual stroke shapes among the failures.
This is a qualitative observation, not a measured causal explanation. The
confusion matrix and per-image CSV files allow specific failures to be revisited.
A softmax score accompanies each prediction, but it has not been calibrated and
should not be presented as a guaranteed probability of correctness.

## Verification and handoff

Automated checks cover disjoint/repeatable splits, valid input shapes and ranges,
checkpoint save/load equivalence for both architectures, digit order, leading
zeros, empty batches, batching, and metrics on a hand-calculated example. The
prepared-crop command-line example was run against the saved CNN. See the
verification record and the integration guide for the exact contract.

The prediction component accepts one prepared digit per 28x28 crop, with values
0..1 and light strokes on a dark background. It returns one label and score per
crop, preserving order. It does not replace Paarth's preprocessing, Manula's
segmentation or Saniru's GUI and image acquisition.

## Limitations and next experiments

Only one training seed was run. The comparison does not quantify training-run
variability or statistical significance. Five epochs and one parameter setting
per model are an initial budget, not evidence of convergence or optimal tuning.
No real group handwriting photos were available to this task, so lighting,
perspective, touching digits, background clutter and writer generalisation have
not been evaluated. Blank or non-digit crops can still receive confident labels.
The Windows setup and cross-computer integration remain unverified.

Next, compare preprocessing on development photos with the selected weights held
fixed, integrate segmentation and evaluate exact-number accuracy on separately
reserved real photos. Additional modelling experiments should use validation data
for choices and a new evaluation protocol if repeated test inspection influences
design. The proposed EMNIST alphanumeric extension remains outside this baseline;
tutor approval has not been established by the supplied discussion.

## Sources consulted

- [PyTorch MNIST dataset loader](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html): dataset access and partition options.
- [Scikit-learn supervised neural-network documentation](https://scikit-learn.org/stable/modules/neural_networks_supervised.html): conceptual MLP background; scikit-learn is not used in this implementation.
- [PyTorch Conv2d documentation](https://docs.pytorch.org/docs/2.6/generated/torch.nn.Conv2d.html): convolution operations and shape parameters.
- [Official PyTorch MNIST example](https://github.com/pytorch/examples/blob/main/mnist/main.py): reference training workflow; this project's architecture, split, settings and reported results are its own implementation and run.
- [PyTorch serialization](https://docs.pytorch.org/docs/2.6/notes/serialization.html): state dictionaries and weights-only loading.
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/2.6/notes/randomness.html): random seeds, deterministic operations and reproducibility limits.

The numerical findings above come from this repository's saved experiment
records, not from published example accuracies.
