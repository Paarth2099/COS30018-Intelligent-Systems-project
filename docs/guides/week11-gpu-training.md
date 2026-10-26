# Week 11 GPU training

This package adds ResNet-18 and MobileNetV3-Small to the existing MLP and baseline
CNN model factories. It prepares and runs training; it does not constitute a
completed four-model comparison. Benchmark/photo evaluation, the comparison
report and final model selection follow after training. GUI choices remain MLP/CNN
until the selected models are integrated and verified.

The imported `week11_gpu_run01` training runs have now been evaluated. See the
[Week 11 comparison report](../reports/model-comparison-week11.md) for results and
recommendations. This guide remains the procedure for reproducing training.

## Copy to the PC

Copy the update files into the existing repository root, preserving paths. Do not
replace the PC's `.venv` with the laptop's environment. The full project must also
contain `data/MNIST`, `data/EMNIST`, `evidence/emnist/preparation`,
`experiments/baseline` and `experiments/emnist_byclass/run_01`.
GitHub alone does not include the ignored data folder.

Install Python 3.12 and the project dependencies using CUDA-enabled Torch 2.6.0
and torchvision 0.21.0. The `cu124` packages are documented in the
[official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/).

In PowerShell from the repository root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\scripts\run_week11_training.ps1 -Action Check
```

Check verifies CUDA, tests gradients and checkpoint loading, runs tiny synthetic
training on CPU and GPU, verifies the original training/validation splits, and
runs a synthetic GPU timing pilot. It does not access the official test sets or
fresh final handwriting. A failed check must be resolved before full training.

Then run:

```powershell
.\scripts\run_week11_training.ps1 -Action Train
```

This trains ResNet-18 and MobileNetV3-Small on MNIST, then both on EMNIST ByClass.
Keep the terminal open and prevent the PC from sleeping during training.
Progress prints every 100 batches and each epoch. No API/AI service is needed
once Python training starts. There is no automatic commit or push.

To run only one dataset, add `-Task mnist` or `-Task emnist`.
The default output is `experiments/week11_gpu_run01`. Each task/model has its own
directory containing config, source snapshot, split indices, history, progress,
best checkpoint and completed-run selection JSON.

## Protocol

- Full original train/validation splits, verified against baseline split hashes.
- Five epochs, seed 42, Adam 0.001, batch 256, no augmentation, no pretrained weights.
- Same 28x28 grayscale inputs scaled to [0,1]. No resize to ImageNet dimensions.
- Torchvision default widths, strides, pooling and blocks retained. Only first
  convolution input channels and classifier outputs are adapted (10 or 62).
- EMNIST is transposed to upright exactly once by the existing data loader.
- Float32, deterministic algorithms, cuDNN benchmarking off, AMP/TF32 off.
- CPU dataset storage; only each minibatch and its labels move to CUDA.
- Highest validation accuracy selects the checkpoint; validation loss breaks
  ties, followed by the earlier epoch. Official tests never select checkpoints.
- Checkpoints contain CPU tensors so the existing CPU prediction API can load them.
- Device, CUDA/runtime versions, source/split hashes and training settings are saved.

Historical MLP/CNN baselines were trained on CPU (the MNIST run on another OS).
GPU results need not be bit-identical across devices. Equal epochs do not prove
each architecture is optimally trained. Historical training times must not be
presented as a controlled architecture speed comparison. Measure inference for
all four on one device later. These are single-seed comparisons.

## Interruptions and errors

Completed runs are hash-checked and skipped when the command is repeated.
Incomplete runs are deliberately not overwritten, and batch/epoch resume is not
implemented. Preserve any partial run. Use a new output root and the relevant task:

```powershell
.\scripts\run_week11_training.ps1 -Action Train -Task emnist -Output experiments/week11_gpu_retry01
```

CUDA is explicit in the launcher: missing GPU support fails instead of silently
starting a long CPU run. For out-of-memory errors, stop and review the pilot;
changing batch size changes the protocol and must be recorded. The direct Python
CLI supports `--batch-size` if a documented adjustment is needed.

## Sources

- [Torchvision 0.21 ResNet-18](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.resnet18.html)
- [Torchvision 0.21 MobileNetV3-Small](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.mobilenet_v3_small.html)

These builders use `weights=None`; published ImageNet accuracies do not describe
the handwritten classifiers being trained here.
