"""Model training for the GUI's Model lab, independent of Tk so it can be tested.

The user chooses the dataset, architecture and hyperparameters; training runs in
a worker thread and reports progress through a callback. The same fixed
train/validation splits as the recorded experiments are used, and the official
test sets are never loaded. Saved checkpoints use the application's existing
format, so DigitPredictor / CharacterPredictor load them unchanged.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

TASKS = {"mnist": "MNIST digits (10 classes)", "emnist": "EMNIST ByClass characters (62 classes)"}
ARCHITECTURES = ("mlp", "cnn", "resnet18", "mobilenet_v3_small")
OPTIMIZERS = ("adam", "sgd")


@dataclass
class TrainingSettings:
    task: str = "mnist"
    model: str = "cnn"
    epochs: int = 3
    learning_rate: float = 0.001
    batch_size: int = 128
    train_samples: int = 10000
    validation_samples: int = 3000
    optimizer: str = "adam"
    seed: int = 42

    def validate(self):
        """Raise ValueError with a message the GUI can show unchanged."""
        if self.task not in TASKS:
            raise ValueError("Choose MNIST or EMNIST.")
        if self.model not in ARCHITECTURES:
            raise ValueError("Choose one of the four architectures.")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError("Optimizer must be Adam or SGD.")
        if not 1 <= self.epochs <= 50:
            raise ValueError("Epochs must be between 1 and 50.")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("Learning rate must be greater than 0 and at most 1.")
        if not 8 <= self.batch_size <= 1024:
            raise ValueError("Batch size must be between 8 and 1024.")
        if self.train_samples < 100 or self.validation_samples < 100:
            raise ValueError("Use at least 100 training and 100 validation samples.")
        return self


@dataclass
class TaskData:
    """uint8 images [N,28,28] (upright), integer targets and fixed split indices."""
    images: torch.Tensor
    targets: torch.Tensor
    train_ids: torch.Tensor
    validation_ids: torch.Tensor
    classes: list


def dataset_available(root, task):
    raw = Path(root) / "data" / ("MNIST" if task == "mnist" else "EMNIST") / "raw"
    needed = (["train-images-idx3-ubyte", "train-labels-idx1-ubyte"] if task == "mnist" else
              ["emnist-byclass-train-images-idx3-ubyte", "emnist-byclass-train-labels-idx1-ubyte",
               "emnist-byclass-mapping.txt"])
    return all((raw / name).exists() for name in needed)


def load_task_data(root, task, download=False):
    """Training partition plus the project's recorded validation split."""
    root = Path(root)
    if task == "mnist":
        from torchvision.datasets import MNIST
        from hnrs_ml.data import development_split
        dataset = MNIST(root / "data", train=True, download=download)
        train_ids, validation_ids = development_split(len(dataset.targets))
        return TaskData(dataset.data, dataset.targets, train_ids, validation_ids, [str(i) for i in range(10)])
    from hnrs_chars.data import CLASSES, download_byclass, load_partition
    if download and not dataset_available(root, "emnist"):
        download_byclass(root / "data")
    images, targets = load_partition(root / "data", train=True)
    split = np.load(root / "results/datasets/emnist_split_indices.npz")
    return TaskData(images, targets, torch.from_numpy(split["train"]), torch.from_numpy(split["validation"]),
                    list(CLASSES))


def build(task, name):
    if task == "mnist":
        from hnrs_ml.models import build_model
    else:
        from hnrs_chars.models import build_model
    return build_model(name)


def to_batch(images, ids, device):
    return images[ids].unsqueeze(1).to(device=device, dtype=torch.float32).div_(255)


@torch.inference_mode()
def evaluate(model, data, ids, batch_size, device):
    """Return loss, accuracy and predictions for the given indices."""
    model.eval()
    loss, predictions = 0.0, []
    for chunk in ids.split(batch_size):
        labels = data.targets[chunk].to(device)
        logits = model(to_batch(data.images, chunk, device))
        loss += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        predictions.append(logits.argmax(1).cpu())
    predicted = torch.cat(predictions)
    accuracy = (predicted == data.targets[ids]).float().mean().item()
    return loss / len(ids), accuracy, predicted


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train(settings, data, output, progress=lambda event: None, should_stop=lambda: False):
    """Train, keep the best-validation checkpoint and write a run folder.

    progress receives dictionaries with kind 'batch', 'epoch' or 'done'.
    Returns the run record (also saved as run.json). A stop request ends
    training after the current minibatch; completed epochs are kept.
    """
    settings.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(settings.seed)
    generator = torch.Generator().manual_seed(settings.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Seeded subsets of the fixed splits: no validation image is ever trained on.
    train_ids = data.train_ids[torch.randperm(len(data.train_ids), generator=generator)[:settings.train_samples]]
    validation_ids = data.validation_ids[
        torch.randperm(len(data.validation_ids), generator=generator)[:settings.validation_samples]]
    model = build(settings.task, settings.model).to(device)
    parameters = model.parameters()
    optimizer = (torch.optim.Adam(parameters, lr=settings.learning_rate) if settings.optimizer == "adam" else
                 torch.optim.SGD(parameters, lr=settings.learning_rate, momentum=0.9))
    config = dict(asdict(settings), created_utc=datetime.now(timezone.utc).isoformat(), device=str(device),
                  train_count=len(train_ids), validation_count=len(validation_ids),
                  source="GUI Model lab; official test set not used",
                  selection_rule="Highest validation accuracy, then lowest validation loss")
    checkpoint = output / f"{settings.model}.pt"
    history, best, stopped = [], None, False
    batches = (len(train_ids) + settings.batch_size - 1) // settings.batch_size
    started = time.perf_counter()
    for epoch in range(1, settings.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        order = train_ids[torch.randperm(len(train_ids), generator=generator)]
        for number, chunk in enumerate(order.split(settings.batch_size), 1):
            if should_stop():
                stopped = True
                break
            labels = data.targets[chunk].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(to_batch(data.images, chunk, device))
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(chunk)
            correct += int((logits.detach().argmax(1) == labels).sum())
            seen += len(chunk)
            if number % 10 == 0 or number == batches:
                progress(dict(kind="batch", epoch=epoch, batch=number, batches=batches, loss=loss.item()))
        if stopped:
            break
        validation_loss, validation_accuracy, _ = evaluate(model, data, validation_ids, 512, device)
        row = dict(epoch=epoch, train_loss=loss_sum / seen, train_accuracy=correct / seen,
                   validation_loss=validation_loss, validation_accuracy=validation_accuracy,
                   seconds=time.perf_counter() - epoch_start)
        history.append(row)
        key = (validation_accuracy, -validation_loss)
        if best is None or key > best["key"]:
            best = dict(key=key, epoch=epoch)
            saved = {"format_version": 1, "model_name": settings.model, "state_dict": model.state_dict(),
                     "epoch": epoch, "validation_accuracy": validation_accuracy,
                     "validation_loss": validation_loss, "config": config}
            if settings.task == "emnist":
                saved.update(task="emnist_byclass_62", classes=data.classes)
            torch.save(saved, checkpoint)
        progress(dict(kind="epoch", **row))
    record = dict(config=config, history=history, stopped=stopped,
                  training_seconds=time.perf_counter() - started, checkpoint=None)
    if best is not None:
        # Confusion matrix of the selected checkpoint on the validation subset.
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"])
        _, accuracy, predicted = evaluate(model, data, validation_ids, 512, device)
        size = len(data.classes)
        confusion = np.zeros((size, size), dtype=np.int64)
        np.add.at(confusion, (data.targets[validation_ids].numpy(), predicted.numpy()), 1)
        record.update(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), best_epoch=best["epoch"],
                      validation_accuracy=accuracy, confusion_matrix=confusion.tolist(), classes=data.classes)
    if history:
        with (output / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(history)
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    progress(dict(kind="done", **{k: record.get(k) for k in ("stopped", "best_epoch", "validation_accuracy")}))
    return record
