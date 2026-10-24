"""Train MLP/CNN using identical development data; never load the final test set."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch import nn
import torchvision

from .data import load_partition, development_split, indices_hash
from .models import build_model


@torch.inference_mode()
def score(model, images, targets, indices, batch_size):
    model.eval()
    total_loss, correct = 0.0, 0
    for selected in indices.split(batch_size):
        logits = model(images[selected])
        total_loss += nn.functional.cross_entropy(logits, targets[selected], reduction="sum").item()
        correct += (logits.argmax(1) == targets[selected]).sum().item()
    return total_loss / len(indices), correct / len(indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/baseline"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) <= 0 or args.learning_rate <= 0:
        parser.error("Epochs, batch size, threads and learning rate must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; choose a new experiment directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    images, targets = load_partition(args.data_dir, train=True, download=args.download)
    train_ids, val_ids = development_split(len(targets), args.seed)
    config = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
              "optimizer": "Adam", "learning_rate": args.learning_rate,
              "weight_decay": 0, "loss": "cross_entropy", "augmentation": "none",
              "normalisation": "uint8 / 255", "device": "cpu", "threads": args.threads,
              "train_count": len(train_ids), "validation_count": len(val_ids),
              "train_indices_sha256": indices_hash(train_ids),
              "validation_indices_sha256": indices_hash(val_ids),
              "python": platform.python_version(), "platform": platform.platform(),
              "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
              "selection_rule": "Highest validation accuracy, then lowest validation loss; earlier epoch retained on exact tie",
              "test_data_used": False}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    np.savez_compressed(args.output_dir / "split_indices.npz", train=train_ids.numpy(), validation=val_ids.numpy())
    histories, summaries = [], []
    for name in ("mlp", "cnn"):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = build_model(name)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        # A dedicated generator gives both models the exact same minibatch order.
        generator = torch.Generator().manual_seed(args.seed + 1)
        best_key = (-1.0, float("-inf"))
        started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            loss_sum, correct = 0.0, 0
            shuffled = train_ids[torch.randperm(len(train_ids), generator=generator)]
            for selected in shuffled.split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                logits = model(images[selected])
                loss = nn.functional.cross_entropy(logits, targets[selected])
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(selected)
                correct += (logits.detach().argmax(1) == targets[selected]).sum().item()
            val_loss, val_accuracy = score(model, images, targets, val_ids, args.batch_size)
            row = {"model": name, "epoch": epoch, "train_loss": loss_sum / len(train_ids),
                   "train_accuracy": correct / len(train_ids), "validation_loss": val_loss,
                   "validation_accuracy": val_accuracy, "seconds": time.perf_counter() - epoch_start}
            histories.append(row)
            key = (val_accuracy, -val_loss)
            if key > best_key:
                best_key, best_epoch = key, epoch
                torch.save({"format_version": 1, "model_name": name, "state_dict": model.state_dict(),
                            "epoch": epoch, "validation_accuracy": val_accuracy,
                            "validation_loss": val_loss, "config": config}, args.output_dir / f"{name}.pt")
            print(json.dumps(row), flush=True)
            with (args.output_dir / "history.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(histories[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(histories)
        checkpoint = args.output_dir / f"{name}.pt"
        summaries.append({"model": name, "best_epoch": best_epoch, "validation_accuracy": best_key[0],
                          "validation_loss": -best_key[1], "training_seconds": time.perf_counter() - started,
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    selected = max(summaries, key=lambda x: (x["validation_accuracy"], -x["validation_loss"]))
    selection = {"selected_model": selected["model"], "selected_before_test_evaluation": True,
                 "models": summaries, "config": config}
    (args.output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"Selected {selected['model']} using validation only.", flush=True)


if __name__ == "__main__":
    main()
