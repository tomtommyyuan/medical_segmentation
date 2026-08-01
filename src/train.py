"""
Train a binary-mask baseline on one split of the official PanNuke protocol.

Trains the classical-era baselines (simple CNN, U-Net, Attention U-Net) that
CHROMA-Net is measured against. They predict a binary nuclei mask only, so they
are scored with Dice and IoU here and with connected-component bPQ in
evaluate_instance.py, where the cost of having no instance or class output
shows up directly.

These are kept as a fair ladder rather than a straw man: they get the same
folds, the same dihedral and stain augmentation, the same cosine schedule and
the same early-stopping-free budget as CHROMA-Net. Use --no-augment to
reproduce the original no-augmentation runs.

Run all three splits so the numbers are comparable to everything else:
    python src/train.py --model unet --split 1
    python src/train.py --model unet --split 2
    python src/train.py --model unet --split 3

Usage:
    python src/train.py --model unet --split 1
    python src/train.py --model cnn --split 1 --no-augment
    python src/train.py --model attention_unet --split 1
"""

import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from attention_unet import AttentionUNet
from cnn_baseline import CNNBaseline
from dataset import SPLITS, BinaryDataset
from unet import UNet

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

EPOCHS = 50
LR = 1e-4
BATCH_SIZE = 32
NUM_WORKERS = 8
WARMUP_EPOCHS = 2

MODELS = {"cnn": CNNBaseline, "unet": UNet, "attention_unet": AttentionUNet}


def set_seed(seed):
    """Seed every generator that affects a run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_schedule(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay, stepped per optimizer step."""

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += float(loss.detach()) * images.size(0)
        n_samples += images.size(0)

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Validation loss and aggregate segmentation metrics.

    Dice, IoU, precision and recall are accumulated over the whole fold and
    divided once at the end, rather than averaged per patch. The previous
    version averaged a per-patch Dice with a 1e-5 smoothing term, which handed
    every patch containing no nuclei a free 1.0 and inflated the score by an
    amount that depended on how many empty patches a fold happened to hold.
    """
    model.eval()

    total_loss = 0.0
    n_samples = 0
    intersection = 0.0
    pred_total = 0.0
    target_total = 0.0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        preds = (torch.sigmoid(logits) > 0.5).float()

        total_loss += float(loss) * images.size(0)
        n_samples += images.size(0)

        intersection += float((preds * masks).sum())
        pred_total += float(preds.sum())
        target_total += float(masks.sum())

    dice = 2.0 * intersection / max(pred_total + target_total, 1.0)
    iou = intersection / max(pred_total + target_total - intersection, 1.0)
    precision = intersection / max(pred_total, 1.0)
    recall = intersection / max(target_total, 1.0)

    return {
        "loss": total_loss / max(n_samples, 1),
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


def main():
    parser = argparse.ArgumentParser(description="Train a binary nuclei baseline")
    parser.add_argument("--model", type=str, default="unet", choices=sorted(MODELS))
    parser.add_argument("--split", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--no-augment", action="store_true",
                        help="reproduce the original runs, which had no augmentation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--out-dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    set_seed(args.seed)

    run = f"{args.model}_split{args.split}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folds = SPLITS[args.split]
    print(f"[{run}] Device: {device}")
    print(f"[{run}] Folds: train={folds['train']} val={folds['val']} test={folds['test']}")
    print(f"[{run}] LR: {args.lr}, Batch: {args.batch_size}, Epochs: {args.epochs}, "
          f"Augment: {not args.no_augment}")

    train_dataset = BinaryDataset(folds["train"], data_dir=args.data_dir,
                                  augment=not args.no_augment, seed=args.seed)
    val_dataset = BinaryDataset(folds["val"], data_dir=args.data_dir, seed=args.seed)
    print(f"[{run}] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0)

    model = MODELS[args.model]().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps_per_epoch = len(train_loader)
    scheduler = cosine_schedule(optimizer, args.warmup_epochs * steps_per_epoch,
                                args.epochs * steps_per_epoch)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{run}] Parameters: {param_count:,}")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, f"{run}_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, f"{run}_config.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2)

    csv_path = os.path.join(args.out_dir, f"{run}_history.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou",
                         "val_precision", "val_recall", "lr", "time_s"])

    # Starts below zero so the first epoch always writes a checkpoint. Starting
    # at 0.0 leaves a run whose Dice never rises above zero with no checkpoint
    # at all, and evaluation then fails on a missing file rather than on a bad
    # score.
    best_dice = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     scheduler, device)
        metrics = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"[{run}] Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {metrics['loss']:.4f} | "
              f"Val Dice: {metrics['dice']:.4f} | "
              f"IoU: {metrics['iou']:.4f} | "
              f"Prec: {metrics['precision']:.4f} | "
              f"Rec: {metrics['recall']:.4f} | "
              f"Time: {elapsed:.1f}s")

        csv_writer.writerow([epoch, f"{train_loss:.6f}", f"{metrics['loss']:.6f}",
                             f"{metrics['dice']:.6f}", f"{metrics['iou']:.6f}",
                             f"{metrics['precision']:.6f}", f"{metrics['recall']:.6f}",
                             f"{current_lr:.3e}", f"{elapsed:.1f}"])
        csv_file.flush()

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"{run}_best.pth"))

        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))

    csv_file.close()

    print(f"\n[{run}] Best val Dice: {best_dice:.4f} at epoch {best_epoch}")
    print(f"[{run}] Best checkpoint: {os.path.join(args.out_dir, f'{run}_best.pth')}")
    print(f"[{run}] History: {csv_path}")


if __name__ == "__main__":
    main()
