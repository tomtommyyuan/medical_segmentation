"""
Training script for nuclei segmentation models.

Trains CNN baseline, U-Net, or Attention U-Net on PanNuke binary masks.
Uses validation set to pick the best checkpoint. Test set is never seen during training.

Usage:
    python src/train.py --model unet --epochs 50 --lr 1e-4 --batch_size 16
    python src/train.py --model cnn
    python src/train.py --model attention_unet
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from cnn_baseline import CNNBaseline
from unet import UNet
from attention_unet import AttentionUNet

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


class NucleiDataset(Dataset):
    def __init__(self, split="train"):
        self.images = np.load(os.path.join(DATA_DIR, f"{split}_images.npy"))
        self.masks = np.load(os.path.join(DATA_DIR, f"{split}_masks.npy"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0
        mask = self.masks[idx].astype(np.float32)

        # (H, W, C) -> (C, H, W)
        image = np.transpose(image, (2, 0, 1))

        return torch.from_numpy(image), torch.from_numpy(mask).unsqueeze(0)


def dice_score(pred, target, smooth=1e-5):
    """Compute Dice coefficient for a batch of predictions."""
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    return ((2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)).mean()


def get_model(model_name, device):
    if model_name == "cnn":
        model = CNNBaseline()
    elif model_name == "unet":
        model = UNet()
    elif model_name == "attention_unet":
        model = AttentionUNet()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        preds = (torch.sigmoid(logits) > 0.5).float()

        total_loss += loss.item() * images.size(0)
        total_dice += dice_score(preds, masks).item()
        n_batches += 1

    avg_loss = total_loss / len(loader.dataset)
    avg_dice = total_dice / n_batches
    return avg_loss, avg_dice


def main():
    parser = argparse.ArgumentParser(description="Train nuclei segmentation model")
    parser.add_argument("--model", type=str, default="unet", choices=["cnn", "unet", "attention_unet"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model}, LR: {args.lr}, Batch: {args.batch_size}, Epochs: {args.epochs}")

    # Data
    train_dataset = NucleiDataset("train")
    val_dataset = NucleiDataset("val")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model, loss, optimizer
    model = get_model(args.model, device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {param_count:,}")

    # Training loop
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, f"{args.model}_best.pth")

    best_dice = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_dice:.4f} | "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    print(f"\nBest val Dice: {best_dice:.4f} at epoch {best_epoch}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
