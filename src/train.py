"""
Training script for nuclei segmentation models.

Trains CNN baseline, U-Net, or Attention U-Net on PanNuke binary masks.
Uses validation set to pick the best checkpoint. Test set is never seen during training.
Saves every epoch's checkpoint and a CSV history.

Usage:
    python src/train.py --model unet
    python src/train.py --model cnn
    python src/train.py --model attention_unet
"""

import csv
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

# Fixed hyperparameters (same for all models)
EPOCHS = 25
LR = 1e-4
BATCH_SIZE = 32
NUM_WORKERS = 8


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
    total_precision = 0.0
    total_recall = 0.0
    total_cell_error = 0.0
    n_batches = 0
    smooth = 1e-5

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        preds = (torch.sigmoid(logits) > 0.5).float()

        total_loss += loss.item() * images.size(0)
        total_dice += dice_score(preds, masks).item()

        # Precision and recall
        pred_flat = preds.view(preds.size(0), -1)
        target_flat = masks.view(masks.size(0), -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        precision = ((intersection + smooth) / (pred_flat.sum(dim=1) + smooth)).mean()
        recall = ((intersection + smooth) / (target_flat.sum(dim=1) + smooth)).mean()
        total_precision += precision.item()
        total_recall += recall.item()

        # Cell count MAE via connected components on CPU
        pred_np = preds.cpu().numpy().squeeze(1).astype(np.uint8)
        mask_np = masks.cpu().numpy().squeeze(1).astype(np.uint8)
        from scipy import ndimage
        for p, m in zip(pred_np, mask_np):
            pred_count = ndimage.label(p)[1]
            true_count = ndimage.label(m)[1]
            total_cell_error += abs(pred_count - true_count)

        n_batches += 1

    n_samples = len(loader.dataset)
    avg_loss = total_loss / n_samples
    avg_dice = total_dice / n_batches
    avg_precision = total_precision / n_batches
    avg_recall = total_recall / n_batches
    avg_cell_mae = total_cell_error / n_samples
    return avg_loss, avg_dice, avg_precision, avg_recall, avg_cell_mae


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train nuclei segmentation model")
    parser.add_argument("--model", type=str, default="unet", choices=["cnn", "unet", "attention_unet"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.model}] Device: {device}")
    print(f"[{args.model}] LR: {LR}, Batch: {BATCH_SIZE}, Epochs: {EPOCHS}")

    # Data
    train_dataset = NucleiDataset("train")
    val_dataset = NucleiDataset("val")
    print(f"[{args.model}] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Model, loss, optimizer
    model = get_model(args.model, device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{args.model}] Parameters: {param_count:,}")

    # Setup output directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ckpt_dir = os.path.join(RESULTS_DIR, f"{args.model}_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    csv_path = os.path.join(RESULTS_DIR, f"{args.model}_history.csv")

    best_dice = 0.0
    best_epoch = 0

    # CSV logger
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_precision", "val_recall", "val_cell_mae", "time_s"])

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_prec, val_rec, val_cell_mae = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        print(f"[{args.model}] Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_dice:.4f} | "
              f"Prec: {val_prec:.4f} | "
              f"Rec: {val_rec:.4f} | "
              f"Cell MAE: {val_cell_mae:.2f} | "
              f"Time: {elapsed:.1f}s")

        csv_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_dice:.6f}",
                             f"{val_prec:.6f}", f"{val_rec:.6f}", f"{val_cell_mae:.4f}", f"{elapsed:.1f}"])
        csv_file.flush()

        # Save every epoch's checkpoint
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch:02d}.pth"))

        # Track best
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch

    csv_file.close()

    # Symlink/copy best checkpoint for easy eval access
    best_src = os.path.join(ckpt_dir, f"epoch_{best_epoch:02d}.pth")
    best_dst = os.path.join(RESULTS_DIR, f"{args.model}_best.pth")
    if os.path.exists(best_dst):
        os.remove(best_dst)
    os.symlink(os.path.abspath(best_src), best_dst)

    print(f"\n[{args.model}] Best val Dice: {best_dice:.4f} at epoch {best_epoch}")
    print(f"[{args.model}] Best checkpoint: {best_src}")
    print(f"[{args.model}] Symlinked to: {best_dst}")
    print(f"[{args.model}] All checkpoints: {ckpt_dir}/")
    print(f"[{args.model}] History: {csv_path}")


if __name__ == "__main__":
    main()
