"""
Generate qualitative comparison figure for the paper.

Picks representative test samples (good, moderate, hard) and plots
input image, ground truth, and predictions from all four methods
side by side with per-sample Dice scores.

Usage:
    python src/visualize_qualitative.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

import torch
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))

from cnn_baseline import CNNBaseline
from unet import UNet
from attention_unet import AttentionUNet
from classical import segment_single

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def dice_single(pred, target):
    smooth = 1e-5
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def predict_single_neural(model, image, device):
    """Run a single image through a neural model."""
    batch = image.astype(np.float32) / 255.0
    batch = np.transpose(batch, (2, 0, 1))[np.newaxis]
    with torch.no_grad():
        logits = model(torch.from_numpy(batch).to(device))
        pred = (torch.sigmoid(logits) > 0.5).cpu().numpy().squeeze().astype(np.uint8)
    return pred


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading test data...")
    images = np.load(os.path.join(DATA_DIR, "test_images.npy"))
    masks = np.load(os.path.join(DATA_DIR, "test_masks.npy"))
    types = np.load(os.path.join(DATA_DIR, "test_types.npy"), allow_pickle=True)
    print(f"Test samples: {len(images)}")

    # Load neural models
    models = {}
    for name, cls in [("cnn", CNNBaseline), ("unet", UNet), ("attention_unet", AttentionUNet)]:
        m = cls()
        ckpt = os.path.join(RESULTS_DIR, f"{name}_best.pth")
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        m.to(device).eval()
        models[name] = m

    # Compute per-sample U-Net Dice to pick representative samples
    print("Computing per-sample Dice for U-Net...")
    unet_dices = []
    for i in range(len(images)):
        pred = predict_single_neural(models["unet"], images[i], device)
        unet_dices.append(dice_single(pred.astype(np.float64), masks[i].astype(np.float64)))
    unet_dices = np.array(unet_dices)

    # Pick samples: high Dice (good), near-median (moderate), low Dice (hard)
    # Filter out empty masks for the "good" case
    nonempty = masks.reshape(len(masks), -1).sum(axis=1) > 100
    valid_dices = np.where(nonempty, unet_dices, -1)

    sorted_idx = np.argsort(valid_dices)
    good_idx = sorted_idx[-10]  # high Dice, offset a bit from absolute max
    median_pos = len(sorted_idx) // 2
    moderate_idx = sorted_idx[median_pos]

    # For hard case, pick from bottom quartile but skip near-empty masks
    nonempty_low = np.where(nonempty)[0]
    nonempty_low_dices = unet_dices[nonempty_low]
    hard_candidates = nonempty_low[np.argsort(nonempty_low_dices)]
    hard_idx = hard_candidates[5]  # a few from the bottom

    sample_indices = [good_idx, moderate_idx, hard_idx]
    sample_labels = ["Good", "Moderate", "Challenging"]
    print(f"Selected indices: {sample_indices}")
    for i, (idx, label) in enumerate(zip(sample_indices, sample_labels)):
        print(f"  {label}: idx={idx}, tissue={types[idx]}, U-Net Dice={unet_dices[idx]:.4f}")

    # Generate predictions for selected samples
    method_names = ["Classical", "CNN", "U-Net", "Att. U-Net"]
    method_keys = ["classical", "cnn", "unet", "attention_unet"]

    fig, axes = plt.subplots(
        len(sample_indices), 6,
        figsize=(18, 3 * len(sample_indices)),
        gridspec_kw={"wspace": 0.05, "hspace": 0.25},
    )

    for row, (idx, label) in enumerate(zip(sample_indices, sample_labels)):
        image = images[idx]
        gt = masks[idx]

        # Input image
        axes[row, 0].imshow(image)
        axes[row, 0].set_title("Input" if row == 0 else "", fontsize=11)
        axes[row, 0].set_ylabel(f"{label}\n({types[idx]})", fontsize=10, rotation=0, labelpad=70, va="center")

        # Ground truth
        axes[row, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground Truth" if row == 0 else "", fontsize=11)

        # Each method's prediction
        for col, (mname, mkey) in enumerate(zip(method_names, method_keys)):
            if mkey == "classical":
                pred = segment_single(image)
            else:
                pred = predict_single_neural(models[mkey], image, device)

            d = dice_single(pred.astype(np.float64), gt.astype(np.float64))

            axes[row, col + 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
            title = f"{mname}" if row == 0 else ""
            axes[row, col + 2].set_title(title, fontsize=11)
            axes[row, col + 2].text(
                128, 245, f"Dice: {d:.3f}",
                ha="center", va="bottom", fontsize=9,
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
            )

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Qualitative Comparison on Test Samples", fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "qualitative_comparison.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved to: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
