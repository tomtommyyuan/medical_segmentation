"""
Evaluate Dice score for all four methods broken down by tissue type.

Picks the 5 most common tissue types in the test set for statistical
reliability, computes per-sample Dice, and reports mean ± std per group.
Also saves a grouped bar chart to figures/.

Usage:
    python src/evaluate_by_tissue.py
"""

import os
import sys

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(__file__))

from cnn_baseline import CNNBaseline
from unet import UNet
from attention_unet import AttentionUNet
from classical import segment_single

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

METHOD_NAMES = ["Classical", "CNN", "U-Net", "Att. U-Net"]
METHOD_KEYS = ["classical", "cnn", "unet", "attention_unet"]
NUM_TISSUE_TYPES = 5


def dice_per_sample(preds, targets):
    """Compute per-sample Dice scores."""
    smooth = 1e-5
    p = preds.reshape(preds.shape[0], -1).astype(np.float64)
    t = targets.reshape(targets.shape[0], -1).astype(np.float64)
    intersection = (p * t).sum(axis=1)
    return (2.0 * intersection + smooth) / (p.sum(axis=1) + t.sum(axis=1) + smooth)


def predict_neural(model_name, images, device):
    """Run batch inference with a trained neural model."""
    model_map = {"cnn": CNNBaseline, "unet": UNet, "attention_unet": AttentionUNet}
    model = model_map[model_name]()

    ckpt = os.path.join(RESULTS_DIR, f"{model_name}_best.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()

    batch_size = 32
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size].astype(np.float32) / 255.0
            batch = np.transpose(batch, (0, 3, 1, 2))
            logits = model(torch.from_numpy(batch).to(device))
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().squeeze(1).astype(np.uint8)
            all_preds.append(preds)

    return np.concatenate(all_preds, axis=0)


def predict_classical(images):
    """Run classical pipeline on all images."""
    preds = np.zeros((len(images), images.shape[1], images.shape[2]), dtype=np.uint8)
    for i in range(len(images)):
        preds[i] = segment_single(images[i])
    return preds


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading test data...")
    images = np.load(os.path.join(DATA_DIR, "test_images.npy"))
    masks = np.load(os.path.join(DATA_DIR, "test_masks.npy"))
    types = np.load(os.path.join(DATA_DIR, "test_types.npy"), allow_pickle=True)
    print(f"Test samples: {len(images)}")

    # Find the most common tissue types
    unique_types, counts = np.unique(types, return_counts=True)
    sorted_idx = np.argsort(-counts)
    top_types = unique_types[sorted_idx[:NUM_TISSUE_TYPES]]
    top_counts = counts[sorted_idx[:NUM_TISSUE_TYPES]]

    print(f"\nTop {NUM_TISSUE_TYPES} tissue types in test set:")
    for t, c in zip(top_types, top_counts):
        print(f"  {t}: {c} samples")

    # Generate predictions for each method
    print("\nGenerating predictions...")
    all_preds = {}
    for key in METHOD_KEYS:
        print(f"  {key}...")
        if key == "classical":
            all_preds[key] = predict_classical(images)
        else:
            all_preds[key] = predict_neural(key, images, device)

    # Compute per-sample Dice for each method
    all_dice = {}
    for key in METHOD_KEYS:
        all_dice[key] = dice_per_sample(all_preds[key], masks)

    # Build results table: tissue type x method
    print(f"\n{'='*90}")
    print(f"  DICE SCORE BY TISSUE TYPE")
    print(f"{'='*90}")
    header = f"  {'Tissue Type':<16} {'N':>5}"
    for name in METHOD_NAMES:
        header += f"  {name:>16}"
    print(header)
    print(f"  {'-'*16} {'-'*5}" + f"  {'-'*16}" * len(METHOD_NAMES))

    # Store for plotting
    table_means = {key: [] for key in METHOD_KEYS}
    table_stds = {key: [] for key in METHOD_KEYS}
    tissue_labels = []

    for tissue in top_types:
        idx = types == tissue
        n = idx.sum()
        tissue_labels.append(tissue)

        row = f"  {tissue:<16} {n:>5}"
        for key, name in zip(METHOD_KEYS, METHOD_NAMES):
            d = all_dice[key][idx]
            mean, std = d.mean(), d.std()
            table_means[key].append(mean)
            table_stds[key].append(std)
            row += f"  {mean:>6.3f} ± {std:.3f}"
        print(row)

    # Overall row
    print(f"  {'-'*16} {'-'*5}" + f"  {'-'*16}" * len(METHOD_NAMES))
    row = f"  {'Overall':<16} {len(images):>5}"
    for key in METHOD_KEYS:
        d = all_dice[key]
        row += f"  {d.mean():>6.3f} ± {d.std():.3f}"
    print(row)

    # Save results as .npy
    results = {
        "tissue_types": tissue_labels,
        "methods": METHOD_KEYS,
        "method_names": METHOD_NAMES,
        "means": {k: np.array(v) for k, v in table_means.items()},
        "stds": {k: np.array(v) for k, v in table_stds.items()},
    }
    results_path = os.path.join(RESULTS_DIR, "tissue_type_dice.npy")
    np.save(results_path, results)
    print(f"\nResults saved to: {results_path}")

    # Grouped bar chart
    x = np.arange(len(tissue_labels))
    width = 0.18
    colors = ["#ff6b6b", "#ffa94d", "#51cf66", "#339af0"]

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (key, name, color) in enumerate(zip(METHOD_KEYS, METHOD_NAMES, colors)):
        means = table_means[key]
        stds = table_stds[key]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, label=name,
                      color=color, edgecolor="white", linewidth=0.5,
                      capsize=3, error_kw={"linewidth": 1})

    ax.set_xlabel("Tissue Type", fontsize=12)
    ax.set_ylabel("Dice Score", fontsize=12)
    ax.set_title("Dice Score by Tissue Type Across Methods", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tissue_labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "dice_by_tissue.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved to: {fig_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
