"""
Generate training curve plots from per-model CSV history files.

Reads results/{model}_history.csv files and produces:
  1. Val Dice curves (all models overlaid)
  2. Train vs Val loss per model (overfitting diagnostic)
  3. Test set bar chart comparison

Usage:
    python src/plot_training.py
    python src/plot_training.py --outdir results/figures
"""

import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

MODELS = ["cnn", "unet", "attention_unet"]

DISPLAY_NAMES = {
    "cnn": "CNN Baseline",
    "unet": "U-Net",
    "attention_unet": "Attention U-Net",
}

COLORS = {
    "cnn": "#e74c3c",
    "unet": "#2ecc71",
    "attention_unet": "#3498db",
}


def load_history(model_name):
    """Load training history from CSV. Returns dict with lists."""
    csv_path = os.path.join(RESULTS_DIR, f"{model_name}_history.csv")
    data = {"epoch": [], "train_loss": [], "val_loss": [], "val_dice": []}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["epoch"].append(int(row["epoch"]))
            data["train_loss"].append(float(row["train_loss"]))
            data["val_loss"].append(float(row["val_loss"]))
            data["val_dice"].append(float(row["val_dice"]))

    return data


def plot_val_dice(all_data, outdir):
    """Val Dice over epochs for all models."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, data in all_data.items():
        ax.plot(data["epoch"], data["val_dice"],
                label=DISPLAY_NAMES[name],
                color=COLORS[name],
                linewidth=2, marker="o", markersize=3)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Dice", fontsize=12)
    ax.set_title("Validation Dice Score During Training", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0.4, 0.9)

    plt.tight_layout()
    path = os.path.join(outdir, "val_dice_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_loss_curves(all_data, outdir):
    """Train and val loss over epochs for each model (subplots)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, name in zip(axes, MODELS):
        data = all_data[name]
        ax.plot(data["epoch"], data["train_loss"],
                label="Train Loss", linewidth=2, color=COLORS[name])
        ax.plot(data["epoch"], data["val_loss"],
                label="Val Loss", linewidth=2, color=COLORS[name],
                linestyle="--", alpha=0.7)

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_title(DISPLAY_NAMES[name], fontsize=13)
        ax.legend(fontsize=10)

    axes[0].set_ylabel("BCE Loss", fontsize=11)
    fig.suptitle("Train vs Validation Loss (Overfitting Diagnostic)", fontsize=14, y=1.02)

    plt.tight_layout()
    path = os.path.join(outdir, "loss_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_combined_loss(all_data, outdir):
    """All models' train loss on one plot to compare convergence speed."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, data in all_data.items():
        ax.plot(data["epoch"], data["train_loss"],
                label=DISPLAY_NAMES[name],
                color=COLORS[name],
                linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Train Loss (BCE)", fontsize=12)
    ax.set_title("Training Loss Convergence", fontsize=14)
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(outdir, "train_loss_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_summary_bar(outdir):
    """Bar chart comparing final test metrics (loads from test_metrics.npy if available)."""
    metrics_path = os.path.join(RESULTS_DIR, "test_metrics.npy")
    if os.path.exists(metrics_path):
        results = np.load(metrics_path, allow_pickle=True).item()
        models = list(results.keys())
        dice = [results[m]["dice"] for m in models]
        iou = [results[m]["iou"] for m in models]
        display = [DISPLAY_NAMES.get(m, m.title()) for m in models]
    else:
        # Fallback hardcoded from previous eval run
        display = ["Classical", "CNN", "U-Net", "Attention\nU-Net"]
        dice = [0.5447, 0.6799, 0.8303, 0.8263]
        iou = [0.4093, 0.5445, 0.7258, 0.7203]

    x = np.arange(len(display))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, dice, width, label="Dice", color="#2ecc71", alpha=0.85)
    bars2 = ax.bar(x + width / 2, iou, width, label="IoU", color="#3498db", alpha=0.85)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Test Set Performance Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(display, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(outdir, "test_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load CSV histories
    all_data = {}
    for name in MODELS:
        csv_path = os.path.join(RESULTS_DIR, f"{name}_history.csv")
        if os.path.exists(csv_path):
            all_data[name] = load_history(name)
            print(f"Loaded {name}: {len(all_data[name]['epoch'])} epochs")
        else:
            print(f"WARNING: {csv_path} not found, skipping {name}")

    if not all_data:
        print("No history CSVs found. Run training first.")
        return

    plot_val_dice(all_data, args.outdir)
    plot_loss_curves(all_data, args.outdir)
    plot_combined_loss(all_data, args.outdir)
    plot_summary_bar(args.outdir)

    print(f"\nDone! All plots saved to: {args.outdir}")


if __name__ == "__main__":
    main()
