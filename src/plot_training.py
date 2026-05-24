"""
Plot training curves for CNN and U-Net from CSV history files.

Generates a 2x2 figure with:
  - Top-left: Train & Val Loss
  - Top-right: Val Dice
  - Bottom-left: Val Precision
  - Bottom-right: Val Recall

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

MODELS = ["cnn", "unet"]

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
    """Load training history from CSV."""
    csv_path = os.path.join(RESULTS_DIR, f"{model_name}_history.csv")
    data = {"epoch": [], "train_loss": [], "val_loss": [], "val_dice": [],
            "val_precision": [], "val_recall": [], "val_cell_mae": []}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["epoch"].append(int(row["epoch"]))
            data["train_loss"].append(float(row["train_loss"]))
            data["val_loss"].append(float(row["val_loss"]))
            data["val_dice"].append(float(row["val_dice"]))
            data["val_precision"].append(float(row["val_precision"]))
            data["val_recall"].append(float(row["val_recall"]))
            data["val_cell_mae"].append(float(row["val_cell_mae"]))

    return data


def plot_training_curves(all_data, outdir):
    """2x2 subplot: loss, dice, precision, recall."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Top-left: Loss
    ax = axes[0, 0]
    for name, data in all_data.items():
        ax.plot(data["epoch"], data["train_loss"],
                label=f"{DISPLAY_NAMES[name]} (train)", color=COLORS[name], linewidth=2)
        ax.plot(data["epoch"], data["val_loss"],
                label=f"{DISPLAY_NAMES[name]} (val)", color=COLORS[name], linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(fontsize=9)

    # Top-right: Dice
    ax = axes[0, 1]
    for name, data in all_data.items():
        ax.plot(data["epoch"], data["val_dice"],
                label=DISPLAY_NAMES[name], color=COLORS[name], linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice Score")
    ax.set_title("Validation Dice")
    ax.legend(fontsize=10)
    ax.set_ylim(0.5, 0.9)

    # Bottom-left: Precision
    ax = axes[1, 0]
    for name, data in all_data.items():
        ax.plot(data["epoch"], data["val_precision"],
                label=DISPLAY_NAMES[name], color=COLORS[name], linewidth=2, marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Precision")
    ax.set_title("Validation Precision")
    ax.legend(fontsize=10)
    ax.set_ylim(0.6, 0.9)

    # Bottom-right: Recall
    ax = axes[1, 1]
    for name, data in all_data.items():
        ax.plot(data["epoch"], data["val_recall"],
                label=DISPLAY_NAMES[name], color=COLORS[name], linewidth=2, marker="^", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall")
    ax.set_title("Validation Recall")
    ax.legend(fontsize=10)
    ax.set_ylim(0.5, 1.0)

    fig.suptitle("Training Curves: CNN Baseline vs U-Net", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(outdir, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_cell_mae(all_data, outdir):
    """Separate plot for cell count MAE over epochs."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, data in all_data.items():
        ax.plot(data["epoch"], data["val_cell_mae"],
                label=DISPLAY_NAMES[name], color=COLORS[name], linewidth=2, marker="D", markersize=4)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cell Count MAE", fontsize=12)
    ax.set_title("Validation Cell Count Error Over Training", fontsize=14)
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(outdir, "cell_mae_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_summary_bar(outdir):
    """Bar chart comparing final test metrics."""
    metrics_path = os.path.join(RESULTS_DIR, "test_metrics.npy")
    if os.path.exists(metrics_path):
        results = np.load(metrics_path, allow_pickle=True).item()
        models = list(results.keys())
        dice = [results[m]["dice"] for m in models]
        iou = [results[m]["iou"] for m in models]
        display = [DISPLAY_NAMES.get(m, m.title()) for m in models]
    else:
        display = ["Classical", "CNN", "U-Net", "Attn U-Net"]
        dice = [0.5447, 0.6953, 0.8281, 0.8280]
        iou = [0.4093, 0.5628, 0.7220, 0.7229]

    x = np.arange(len(display))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
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

    plot_training_curves(all_data, args.outdir)
    plot_cell_mae(all_data, args.outdir)
    plot_summary_bar(args.outdir)

    print(f"\nDone! All plots saved to: {args.outdir}")


if __name__ == "__main__":
    main()
