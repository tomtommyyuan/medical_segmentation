"""
Plot training curves and the final benchmark comparison.

Deliberately imports nothing heavier than matplotlib, so curves can be plotted
on a login node while jobs are still running on the GPU nodes.

Three panels, one metric each. The baselines are selected on Dice and
CHROMA-Net on mPQ, and those do not belong on shared axes, so they get separate
panels rather than a second y-axis.

Usage:
    python src/plot_training.py
    python src/plot_training.py --splits 1 --figures-dir figures
"""

import argparse
import csv
import glob
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

BASELINES = ["cnn", "unet", "attention_unet"]

DISPLAY_NAMES = {
    "classical": "Classical",
    "cnn": "CNN",
    "unet": "U-Net",
    "attention_unet": "Att. U-Net",
    "chroma": "CHROMA-Net",
}

# Okabe-Ito hues in fixed order, checked for colour-vision deficiency
# separation. Kept in step with evaluate_by_tissue.py; a method keeps its
# colour across every figure in the repo.
METHOD_COLORS = {
    "classical": "#0072B2",
    "cnn": "#D55E00",
    "unet": "#009E73",
    "attention_unet": "#E69F00",
    "chroma": "#CC79A7",
}


def load_history(path):
    """Read a training history CSV into a dict of float columns."""
    with open(path) as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return {}

    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def find_histories(results_dir, splits):
    """Locate every history CSV, keyed by (method, split)."""
    histories = {}

    for path in sorted(glob.glob(os.path.join(results_dir, "*_history.csv"))):
        name = os.path.basename(path)[: -len("_history.csv")]
        if "_split" not in name:
            continue

        method, _, split_text = name.rpartition("_split")
        if not split_text.isdigit() or int(split_text) not in splits:
            continue

        histories[(method, int(split_text))] = load_history(path)

    return histories


def plot_curves(histories, out_path):
    """Loss, baseline Dice, and CHROMA-Net mPQ, one metric per panel."""
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    for (method, split), data in sorted(histories.items()):
        if not data:
            continue

        color = METHOD_COLORS.get(method, "#888888")
        label = f"{DISPLAY_NAMES.get(method, method)} s{split}"

        axes[0].plot(data["epoch"], data["train_loss"], color=color, linewidth=1.6,
                     alpha=0.85, label=f"{label} train")
        axes[0].plot(data["epoch"], data["val_loss"], color=color, linewidth=1.6,
                     alpha=0.85, linestyle="--", label=f"{label} val")

        if "val_dice" in data:
            axes[1].plot(data["epoch"], data["val_dice"], color=color, linewidth=1.8,
                         label=label)
        if "val_mpq" in data:
            axes[2].plot(data["epoch"], data["val_mpq"], color=color, linewidth=1.8,
                         label=f"{label} mPQ")
            axes[2].plot(data["epoch"], data["val_bpq"], color=color, linewidth=1.8,
                         linestyle="--", alpha=0.7, label=f"{label} bPQ")

    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and validation loss", fontsize=12, fontweight="bold")

    axes[1].set_ylabel("Dice")
    axes[1].set_title("Baseline validation Dice", fontsize=12, fontweight="bold")

    axes[2].set_ylabel("Panoptic quality")
    axes[2].set_title("CHROMA-Net validation PQ", fontsize=12, fontweight="bold")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, frameon=False, ncol=1)

    figure.tight_layout()
    figure.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {out_path}")


def plot_benchmark(results_dir, out_path):
    """
    Final mPQ and bPQ per method, against the published results.

    Reads whatever evaluate_instance.py has written, so the figure only ever
    shows numbers that were actually measured.
    """
    measured = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*_instance_metrics.json"))):
        with open(path) as handle:
            summary = json.load(handle)
        measured[summary["name"]] = (summary["mpq"], summary["bpq"])

    if not measured:
        print("No *_instance_metrics.json found; run evaluate_instance.py first.")
        return

    published = [
        ("HoVer-Net", 0.4629, 0.6596),
        ("CellViT-SAM-H", 0.4980, 0.6793),
        ("LKCell-L", 0.5080, 0.6851),
    ]

    names = [p[0] for p in published] + list(measured)
    mpq = [p[1] for p in published] + [v[0] for v in measured.values()]
    bpq = [p[2] for p in published] + [v[1] for v in measured.values()]
    is_ours = [False] * len(published) + [True] * len(measured)

    positions = np.arange(len(names))
    width = 0.38

    figure, axes = plt.subplots(figsize=(1.5 * len(names) + 3, 5))

    for offset, values, label, color in [
        (-width / 2, mpq, "mPQ", "#0072B2"),
        (width / 2, bpq, "bPQ", "#E69F00"),
    ]:
        bars = axes.bar(positions + offset, values, width * 0.92, label=label,
                        color=color, edgecolor="white", linewidth=0.6)
        # Published rows are drawn hollow so the measured ones read as the
        # contribution rather than sitting anonymously in the same row.
        for bar, ours in zip(bars, is_ours):
            if not ours:
                bar.set_alpha(0.45)
        for bar, value in zip(bars, values):
            axes.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f}",
                      ha="center", va="bottom", fontsize=8, color="#444444")

    axes.set_xticks(positions)
    axes.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
    axes.set_ylabel("Panoptic quality")
    axes.set_ylim(0, max(max(mpq), max(bpq)) * 1.2)
    axes.set_title("PanNuke, official three-fold protocol (solid: this repo)",
                   fontsize=12, fontweight="bold")
    axes.legend(fontsize=10, frameon=False)
    axes.grid(axis="y", alpha=0.25, linewidth=0.6)
    axes.set_axisbelow(True)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)

    figure.tight_layout()
    figure.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves and benchmark comparison")
    parser.add_argument("--splits", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--figures-dir", type=str, default=FIGURES_DIR)
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    histories = find_histories(args.results_dir, args.splits)
    if histories:
        for (method, split), data in sorted(histories.items()):
            print(f"Loaded {method} split {split}: {len(data.get('epoch', []))} epochs")
        plot_curves(histories, os.path.join(args.figures_dir, "training_curves.png"))
    else:
        print("No history CSVs found. Run training first.")

    plot_benchmark(args.results_dir, os.path.join(args.figures_dir, "benchmark_comparison.png"))


if __name__ == "__main__":
    main()
