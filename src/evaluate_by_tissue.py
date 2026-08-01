"""
Per-tissue bPQ for every method, and the figure that goes with it.

Cross-tissue variance is the axis the stain consistency term is meant to move,
so it gets its own figure. Every method is scored on the same metric: binary
panoptic quality on the official test folds. The binary baselines get instances
from connected components, which is the best they can do without distance maps.

Reporting one metric for all methods matters. The earlier version of this
script plotted per-tissue Dice, which the binary baselines can score well on
while merging every touching nucleus, so it flattered exactly the failure the
project is about.

Usage:
    python src/evaluate_by_tissue.py
    python src/evaluate_by_tissue.py --splits 1 --methods unet chroma
"""

import argparse
import json
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import SPLITS, NucleiDataset
from evaluate_instance import (
    load_chroma,
    predict_binary_baseline,
    predict_chroma,
    score_fold,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

METHOD_LABELS = {
    "classical": "Classical",
    "cnn": "CNN",
    "unet": "U-Net",
    "attention_unet": "Att. U-Net",
    "chroma": "CHROMA-Net",
}
DEFAULT_METHODS = ["classical", "cnn", "unet", "attention_unet", "chroma"]

# Okabe-Ito hues in fixed order, checked for colour-vision deficiency
# separation (worst adjacent pair dE 11.0 deutan). Assigned by method, never
# cycled, so a method keeps its colour when the set being plotted changes.
METHOD_COLORS = {
    "classical": "#0072B2",
    "cnn": "#D55E00",
    "unet": "#009E73",
    "attention_unet": "#E69F00",
    "chroma": "#CC79A7",
}


def score_method(method, split, args, device):
    """Per-tissue bPQ for one method on one split's test fold."""
    fold = SPLITS[split]["test"]

    dataset = NucleiDataset(fold, data_dir=args.data_dir, return_instances=True)
    true_inst = np.asarray(dataset.data["insts"]).astype(np.int32)
    true_type = np.asarray(dataset.data["types"])
    tissues = np.asarray([str(t) for t in dataset.data["tissues"]])

    if method == "chroma":
        checkpoint_path = os.path.join(args.out_dir, f"{args.tag}_split{split}_best.pth")
        model, _ = load_chroma(checkpoint_path, device)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        pred_inst, pred_type = predict_chroma(model, loader, device, args.tta)
    else:
        pred_inst, pred_type = predict_binary_baseline(
            method, fold, args.data_dir, device, split, args.out_dir
        )

    return score_fold(pred_inst, pred_type, true_inst, true_type, tissues)


def plot_by_tissue(table, tissues, methods, out_path):
    """
    Horizontal grouped bars, one row per tissue, sorted by the best method.

    Horizontal rather than vertical because 19 tissue names do not fit on an x
    axis without rotating them, and rotated labels are harder to scan than the
    values they label.
    """
    positions = np.arange(len(tissues))
    height = 0.8 / len(methods)

    figure, axes = plt.subplots(figsize=(9, 0.55 * len(tissues) + 1.6))

    for i, method in enumerate(methods):
        values = [table[method][tissue] for tissue in tissues]
        offset = (i - (len(methods) - 1) / 2) * height

        axes.barh(positions + offset, values, height * 0.9,
                  label=METHOD_LABELS[method], color=METHOD_COLORS[method],
                  edgecolor="white", linewidth=0.5)

        # Direct labels on the leading method only. A number on all 95 bars is
        # noise, and one series labelled is enough to read the scale.
        if method == methods[-1]:
            for y, value in zip(positions + offset, values):
                if np.isfinite(value):
                    axes.text(value + 0.008, y, f"{value:.2f}", va="center",
                              fontsize=7, color="#444444")

    axes.set_yticks(positions)
    axes.set_yticklabels(tissues, fontsize=9)
    axes.set_xlabel("Binary Panoptic Quality (bPQ)", fontsize=11)
    axes.set_xlim(0, 1.0)
    axes.set_title("Per-tissue instance segmentation quality", fontsize=13, fontweight="bold")
    axes.legend(fontsize=9, loc="lower right", frameon=False)

    axes.grid(axis="x", alpha=0.25, linewidth=0.6)
    axes.set_axisbelow(True)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_visible(False)
    axes.tick_params(left=False)

    figure.tight_layout()
    figure.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Per-tissue bPQ across methods")
    parser.add_argument("--methods", type=str, nargs="+", default=DEFAULT_METHODS,
                        choices=DEFAULT_METHODS)
    parser.add_argument("--splits", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument("--tag", type=str, default="chroma")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--out-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--figures-dir", type=str, default=FIGURES_DIR)
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Methods: {args.methods}, splits: {args.splits}")

    # table[method][tissue] = bPQ averaged over the evaluated splits
    table = {}
    for method in args.methods:
        print(f"\nScoring {method}...")
        per_split = [score_method(method, split, args, device) for split in args.splits]

        tissue_scores = {}
        for tissue in sorted({t for r in per_split for t in r["per_tissue"]}):
            values = [r["per_tissue"][tissue]["bpq"] for r in per_split
                      if tissue in r["per_tissue"]]
            tissue_scores[tissue] = float(np.nanmean(values))

        table[method] = tissue_scores
        overall = float(np.nanmean(list(tissue_scores.values())))
        spread = float(np.nanstd(list(tissue_scores.values())))
        print(f"  bPQ {overall:.4f}, across-tissue std {spread:.4f}")

    # Sort tissues by the last method's score so the figure reads as a ranking.
    reference = args.methods[-1]
    tissues = sorted(table[reference], key=lambda t: table[reference][t])

    print(f"\n{'=' * 80}")
    print("  bPQ BY TISSUE TYPE")
    print(f"{'=' * 80}")
    header = f"  {'Tissue':<18}"
    for method in args.methods:
        header += f"{METHOD_LABELS[method]:>14}"
    print(header)
    print(f"  {'-' * 18}" + f"{'-' * 14}" * len(args.methods))

    for tissue in tissues:
        row = f"  {tissue:<18}"
        for method in args.methods:
            row += f"{table[method][tissue]:>14.4f}"
        print(row)

    print(f"  {'-' * 18}" + f"{'-' * 14}" * len(args.methods))
    row = f"  {'Mean':<18}"
    spread_row = f"  {'Std across tissue':<18}"
    for method in args.methods:
        values = list(table[method].values())
        row += f"{np.nanmean(values):>14.4f}"
        spread_row += f"{np.nanstd(values):>14.4f}"
    print(row)
    print(spread_row)

    figure_path = os.path.join(args.figures_dir, "bpq_by_tissue.png")
    plot_by_tissue(table, tissues, args.methods, figure_path)
    print(f"\nFigure saved to: {figure_path}")

    results_path = os.path.join(args.out_dir, "tissue_bpq.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(results_path, "w") as handle:
        json.dump(table, handle, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
