"""
Qualitative comparison figure: instances, not masks.

Nuclei are drawn with one colour per instance rather than as a white
foreground, because that is the only way the interesting failure is visible. A
binary mask of two merged nuclei and a binary mask of two separated nuclei look
identical; coloured instances show the merge immediately, and the per-panel bPQ
puts a number on it.

Rows are chosen by CHROMA-Net's per-patch bPQ so the figure spans an easy case,
a typical one and a hard one rather than three cherry-picked wins.

Usage:
    python src/visualize_qualitative.py
    python src/visualize_qualitative.py --split 1 --methods unet chroma
"""

import argparse
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from skimage.color import label2rgb
from torch.utils.data import DataLoader

from dataset import SPLITS, NucleiDataset
from evaluate_instance import load_chroma, predict_binary_baseline, predict_chroma
from pq_metrics import get_fast_pq

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
DEFAULT_METHODS = ["classical", "unet", "chroma"]

ROW_LABELS = ["Easy", "Typical", "Hard"]


def instances_to_rgb(inst_map, seed=0):
    """Colour an instance map, one hue per nucleus, black background."""
    if inst_map.max() == 0:
        return np.zeros(inst_map.shape + (3,), dtype=np.float64)

    rng = np.random.default_rng(seed)
    colors = rng.uniform(0.35, 1.0, size=(int(inst_map.max()) + 1, 3))

    return label2rgb(inst_map, colors=colors, bg_label=0, bg_color=(0, 0, 0))


def pick_rows(scores, n_nuclei, count=3):
    """
    Pick an easy, a typical and a hard patch by score.

    Patches with almost no nuclei are skipped: they score extreme values for
    uninteresting reasons and would waste a row.
    """
    eligible = np.where(n_nuclei >= 10)[0]
    if len(eligible) < count:
        eligible = np.arange(len(scores))

    order = eligible[np.argsort(scores[eligible])]

    return [order[-1], order[len(order) // 2], order[0]][:count]


def main():
    parser = argparse.ArgumentParser(description="Qualitative instance comparison figure")
    parser.add_argument("--methods", type=str, nargs="+", default=DEFAULT_METHODS,
                        choices=list(METHOD_LABELS))
    parser.add_argument("--split", type=int, default=1, choices=[1, 2, 3])
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
    fold = SPLITS[args.split]["test"]
    print(f"Device: {device}")
    print(f"Split {args.split}, test fold {fold}")

    dataset = NucleiDataset(fold, data_dir=args.data_dir, return_instances=True)
    images = np.asarray(dataset.data["images"])
    true_inst = np.asarray(dataset.data["insts"]).astype(np.int32)
    tissues = np.asarray([str(t) for t in dataset.data["tissues"]])
    print(f"Test samples: {len(images)}")

    predictions = {}
    for method in args.methods:
        print(f"Predicting with {method}...")
        if method == "chroma":
            checkpoint_path = os.path.join(args.out_dir, f"{args.tag}_split{args.split}_best.pth")
            model, _ = load_chroma(checkpoint_path, device)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
            predictions[method] = predict_chroma(model, loader, device, args.tta)[0]
        else:
            predictions[method] = predict_binary_baseline(
                method, fold, args.data_dir, device, args.split, args.out_dir
            )[0]

    # Rank patches by the last method's bPQ.
    reference = args.methods[-1]
    print(f"Scoring every patch with {reference} to choose rows...")
    scores = np.array([
        get_fast_pq(true_inst[i], predictions[reference][i])[0][2]
        for i in range(len(images))
    ])
    n_nuclei = np.array([len(np.unique(t)) - 1 for t in true_inst])

    rows = pick_rows(scores, n_nuclei)
    for index, label in zip(rows, ROW_LABELS):
        print(f"  {label}: idx={index}, tissue={tissues[index]}, "
              f"nuclei={n_nuclei[index]}, {reference} bPQ={scores[index]:.3f}")

    n_columns = 2 + len(args.methods)
    figure, axes = plt.subplots(
        len(rows), n_columns,
        figsize=(2.7 * n_columns, 2.9 * len(rows)),
        gridspec_kw={"wspace": 0.04, "hspace": 0.12},
        squeeze=False,
    )

    for row, (index, row_label) in enumerate(zip(rows, ROW_LABELS)):
        axes[row][0].imshow(images[index])
        axes[row][0].set_ylabel(f"{row_label}\n{tissues[index]}\n{n_nuclei[index]} nuclei",
                                fontsize=9, rotation=0, labelpad=42, va="center")

        axes[row][1].imshow(instances_to_rgb(true_inst[index]))

        for column, method in enumerate(args.methods, start=2):
            pred = predictions[method][index]
            axes[row][column].imshow(instances_to_rgb(pred))

            bpq = get_fast_pq(true_inst[index], pred)[0][2]
            found = len(np.unique(pred)) - 1
            axes[row][column].text(
                0.5, 0.02, f"bPQ {bpq:.3f}   {found} found",
                transform=axes[row][column].transAxes,
                ha="center", va="bottom", fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.65,
                          edgecolor="none"),
            )

        if row == 0:
            axes[row][0].set_title("H&E patch", fontsize=11)
            axes[row][1].set_title("Ground truth", fontsize=11)
            for column, method in enumerate(args.methods, start=2):
                axes[row][column].set_title(METHOD_LABELS[method], fontsize=11)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    figure.suptitle("Nuclei instances, one colour per nucleus",
                    fontsize=13, fontweight="bold", y=0.995)
    figure.tight_layout()

    out_path = os.path.join(args.figures_dir, "qualitative_instances.png")
    figure.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
