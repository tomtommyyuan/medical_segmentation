"""
Evaluate the binary baselines on the official PanNuke protocol.

Reports Dice, IoU, precision and recall on each split's test fold, averaged
over the three splits. These are pixel-level scores only. For the metrics the
PanNuke benchmark is ranked on, and for anything involving separate nuclei or
their classes, use evaluate_instance.py.

Two changes from the earlier version of this script, both of which moved the
numbers:

  - Scores are accumulated over the whole fold and divided once, rather than
    averaged per patch with a 1e-5 smoothing term. That term gave every patch
    containing no nuclei a Dice of exactly 1.0, so the reported score partly
    measured how many empty patches a fold happened to contain.

  - The cell-count error is gone. It counted connected components of a binary
    mask, which merges every touching nucleus into one, so it measured blob
    count rather than nuclei. evaluate_instance.py counts real instances.

Usage:
    python src/evaluate.py                    # all baselines, all splits
    python src/evaluate.py --model unet       # one model
    python src/evaluate.py --splits 1         # one split
"""

import argparse
import json
import os

import numpy as np
import torch

from attention_unet import AttentionUNet
from classical import segment_batch
from cnn_baseline import CNNBaseline
from dataset import SPLITS, load_fold
from unet import UNet

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

MODELS = {"cnn": CNNBaseline, "unet": UNet, "attention_unet": AttentionUNet}
ALL_MODELS = ["classical", "cnn", "unet", "attention_unet"]


def compute_metrics(pred, target):
    """
    Aggregate segmentation metrics over a whole fold.

    Args:
        pred: (N, H, W) uint8 binary predictions
        target: (N, H, W) uint8 binary ground truth

    Returns:
        dict with dice, iou, precision, recall, plus the per-image Dice over
        patches that actually contain nuclei, for distribution reporting
    """
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    intersection = float((pred * target).sum())
    pred_total = float(pred.sum())
    target_total = float(target.sum())

    per_image_intersection = (pred * target).sum(axis=(1, 2))
    per_image_pred = pred.sum(axis=(1, 2))
    per_image_target = target.sum(axis=(1, 2))

    # Only patches with nuclei: an empty patch has no meaningful Dice, and
    # scoring it 1.0 is what inflated the original numbers.
    nonempty = per_image_target > 0
    per_image_dice = (
        2.0 * per_image_intersection[nonempty]
        / np.maximum(per_image_pred[nonempty] + per_image_target[nonempty], 1.0)
    )

    return {
        "dice": 2.0 * intersection / max(pred_total + target_total, 1.0),
        "iou": intersection / max(pred_total + target_total - intersection, 1.0),
        "precision": intersection / max(pred_total, 1.0),
        "recall": intersection / max(target_total, 1.0),
        "per_image_dice_mean": float(per_image_dice.mean()) if len(per_image_dice) else float("nan"),
        "per_image_dice_std": float(per_image_dice.std()) if len(per_image_dice) else float("nan"),
        "n_nonempty": int(nonempty.sum()),
        "n_total": int(len(pred)),
    }


@torch.no_grad()
def predict_neural(model_name, images, split, device, results_dir):
    """Run inference with a trained baseline."""
    model = MODELS[model_name]()
    checkpoint_path = os.path.join(results_dir, f"{model_name}_split{split}_best.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device).eval()

    predictions = []
    for start in range(0, len(images), 32):
        batch = np.asarray(images[start : start + 32]).astype(np.float32) / 255.0
        batch = np.transpose(batch, (0, 3, 1, 2))
        logits = model(torch.from_numpy(batch).to(device))
        predictions.append((torch.sigmoid(logits) > 0.5).cpu().numpy().squeeze(1).astype(np.uint8))

    return np.concatenate(predictions)


def evaluate_model(model_name, splits, device, data_dir, results_dir):
    """Score one model across the requested splits."""
    per_split = {}

    for split in splits:
        fold = SPLITS[split]["test"]
        data = load_fold(fold, data_dir)
        images = data["images"]
        masks = np.asarray(data["masks"])

        if model_name == "classical":
            predictions = segment_batch(np.asarray(images))
        else:
            predictions = predict_neural(model_name, images, split, device, results_dir)

        per_split[split] = compute_metrics(predictions, masks)

    averaged = {
        key: float(np.mean([per_split[s][key] for s in splits]))
        for key in ["dice", "iou", "precision", "recall", "per_image_dice_mean"]
    }
    averaged["per_split"] = {str(s): per_split[s] for s in splits}

    return averaged


def print_results(name, metrics):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Dice (aggregate):  {metrics['dice']:.4f}")
    print(f"  IoU  (aggregate):  {metrics['iou']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  Dice (per patch, nuclei-bearing only): {metrics['per_image_dice_mean']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate binary baselines on the official folds")
    parser.add_argument("--model", type=str, default="all", choices=["all"] + ALL_MODELS)
    parser.add_argument("--splits", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--out-dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Splits: {args.splits}")

    models = ALL_MODELS if args.model == "all" else [args.model]
    all_results = {}

    for model_name in models:
        print(f"\nRunning {model_name}...")
        all_results[model_name] = evaluate_model(
            model_name, args.splits, device, args.data_dir, args.out_dir
        )
        print_results(model_name, all_results[model_name])

    if len(all_results) > 1:
        print(f"\n\n{'=' * 72}")
        print("  SUMMARY (averaged over splits)")
        print(f"{'=' * 72}")
        print(f"  {'Model':<18} {'Dice':<10} {'IoU':<10} {'Precision':<12} {'Recall':<10}")
        print(f"  {'-' * 18} {'-' * 10} {'-' * 10} {'-' * 12} {'-' * 10}")
        for name, m in all_results.items():
            print(f"  {name:<18} {m['dice']:<10.4f} {m['iou']:<10.4f} "
                  f"{m['precision']:<12.4f} {m['recall']:<10.4f}")
        print("\n  Pixel-level only. See evaluate_instance.py for mPQ and bPQ.")

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "binary_metrics.json")
    with open(results_path, "w") as handle:
        json.dump(all_results, handle, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
