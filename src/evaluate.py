"""
Evaluate trained models on the test set.

Loads best checkpoints and computes Dice, IoU, precision, recall, and cell count error.
Also runs the classical baseline (no checkpoint needed).

Usage:
    python src/evaluate.py                    # Evaluate all models
    python src/evaluate.py --model unet       # Evaluate a single model
"""

import argparse
import os

import numpy as np
import torch
from scipy import ndimage

from cnn_baseline import CNNBaseline
from unet import UNet
from attention_unet import AttentionUNet
from classical import segment_batch

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_test_data():
    images = np.load(os.path.join(DATA_DIR, "test_images.npy"))
    masks = np.load(os.path.join(DATA_DIR, "test_masks.npy"))
    types = np.load(os.path.join(DATA_DIR, "test_types.npy"), allow_pickle=True)
    return images, masks, types


def compute_metrics(pred, target):
    """
    Compute segmentation metrics for binary masks.

    Args:
        pred: (N, 256, 256) uint8 binary predictions
        target: (N, 256, 256) uint8 binary ground truth

    Returns:
        dict with dice, iou, precision, recall, cell_count_error
    """
    pred_flat = pred.reshape(pred.shape[0], -1).astype(np.float64)
    target_flat = target.reshape(target.shape[0], -1).astype(np.float64)

    smooth = 1e-5

    intersection = (pred_flat * target_flat).sum(axis=1)
    pred_sum = pred_flat.sum(axis=1)
    target_sum = target_flat.sum(axis=1)

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    iou = (intersection + smooth) / (pred_sum + target_sum - intersection + smooth)
    precision = (intersection + smooth) / (pred_sum + smooth)
    recall = (intersection + smooth) / (target_sum + smooth)

    # Cell count error via connected components
    pred_counts = np.array([ndimage.label(p)[1] for p in pred])
    target_counts = np.array([ndimage.label(t)[1] for t in target])
    cell_count_error = np.abs(pred_counts - target_counts).astype(np.float64)

    return {
        "dice": dice.mean(),
        "iou": iou.mean(),
        "precision": precision.mean(),
        "recall": recall.mean(),
        "cell_count_mae": cell_count_error.mean(),
        "dice_std": dice.std(),
        "iou_std": iou.std(),
    }


def predict_neural(model_name, images, device):
    """Run inference with a trained neural model."""
    if model_name == "cnn":
        model = CNNBaseline()
    elif model_name == "unet":
        model = UNet()
    elif model_name == "attention_unet":
        model = AttentionUNet()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    checkpoint_path = os.path.join(RESULTS_DIR, f"{model_name}_best.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    batch_size = 32
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size].astype(np.float32) / 255.0
            batch = np.transpose(batch, (0, 3, 1, 2))
            batch_tensor = torch.from_numpy(batch).to(device)

            logits = model(batch_tensor)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().squeeze(1).astype(np.uint8)
            all_preds.append(preds)

    return np.concatenate(all_preds, axis=0)


def print_results(name, metrics):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Dice:       {metrics['dice']:.4f} ± {metrics['dice_std']:.4f}")
    print(f"  IoU:        {metrics['iou']:.4f} ± {metrics['iou_std']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  Cell Count MAE: {metrics['cell_count_mae']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation models on test set")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "classical", "cnn", "unet", "attention_unet"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading test data...")
    images, masks, types = load_test_data()
    print(f"Test samples: {len(images)}")

    models_to_eval = (
        ["classical", "cnn", "unet", "attention_unet"] if args.model == "all" else [args.model]
    )

    all_results = {}

    for model_name in models_to_eval:
        print(f"\nRunning {model_name}...")

        if model_name == "classical":
            preds = segment_batch(images)
        else:
            preds = predict_neural(model_name, images, device)

        metrics = compute_metrics(preds, masks)
        all_results[model_name] = metrics
        print_results(model_name, metrics)

    # Summary table
    if len(all_results) > 1:
        print(f"\n\n{'='*70}")
        print(f"  SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Model':<18} {'Dice':<12} {'IoU':<12} {'Precision':<12} {'Recall':<12} {'Cell MAE':<10}")
        print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
        for name, m in all_results.items():
            print(f"  {name:<18} {m['dice']:<12.4f} {m['iou']:<12.4f} "
                  f"{m['precision']:<12.4f} {m['recall']:<12.4f} {m['cell_count_mae']:<10.2f}")

    # Save results
    results_path = os.path.join(RESULTS_DIR, "test_metrics.npy")
    np.save(results_path, all_results)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
