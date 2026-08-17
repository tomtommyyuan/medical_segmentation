"""
Evaluate on the official PanNuke protocol and report mPQ and bPQ.

Scores the test fold of each of the three splits and averages, which is the
number every published PanNuke result quotes. Anything computed on a different
split is not comparable, however good it looks.

Also scores the binary baselines by taking connected components of their masks
as instances. They cannot separate touching nuclei and have no class output, so
their bPQ is low and their mPQ undefined, which is exactly the gap the distance
maps and the class branch were added to close.

Usage:
    python src/evaluate_instance.py                          # CHROMA-Net, all splits
    python src/evaluate_instance.py --tta                    # with test-time augmentation
    python src/evaluate_instance.py --model unet             # a binary baseline
    python src/evaluate_instance.py --tag chroma_noconsist   # an ablation run
    python src/evaluate_instance.py --splits 1               # a single split
"""

import argparse
import json
import os

import numpy as np
import torch
from scipy.ndimage import label
from torch.utils.data import DataLoader

from attention_unet import AttentionUNet
from chroma_net import ChromaNet
from classical import segment_single
from cnn_baseline import CNNBaseline
from dataset import SPLITS, NucleiDataset, load_fold
from postprocess import decode_batch
from pq_metrics import aggregate_pq, pq_per_image
from preprocess import TYPE_NAMES
from tta import predict_plain, predict_tta
from unet import UNet

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

BINARY_MODELS = {"cnn": CNNBaseline, "unet": UNet, "attention_unet": AttentionUNet}

# Published results on the same three-fold protocol, for context in the report.
# Source: LKCell (arXiv 2407.18054) Table 2.
PUBLISHED = [
    ("HoVer-Net (2019)", 0.4629, 0.6596),
    ("StarDist", 0.4796, 0.6692),
    ("CPP-Net", 0.4815, 0.6767),
    ("CellViT-256", 0.4846, 0.6696),
    ("CellViT-SAM-H", 0.4980, 0.6793),
    ("LKCell-L (2024)", 0.5080, 0.6851),
]


def load_chroma(checkpoint_path, device):
    """Rebuild a CHROMA-Net from a checkpoint's saved configuration."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved = checkpoint["args"]

    model = ChromaNet(
        encoder=saved["encoder"],
        lora_rank=saved["lora_rank"],
        tissue_film=not saved["no_tissue_film"],
        # Weights come from the checkpoint, so skip the hub download.
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    return model, checkpoint


@torch.no_grad()
def predict_chroma(model, loader, device, use_tta, decode_params=None):
    """Run CHROMA-Net over a fold and decode instances."""
    decode_params = decode_params or {}
    inst_maps = []
    type_maps = []

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        tissue = batch["tissue"].to(device, non_blocking=True)

        predict = predict_tta if use_tta else predict_plain
        predictions = predict(model, image, tissue)

        np_prob = predictions["np"].float().squeeze(1).cpu().numpy()
        hv_pred = predictions["hv"].float().cpu().numpy()
        tp_pred = predictions["tp"].float().argmax(dim=1).cpu().numpy().astype(np.uint8)

        pred_inst, pred_type = decode_batch(np_prob, hv_pred, tp_pred, **decode_params)
        inst_maps.append(pred_inst)
        type_maps.append(pred_type)

    return np.concatenate(inst_maps), np.concatenate(type_maps)


@torch.no_grad()
def predict_binary_baseline(model_name, fold, data_dir, device, split,
                            results_dir=RESULTS_DIR):
    """
    Instances from a binary baseline, via connected components.

    Touching nuclei share a component and come out as one instance, which is
    the entire reason the distance maps exist. Class output does not exist, so
    the type map is left at zero and mPQ is undefined for these rows.
    """
    data = load_fold(fold, data_dir)
    images = data["images"]

    if model_name == "classical":
        masks = np.stack([segment_single(np.asarray(images[i])) for i in range(len(images))])
    else:
        model = BINARY_MODELS[model_name]()
        checkpoint_path = os.path.join(results_dir, f"{model_name}_split{split}_best.pth")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.to(device).eval()

        masks = []
        for start in range(0, len(images), 32):
            batch = np.asarray(images[start : start + 32]).astype(np.float32) / 255.0
            batch = np.transpose(batch, (0, 3, 1, 2))
            logits = model(torch.from_numpy(batch).to(device))
            masks.append((torch.sigmoid(logits) > 0.5).cpu().numpy().squeeze(1).astype(np.uint8))
        masks = np.concatenate(masks)

    inst_maps = np.stack([label(m)[0] for m in masks]).astype(np.int32)
    type_maps = np.zeros_like(inst_maps, dtype=np.uint8)

    return inst_maps, type_maps


def score_fold(pred_inst, pred_type, true_inst, true_type, tissues):
    """Per-image PQ over a whole fold."""
    bpq_list = []
    class_pq_list = []
    bdq_list = []
    bsq_list = []

    for i in range(len(pred_inst)):
        bpq, class_pq, bdq, bsq = pq_per_image(
            true_inst[i], true_type[i], pred_inst[i], pred_type[i]
        )
        bpq_list.append(bpq)
        class_pq_list.append(class_pq)
        bdq_list.append(bdq)
        bsq_list.append(bsq)

    return aggregate_pq(bpq_list, class_pq_list, tissues, bdq_list, bsq_list)


def decode_params(args):
    """Watershed parameters from the command line, tuned by tune_postprocess.py."""
    return {
        "np_threshold": args.np_threshold,
        "marker_threshold": args.marker_threshold,
        "min_size": args.min_size,
        "sobel_ksize": args.sobel_ksize,
    }


def evaluate_split(args, split, device):
    """Score one split's test fold."""
    fold = SPLITS[split]["test"]
    print(f"\n  Split {split}: testing on {fold}")

    dataset = NucleiDataset(fold, data_dir=args.data_dir, return_instances=True)
    true_inst = np.asarray(dataset.data["insts"]).astype(np.int32)
    true_type = np.asarray(dataset.data["types"])
    tissues = np.asarray([str(t) for t in dataset.data["tissues"]])

    if args.model == "chroma":
        checkpoint_path = os.path.join(args.out_dir, f"{args.tag}_split{split}_best.pth")
        model, checkpoint = load_chroma(checkpoint_path, device)
        print(f"    checkpoint epoch {checkpoint['epoch']}, val mPQ {checkpoint['val_mpq']:.4f}")

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        pred_inst, pred_type = predict_chroma(model, loader, device, args.tta,
                                              decode_params(args))
    else:
        pred_inst, pred_type = predict_binary_baseline(
            args.model, fold, args.data_dir, device, split, args.out_dir
        )

    return score_fold(pred_inst, pred_type, true_inst, true_type, tissues)


def main():
    parser = argparse.ArgumentParser(description="Evaluate on the official PanNuke protocol")
    parser.add_argument("--model", type=str, default="chroma",
                        choices=["chroma", "classical", "cnn", "unet", "attention_unet"])
    parser.add_argument("--tag", type=str, default="chroma",
                        help="run tag used by train_chroma.py")
    parser.add_argument("--splits", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument("--tta", action="store_true", help="average over the 8 dihedral views")
    parser.add_argument("--np-threshold", type=float, default=0.5)
    parser.add_argument("--marker-threshold", type=float, default=0.4)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--sobel-ksize", type=int, default=21)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--out-dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.tag if args.model == "chroma" else args.model
    if args.tta:
        name += " + TTA"

    print(f"Device: {device}")
    print(f"Evaluating: {name} on splits {args.splits}")

    per_split = {}
    for split in args.splits:
        per_split[split] = evaluate_split(args, split, device)
        print(f"    mPQ {per_split[split]['mpq']:.4f}   bPQ {per_split[split]['bpq']:.4f}"
              f"   (bDQ {per_split[split]['bdq']:.4f} x bSQ {per_split[split]['bsq']:.4f})")

    mpq = float(np.nanmean([r["mpq"] for r in per_split.values()]))
    bpq = float(np.nanmean([r["bpq"] for r in per_split.values()]))

    print(f"\n{'=' * 70}")
    print(f"  {name}: averaged over {len(args.splits)} split(s)")
    print(f"{'=' * 70}")
    bdq = float(np.nanmean([r["bdq"] for r in per_split.values()]))
    bsq = float(np.nanmean([r["bsq"] for r in per_split.values()]))

    print(f"  mPQ: {mpq:.4f}    bPQ: {bpq:.4f}")
    print(f"  bDQ: {bdq:.4f}    bSQ: {bsq:.4f}")
    print()
    print("  bDQ is detection: nuclei missed, invented, merged or split.")
    print("  bSQ is segmentation: how well matched nuclei overlap.")
    print("  A weak bDQ points at the watershed decoding, a weak bSQ at the decoder.")

    # Per-split detail, then the breakdown from the first split for context.
    print(f"\n  {'Split':<10} {'mPQ':>10} {'bPQ':>10}")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 10}")
    for split, results in per_split.items():
        print(f"  {split:<10} {results['mpq']:>10.4f} {results['bpq']:>10.4f}")

    # Per-tissue and per-class, averaged across the evaluated splits.
    tissue_names = sorted({t for r in per_split.values() for t in r["per_tissue"]})
    print(f"\n  {'Tissue':<18} {'mPQ':>10} {'bPQ':>10}")
    print(f"  {'-' * 18} {'-' * 10} {'-' * 10}")
    tissue_rows = {}
    for tissue in tissue_names:
        values = [r["per_tissue"][tissue] for r in per_split.values() if tissue in r["per_tissue"]]
        row = {
            "mpq": float(np.nanmean([v["mpq"] for v in values])),
            "bpq": float(np.nanmean([v["bpq"] for v in values])),
        }
        tissue_rows[tissue] = row
        print(f"  {tissue:<18} {row['mpq']:>10.4f} {row['bpq']:>10.4f}")

    per_class = np.nanmean(np.stack([r["per_class"] for r in per_split.values()]), axis=0)
    print(f"\n  {'Class':<18} {'PQ':>10}")
    print(f"  {'-' * 18} {'-' * 10}")
    for class_name, value in zip(TYPE_NAMES, per_class):
        print(f"  {class_name:<18} {value:>10.4f}")

    print(f"\n{'=' * 70}")
    print("  Published results on the same protocol")
    print(f"{'=' * 70}")
    print(f"  {'Method':<24} {'mPQ':>10} {'bPQ':>10}")
    print(f"  {'-' * 24} {'-' * 10} {'-' * 10}")
    for method, published_mpq, published_bpq in PUBLISHED:
        print(f"  {method:<24} {published_mpq:>10.4f} {published_bpq:>10.4f}")
    print(f"  {name:<24} {mpq:>10.4f} {bpq:>10.4f}   <- this run")

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {
        "name": name,
        "model": args.model,
        "tag": args.tag,
        "tta": args.tta,
        "splits": args.splits,
        "decode_params": decode_params(args),
        "mpq": mpq,
        "bpq": bpq,
        "bdq": bdq,
        "bsq": bsq,
        "per_split": {
            str(split): {"mpq": results["mpq"], "bpq": results["bpq"]}
            for split, results in per_split.items()
        },
        "per_tissue": tissue_rows,
        "per_class": {n: float(v) for n, v in zip(TYPE_NAMES, per_class)},
    }

    suffix = "_tta" if args.tta else ""
    results_path = os.path.join(args.out_dir, f"{name.split()[0]}{suffix}_instance_metrics.json")
    with open(results_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
