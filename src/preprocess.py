"""
Preprocess PanNuke into the official three-fold benchmark format.

Loads each raw fold one at a time and writes, per fold:
    images    (N, 256, 256, 3) uint8   H&E patch
    insts     (N, 256, 256)    int16   nucleus instance IDs, 0 = background
    types     (N, 256, 256)    uint8   nucleus class 1-5, 0 = background
    masks     (N, 256, 256)    uint8   binary nuclei mask (legacy baselines)
    tissues   (N,)             object  tissue type string per patch

The raw PanNuke masks are (N, 256, 256, 6) where channels 0-4 hold per-class
instance IDs and channel 5 is background. IDs are only unique within a channel,
so each channel is relabelled to a contiguous range and offset past the IDs
already assigned, giving one unique ID per nucleus in the patch.

Folds are kept intact. The official PanNuke protocol trains on one fold,
validates on a second and tests on the third, rotating over all three splits
(see SPLITS in dataset.py). The folds must never be pooled and re-split at
random: patches within a fold can come from the same tissue section, so a
pooled random split leaks between train and test and produces numbers that
cannot be compared against any published result.

Memory-efficient: masks are memory-mapped and converted one patch at a time,
so peak RAM stays at roughly one fold of images.

Output structure:
    data/processed/
        fold1_images.npy
        fold1_insts.npy
        fold1_types.npy
        fold1_masks.npy
        fold1_tissues.npy
        fold2_...
        fold3_...

Usage:
    python src/preprocess.py
    python src/preprocess.py --folds fold_1 fold_2
"""

import argparse
import os

import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

FOLDS = ["fold_1", "fold_2", "fold_3"]

# PanNuke nuclei classes, in raw mask channel order. Channel 5 is background.
TYPE_NAMES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
NUM_TYPES = len(TYPE_NAMES)


def get_fold_paths(fold_name):
    """Return paths to images, tissue types, masks for a fold."""
    fold_num = fold_name.split("_")[1]
    folder_name = f"Fold {fold_num}"
    img_path = os.path.join(RAW_DIR, folder_name, "images", f"fold{fold_num}", "images.npy")
    tissue_path = os.path.join(RAW_DIR, folder_name, "images", f"fold{fold_num}", "types.npy")
    masks_path = os.path.join(RAW_DIR, folder_name, "masks", f"fold{fold_num}", "masks.npy")
    return img_path, tissue_path, masks_path


def remap_label(label):
    """
    Relabel an ID map so IDs are contiguous 1..N, keeping 0 as background.

    PanNuke IDs are sparse (a patch can hold IDs 3, 17, 204), and the fast PQ
    implementation indexes instance masks by ID, so IDs must be contiguous.

    Args:
        label: (H, W) integer-valued array

    Returns:
        (H, W) int32 array with IDs 1..N
    """
    label = np.asarray(label).astype(np.int32)
    ids, inverse = np.unique(label, return_inverse=True)
    out = inverse.reshape(label.shape).astype(np.int32)

    # np.unique sorts, so background (0) lands at index 0 when present. If the
    # patch has no background at all, shift so IDs still start at 1.
    if ids[0] != 0:
        out += 1

    return out


def mask_to_instance_and_type(mask):
    """
    Convert one PanNuke (256, 256, 6) mask into an instance map and a type map.

    Args:
        mask: (256, 256, 6) per-class instance IDs, channel 5 is background

    Returns:
        inst: (256, 256) int16, unique ID per nucleus, 0 = background
        type_map: (256, 256) uint8, class 1-5, 0 = background
    """
    inst = np.zeros(mask.shape[:2], dtype=np.int32)
    type_map = np.zeros(mask.shape[:2], dtype=np.uint8)

    offset = 0
    for ch in range(NUM_TYPES):
        layer = remap_label(mask[:, :, ch])
        if layer.max() == 0:
            continue

        foreground = layer > 0
        inst[foreground] = layer[foreground] + offset
        type_map[foreground] = ch + 1
        offset = int(inst.max())

    if inst.max() > np.iinfo(np.int16).max:
        raise ValueError(f"Patch has {inst.max()} instances, too many for int16")

    return inst.astype(np.int16), type_map


def process_fold(fold_name, out_dir):
    """Convert one raw fold and save its arrays to out_dir."""
    fold_key = fold_name.replace("_", "")
    img_path, tissue_path, masks_path = get_fold_paths(fold_name)

    print(f"\nProcessing {fold_name}...")

    images = np.load(img_path)
    if images.dtype != np.uint8:
        # Some PanNuke releases ship images as float64 in [0, 255].
        images = np.clip(images, 0, 255).astype(np.uint8)

    tissues = np.load(tissue_path, allow_pickle=True)

    # Masks are the largest array by far (float64, 6 channels), so stream them.
    masks = np.load(masks_path, mmap_mode="r")

    n = images.shape[0]
    print(f"  {n} patches, images {images.shape} {images.dtype}, masks {masks.shape} {masks.dtype}")

    insts = np.zeros((n, 256, 256), dtype=np.int16)
    types = np.zeros((n, 256, 256), dtype=np.uint8)

    for i in range(n):
        inst, type_map = mask_to_instance_and_type(np.asarray(masks[i]))
        insts[i] = inst
        types[i] = type_map

        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n} patches converted")

    binary = (insts > 0).astype(np.uint8)

    total_nuclei = sum(int(insts[i].max()) for i in range(n))
    print(f"  {total_nuclei} nuclei, {binary.mean() * 100:.2f}% foreground pixels")

    np.save(os.path.join(out_dir, f"{fold_key}_images.npy"), images)
    np.save(os.path.join(out_dir, f"{fold_key}_insts.npy"), insts)
    np.save(os.path.join(out_dir, f"{fold_key}_types.npy"), types)
    np.save(os.path.join(out_dir, f"{fold_key}_masks.npy"), binary)
    np.save(os.path.join(out_dir, f"{fold_key}_tissues.npy"), tissues)

    print(f"  Saved {fold_key}_*.npy")

    return n, total_nuclei


def main():
    parser = argparse.ArgumentParser(description="Preprocess PanNuke into official fold arrays")
    parser.add_argument("--folds", type=str, nargs="+", default=FOLDS, choices=FOLDS)
    parser.add_argument("--out-dir", type=str, default=PROCESSED_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    totals = []
    for fold_name in args.folds:
        totals.append(process_fold(fold_name, args.out_dir))

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for fold_name, (n, nuclei) in zip(args.folds, totals):
        print(f"  {fold_name}: {n} patches, {nuclei} nuclei")
    print(f"  Total: {sum(t[0] for t in totals)} patches, {sum(t[1] for t in totals)} nuclei")
    print(f"\nDone. Saved to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
