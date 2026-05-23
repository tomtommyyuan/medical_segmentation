"""
Preprocess PanNuke: convert to binary masks and create deterministic splits.

Loads folds one at a time from data/raw/, converts the 6-channel masks
into binary (nuclei vs background), and splits deterministically into
train/val/test (70/15/15).

Memory-efficient: processes and frees each fold before loading the next.

Output structure:
    data/processed/
        train_images.npy   (N_train, 256, 256, 3)
        train_masks.npy    (N_train, 256, 256)      binary uint8
        train_types.npy    (N_train,)               tissue type strings
        val_images.npy
        val_masks.npy
        val_types.npy
        test_images.npy
        test_masks.npy
        test_types.npy

Usage:
    python src/preprocess.py
"""

import os
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15 (remainder)

FOLDS = ["fold_1", "fold_2", "fold_3"]


def get_fold_paths(fold_name):
    """Return paths to images, types, masks for a fold."""
    fold_num = fold_name.split("_")[1]
    folder_name = f"Fold {fold_num}"
    img_path = os.path.join(RAW_DIR, folder_name, "images", f"fold{fold_num}", "images.npy")
    types_path = os.path.join(RAW_DIR, folder_name, "images", f"fold{fold_num}", "types.npy")
    masks_path = os.path.join(RAW_DIR, folder_name, "masks", f"fold{fold_num}", "masks.npy")
    return img_path, types_path, masks_path


def get_fold_size(fold_name):
    """Get number of samples in a fold without loading the full array."""
    img_path, _, _ = get_fold_paths(fold_name)
    images = np.load(img_path, mmap_mode="r")
    n = images.shape[0]
    del images
    return n


def masks_to_binary(masks):
    """
    Convert PanNuke multi-class masks to binary.

    PanNuke masks shape: (N, 256, 256, 6)
    Channels 0-4: five nuclei classes (neoplastic, inflammatory, connective, dead, epithelial)
    Channel 5: background

    Binary mask: 1 where any nuclei class > 0, else 0.
    """
    nuclei_channels = masks[:, :, :, :5]
    binary = (nuclei_channels.sum(axis=-1) > 0).astype(np.uint8)
    return binary


def deterministic_split(n, seed=SEED):
    """Return train/val/test index arrays with fixed random seed."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return train_idx, val_idx, test_idx


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # First pass: get total size
    fold_sizes = []
    for fold_name in FOLDS:
        n = get_fold_size(fold_name)
        fold_sizes.append(n)
        print(f"{fold_name}: {n} samples")

    total = sum(fold_sizes)
    print(f"Total: {total} samples")

    # Compute split indices over the full dataset
    train_idx, val_idx, test_idx = deterministic_split(total)
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Convert global indices to sets for fast lookup
    split_indices = {"train": np.sort(train_idx), "val": np.sort(val_idx), "test": np.sort(test_idx)}

    # Pre-allocate output arrays
    split_images = {s: np.zeros((len(idx), 256, 256, 3), dtype=np.uint8) for s, idx in split_indices.items()}
    split_masks = {s: np.zeros((len(idx), 256, 256), dtype=np.uint8) for s, idx in split_indices.items()}
    split_types = {s: np.empty(len(idx), dtype=object) for s, idx in split_indices.items()}

    # Track where to insert into each split's output array
    split_pos = {"train": 0, "val": 0, "test": 0}

    # Build a mapping: for each global index, which split and what position
    index_to_split = {}
    for split_name, idx in split_indices.items():
        for pos, global_idx in enumerate(idx):
            index_to_split[int(global_idx)] = (split_name, pos)

    # Second pass: load one fold at a time and distribute samples
    offset = 0
    for fold_name, fold_size in zip(FOLDS, fold_sizes):
        print(f"\nProcessing {fold_name}...")
        img_path, types_path, masks_path = get_fold_paths(fold_name)

        images = np.load(img_path)
        types = np.load(types_path, allow_pickle=True)
        masks = np.load(masks_path)

        print(f"  Converting masks to binary...")
        binary_masks = masks_to_binary(masks)
        del masks

        # Place each sample in its split
        for local_i in range(fold_size):
            global_i = offset + local_i
            split_name, pos = index_to_split[global_i]
            split_images[split_name][pos] = images[local_i]
            split_masks[split_name][pos] = binary_masks[local_i]
            split_types[split_name][pos] = types[local_i]

        offset += fold_size
        del images, binary_masks, types

    # Save
    print("\nSaving...")
    for split_name in ["train", "val", "test"]:
        np.save(os.path.join(PROCESSED_DIR, f"{split_name}_images.npy"), split_images[split_name])
        np.save(os.path.join(PROCESSED_DIR, f"{split_name}_masks.npy"), split_masks[split_name])
        np.save(os.path.join(PROCESSED_DIR, f"{split_name}_types.npy"), split_types[split_name])
        print(f"  {split_name}: {len(split_images[split_name])} samples")

    print(f"\nDone. Saved to: {os.path.abspath(PROCESSED_DIR)}")


if __name__ == "__main__":
    main()
