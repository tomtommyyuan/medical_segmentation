"""
Preprocess PanNuke: convert to binary masks and create deterministic splits.

Loads all 3 folds from data/raw/, merges them, converts the 6-channel masks
into binary (nuclei vs background), and splits deterministically into
train/val/test (70/15/15).

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


def load_fold(fold_name):
    """Load images, masks, and types from a single fold."""
    # PanNuke folder structure after extraction:
    #   Fold 1/images/fold1/images.npy
    #   Fold 1/images/fold1/types.npy
    #   Fold 1/masks/fold1/masks.npy
    fold_num = fold_name.split("_")[1]
    folder_name = f"Fold {fold_num}"
    img_path = os.path.join(RAW_DIR, folder_name, "images", f"fold{fold_num}", "images.npy")
    types_path = os.path.join(RAW_DIR, folder_name, "images", f"fold{fold_num}", "types.npy")
    masks_path = os.path.join(RAW_DIR, folder_name, "masks", f"fold{fold_num}", "masks.npy")

    print(f"Loading {fold_name}...")
    images = np.load(img_path)
    types = np.load(types_path, allow_pickle=True)
    masks = np.load(masks_path)

    print(f"  images: {images.shape}, masks: {masks.shape}, types: {types.shape}")
    return images, masks, types


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

    all_images = []
    all_masks = []
    all_types = []

    for fold_name in ["fold_1", "fold_2", "fold_3"]:
        images, masks, types = load_fold(fold_name)
        all_images.append(images)
        all_masks.append(masks)
        all_types.append(types)

    images = np.concatenate(all_images, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    types = np.concatenate(all_types, axis=0)

    print(f"\nTotal samples: {len(images)}")

    print("Converting masks to binary...")
    binary_masks = masks_to_binary(masks)

    print("Splitting into train/val/test...")
    train_idx, val_idx, test_idx = deterministic_split(len(images))
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        np.save(os.path.join(PROCESSED_DIR, f"{split_name}_images.npy"), images[idx])
        np.save(os.path.join(PROCESSED_DIR, f"{split_name}_masks.npy"), binary_masks[idx])
        np.save(os.path.join(PROCESSED_DIR, f"{split_name}_types.npy"), types[idx])

    print(f"\nSaved to: {os.path.abspath(PROCESSED_DIR)}")
    print("Files: {train,val,test}_{images,masks,types}.npy")


if __name__ == "__main__":
    main()
