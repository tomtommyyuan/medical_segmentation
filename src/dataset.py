"""
Datasets and the official PanNuke split definitions.

SPLITS holds the three-fold rotation the PanNuke benchmark is reported on: one
fold trains, a second validates, the third tests, and the three runs are
averaged. Every published mPQ/bPQ number is an average over these three, so
results are only comparable if the same rotation is used.

NucleiDataset yields the multi-task targets (nuclei map, distance maps, type
map) and, when asked, a second stain-jittered view of the same patch for the
consistency loss. BinaryDataset yields the plain binary mask the original
baselines were trained on, so those models can be retrained on the same folds
and appear in the same table.

Targets are built on the fly from the instance map, so geometric augmentation
is applied to the instance map alone and the distance maps are derived after.
Transforming precomputed distance maps would require negating and swapping
their channels as well.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from hover_targets import gen_targets
from stain_augment import augment_stain

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# The official PanNuke protocol: train on one fold, validate on a second, test
# on the third, then average the three splits. Never pool the folds.
SPLITS = {
    1: {"train": "fold1", "val": "fold2", "test": "fold3"},
    2: {"train": "fold2", "val": "fold3", "test": "fold1"},
    3: {"train": "fold3", "val": "fold1", "test": "fold2"},
}

# The 19 organs PanNuke covers. Order fixes the tissue embedding index.
TISSUE_TYPES = [
    "Adrenal_gland", "Bile-duct", "Bladder", "Breast", "Cervix", "Colon",
    "Esophagus", "HeadNeck", "Kidney", "Liver", "Lung", "Ovarian",
    "Pancreatic", "Prostate", "Skin", "Stomach", "Testis", "Thyroid", "Uterus",
]
TISSUE_TO_INDEX = {name: i for i, name in enumerate(TISSUE_TYPES)}

# Pathology foundation encoders are trained with ImageNet normalisation.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_fold(fold, data_dir=DATA_DIR, mmap=True):
    """
    Load one preprocessed fold.

    Memory-mapped by default: the DataLoader forks a copy of the dataset per
    worker, and eight workers each holding a materialised fold is several GB of
    duplicated pages for no benefit.

    Args:
        fold: "fold1", "fold2" or "fold3"
        data_dir: directory holding the preprocessed arrays
        mmap: memory-map the large arrays instead of reading them into RAM

    Returns:
        dict with images, insts, types, masks, tissues
    """
    mode = "r" if mmap else None
    return {
        "images": np.load(os.path.join(data_dir, f"{fold}_images.npy"), mmap_mode=mode),
        "insts": np.load(os.path.join(data_dir, f"{fold}_insts.npy"), mmap_mode=mode),
        "types": np.load(os.path.join(data_dir, f"{fold}_types.npy"), mmap_mode=mode),
        "masks": np.load(os.path.join(data_dir, f"{fold}_masks.npy"), mmap_mode=mode),
        "tissues": np.load(os.path.join(data_dir, f"{fold}_tissues.npy"), allow_pickle=True),
    }


def normalize_image(image, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Scale an 8-bit RGB patch to a normalised (3, H, W) float tensor."""
    x = np.asarray(image, dtype=np.float32) / 255.0
    x = (x - mean) / std
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def augment_geometry(image, inst, type_map, rng):
    """
    Apply a random dihedral transform to a patch and its labels.

    Histology has no canonical orientation, so all four rotations and both
    reflections are label preserving.
    """
    k = int(rng.integers(0, 4))
    if k:
        image = np.rot90(image, k, axes=(0, 1))
        inst = np.rot90(inst, k)
        type_map = np.rot90(type_map, k)

    if rng.random() < 0.5:
        image = image[:, ::-1]
        inst = inst[:, ::-1]
        type_map = type_map[:, ::-1]

    return (
        np.ascontiguousarray(image),
        np.ascontiguousarray(inst),
        np.ascontiguousarray(type_map),
    )


class _FoldDataset(Dataset):
    """Shared fold loading and per-worker RNG handling."""

    def __init__(self, fold, data_dir=DATA_DIR, augment=False, seed=None):
        self.fold = fold
        self.data = load_fold(fold, data_dir)
        self.augment = augment
        self.seed = seed
        self._rng = None

    def __len__(self):
        return len(self.data["images"])

    def rng(self):
        """
        Per-worker random generator.

        DataLoader workers are forked copies, so a generator built in __init__
        would hand every worker the same stream and every batch would contain
        the same augmentation repeated. torch reseeds each worker per epoch, so
        deriving from the worker seed keeps the streams distinct and the run
        reproducible.
        """
        if self._rng is None:
            worker = torch.utils.data.get_worker_info()
            if worker is not None:
                self._rng = np.random.default_rng(worker.seed % (2 ** 32))
            else:
                self._rng = np.random.default_rng(self.seed)
        return self._rng

    def tissue_index(self, idx):
        return TISSUE_TO_INDEX.get(str(self.data["tissues"][idx]), 0)


class NucleiDataset(_FoldDataset):
    """
    Multi-task dataset: image plus nuclei, distance and type targets.

    Args:
        fold: "fold1", "fold2" or "fold3"
        data_dir: directory holding the preprocessed arrays
        augment: apply dihedral and stain augmentation
        stain_views: also return a strongly stain-jittered second view, exactly
            registered with the first, for the consistency loss
        return_instances: also return the raw instance and type maps, needed to
            score PQ during evaluation
        seed: seed used when running without DataLoader workers
        sigma_scale, sigma_shift: stain jitter for the first view
        strong_sigma_scale, strong_sigma_shift: stain jitter for the second view
    """

    def __init__(self, fold, data_dir=DATA_DIR, augment=False, stain_views=False,
                 return_instances=False, seed=None,
                 sigma_scale=0.25, sigma_shift=0.05,
                 strong_sigma_scale=0.5, strong_sigma_shift=0.10,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__(fold, data_dir=data_dir, augment=augment, seed=seed)
        self.stain_views = stain_views
        self.return_instances = return_instances
        self.sigma_scale = sigma_scale
        self.sigma_shift = sigma_shift
        self.strong_sigma_scale = strong_sigma_scale
        self.strong_sigma_shift = strong_sigma_shift
        self.mean = mean
        self.std = std

    def __getitem__(self, idx):
        image = np.asarray(self.data["images"][idx])
        inst = np.asarray(self.data["insts"][idx]).astype(np.int32)
        type_map = np.asarray(self.data["types"][idx]).astype(np.uint8)

        rng = self.rng()

        if self.augment:
            image, inst, type_map = augment_geometry(image, inst, type_map, rng)

        # Geometry is fixed before colour, so both views share one label set.
        view = image
        if self.augment:
            view = augment_stain(image, self.sigma_scale, self.sigma_shift, rng=rng)

        np_map, hv_map, tp_map = gen_targets(inst, type_map)

        sample = {
            "image": normalize_image(view, self.mean, self.std),
            "np_map": torch.from_numpy(np_map),
            "hv_map": torch.from_numpy(hv_map),
            "tp_map": torch.from_numpy(tp_map),
            "tissue": self.tissue_index(idx),
        }

        if self.stain_views:
            strong = augment_stain(image, self.strong_sigma_scale,
                                   self.strong_sigma_shift, rng=rng)
            sample["image_strong"] = normalize_image(strong, self.mean, self.std)

        if self.return_instances:
            sample["inst"] = torch.from_numpy(inst)
            sample["type"] = torch.from_numpy(type_map.astype(np.int64))

        return sample


class BinaryDataset(_FoldDataset):
    """
    Binary nuclei masks, for the classical/CNN/U-Net baselines.

    Returns the same (image, mask) pair the original training loop expected, so
    those models can be retrained on the official folds without changing them.
    """

    def __init__(self, fold, data_dir=DATA_DIR, augment=False, seed=None,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD, normalize=False):
        super().__init__(fold, data_dir=data_dir, augment=augment, seed=seed)
        self.mean = mean
        self.std = std
        self.normalize = normalize

    def __getitem__(self, idx):
        image = np.asarray(self.data["images"][idx])
        mask = np.asarray(self.data["masks"][idx]).astype(np.uint8)

        if self.augment:
            rng = self.rng()
            image, mask, _ = augment_geometry(image, mask, mask, rng)
            image = augment_stain(image, rng=rng)

        if self.normalize:
            image_tensor = normalize_image(image, self.mean, self.std)
        else:
            # The original baselines were trained on plain [0, 1] scaling.
            x = np.asarray(image, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))

        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor
