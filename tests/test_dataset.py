"""
Tests for the fold definitions and the datasets built on them.

Covers the two things that would silently corrupt training: a split rotation
that reuses a fold in two roles, and geometric augmentation that moves the
image without moving its labels.
"""

import numpy as np
import pytest
from synthetic import make_image, make_patch

from dataset import (
    SPLITS,
    TISSUE_TO_INDEX,
    TISSUE_TYPES,
    BinaryDataset,
    NucleiDataset,
    augment_geometry,
    normalize_image,
)


@pytest.fixture
def fold_dir(tmp_path):
    """Write two tiny synthetic folds in the preprocessed layout."""
    inst, type_map = make_patch()
    image = make_image(inst)

    for fold, tissues in [("fold1", ["Breast", "Colon"]), ("fold2", ["Skin", "Breast"])]:
        n = len(tissues)
        np.save(tmp_path / f"{fold}_images.npy", np.stack([image] * n))
        np.save(tmp_path / f"{fold}_insts.npy", np.stack([inst] * n).astype(np.int16))
        np.save(tmp_path / f"{fold}_types.npy", np.stack([type_map] * n))
        np.save(tmp_path / f"{fold}_masks.npy", np.stack([(inst > 0).astype(np.uint8)] * n))
        np.save(tmp_path / f"{fold}_tissues.npy", np.array(tissues, dtype=object))

    return str(tmp_path)


def test_every_split_uses_each_fold_in_exactly_one_role():
    for split, roles in SPLITS.items():
        assert sorted(roles.values()) == ["fold1", "fold2", "fold3"], f"split {split} reuses a fold"


def test_every_fold_is_tested_on_exactly_once_across_the_rotation():
    tested = [roles["test"] for roles in SPLITS.values()]
    assert sorted(tested) == ["fold1", "fold2", "fold3"]


def test_tissue_list_matches_pannuke():
    assert len(TISSUE_TYPES) == 19
    assert len(TISSUE_TO_INDEX) == 19
    assert TISSUE_TO_INDEX["Breast"] == TISSUE_TYPES.index("Breast")


def test_normalize_image_produces_channel_first_float():
    image = make_image(make_patch()[0])
    tensor = normalize_image(image)

    assert tensor.shape == (3, 256, 256)
    assert tensor.dtype.is_floating_point


def test_geometric_augmentation_keeps_image_and_labels_aligned():
    inst, type_map = make_patch()
    image = make_image(inst)
    rng = np.random.default_rng(0)

    # Nuclei are darker than stroma, so foreground can be recovered from the
    # image alone and checked against the transformed instance map.
    def foreground_from_image(img):
        return img.mean(axis=2) < img.mean() - 10

    for _ in range(8):
        aug_image, aug_inst, aug_type = augment_geometry(image, inst, type_map, rng)

        overlap = (foreground_from_image(aug_image) & (aug_inst > 0)).sum()
        expected = foreground_from_image(aug_image).sum()

        assert overlap / max(expected, 1) > 0.95, "image and instance map drifted apart"
        assert np.array_equal(aug_inst > 0, aug_type > 0)


def test_geometric_augmentation_preserves_instance_count():
    inst, type_map = make_patch()
    image = make_image(inst)
    rng = np.random.default_rng(1)

    for _ in range(8):
        _, aug_inst, _ = augment_geometry(image, inst, type_map, rng)
        assert len(np.unique(aug_inst)) == len(np.unique(inst))


def test_nuclei_dataset_yields_the_multi_task_targets(fold_dir):
    dataset = NucleiDataset("fold1", data_dir=fold_dir)
    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["image"].shape == (3, 256, 256)
    assert sample["np_map"].shape == (256, 256)
    assert sample["hv_map"].shape == (2, 256, 256)
    assert sample["tp_map"].shape == (256, 256)
    assert sample["tp_map"].dtype == np.int64 or str(sample["tp_map"].dtype) == "torch.int64"
    assert sample["tissue"] == TISSUE_TO_INDEX["Breast"]
    assert "image_strong" not in sample


def test_stain_views_share_geometry_and_differ_in_colour(fold_dir):
    dataset = NucleiDataset("fold1", data_dir=fold_dir, augment=True,
                            stain_views=True, seed=0)
    sample = dataset[0]

    assert sample["image_strong"].shape == sample["image"].shape
    assert not np.allclose(sample["image"].numpy(), sample["image_strong"].numpy())

    # One target set serves both views, which is only sound because stain
    # augmentation never moves a pixel.
    assert sample["np_map"].shape == (256, 256)
    assert float(sample["np_map"].sum()) > 0


def test_return_instances_gives_the_maps_pq_needs(fold_dir):
    dataset = NucleiDataset("fold2", data_dir=fold_dir, return_instances=True)
    sample = dataset[0]

    assert sample["inst"].shape == (256, 256)
    assert sample["type"].shape == (256, 256)
    assert int(sample["inst"].max()) == 9


def test_binary_dataset_matches_the_original_baseline_interface(fold_dir):
    dataset = BinaryDataset("fold1", data_dir=fold_dir)
    image, mask = dataset[0]

    assert image.shape == (3, 256, 256)
    assert mask.shape == (1, 256, 256)
    assert float(image.max()) <= 1.0
    assert set(np.unique(mask.numpy()).tolist()) <= {0.0, 1.0}


def test_unknown_tissue_falls_back_rather_than_raising(fold_dir, tmp_path):
    np.save(tmp_path / "fold1_tissues.npy", np.array(["Not_A_Tissue", "Breast"], dtype=object))
    dataset = NucleiDataset("fold1", data_dir=fold_dir)

    assert dataset[0]["tissue"] == 0
