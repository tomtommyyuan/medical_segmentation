"""
Tests for dihedral test-time augmentation.

Correctness is stated against distance maps regenerated from transformed
instance maps, rather than against the implementation's own arithmetic. If a
perfect model saw the transformed patch, it would emit exactly those maps, so
inverting them has to return the maps of the untransformed patch.

A naive inversion that moves the pixels but leaves the channels alone is
checked to fail, so the tests have teeth: without that, the sign correction
could be deleted and everything here would still pass.
"""

import numpy as np
import pytest
import torch
from synthetic import make_patch

from hover_targets import gen_hv_map
from postprocess import decode_instances
from pq_metrics import get_fast_pq
from tta import (
    TRANSFORMS,
    inverse_transform,
    predict_tta,
    transform_image,
)


def targets_for(inst):
    """Distance maps a perfect model would emit, as a batched tensor."""
    return torch.from_numpy(gen_hv_map(np.ascontiguousarray(inst))).unsqueeze(0)


def transform_labels(inst, k, flip):
    """Apply the same dihedral transform to a label map."""
    tensor = torch.from_numpy(inst.astype(np.int64)).unsqueeze(0).unsqueeze(0)
    return transform_image(tensor, k, flip)[0, 0].numpy()


def test_there_are_eight_distinct_transforms():
    assert len(TRANSFORMS) == 8
    assert len(set(TRANSFORMS)) == 8


@pytest.mark.parametrize("k,flip", TRANSFORMS)
def test_inverting_a_perfect_prediction_recovers_the_original_maps(k, flip):
    inst, _ = make_patch()

    original = targets_for(inst)
    # What a perfect model would predict on the transformed patch.
    predicted = targets_for(transform_labels(inst, k, flip))

    outputs = {
        "np": torch.zeros(1, 1, 256, 256),
        "hv": predicted,
        "tp": torch.zeros(1, 6, 256, 256),
    }
    recovered = inverse_transform(outputs, k, flip)["hv"]

    assert torch.allclose(recovered, original, atol=1e-5), (
        f"k={k} flip={flip}: max error {float((recovered - original).abs().max()):.4f}"
    )


@pytest.mark.parametrize("k,flip", TRANSFORMS)
def test_the_nuclei_branch_is_geometrically_inverted(k, flip):
    inst, _ = make_patch()
    np_map = torch.from_numpy((inst > 0).astype(np.float32))[None, None]

    outputs = {
        "np": transform_image(np_map, k, flip),
        "hv": torch.zeros(1, 2, 256, 256),
        "tp": torch.zeros(1, 6, 256, 256),
    }

    assert torch.allclose(inverse_transform(outputs, k, flip)["np"], np_map)


def test_ignoring_the_sign_correction_breaks_the_distance_maps():
    # A mirror is the cheapest counter-example: the horizontal map must be
    # negated, and simply flipping it back leaves every offset pointing the
    # wrong way.
    inst, _ = make_patch()

    original = targets_for(inst)
    predicted = targets_for(transform_labels(inst, 0, True))

    naive = torch.flip(predicted, dims=(-1,))
    correct = inverse_transform(
        {"np": torch.zeros(1, 1, 256, 256), "hv": predicted,
         "tp": torch.zeros(1, 6, 256, 256)}, 0, True
    )["hv"]

    assert torch.allclose(correct, original, atol=1e-5)
    assert not torch.allclose(naive, original, atol=1e-2)


def test_averaging_wrongly_inverted_maps_destroys_the_watershed_ridges():
    # The failure this module exists to prevent. Averaging the eight views
    # without the sign correction cancels the distance maps out, the ridges
    # between touching nuclei vanish, and every touching pair merges.
    inst, _ = make_patch()
    np_map = (inst > 0).astype(np.float32)

    correct_sum = torch.zeros(1, 2, 256, 256)
    naive_sum = torch.zeros(1, 2, 256, 256)

    for k, flip in TRANSFORMS:
        predicted = targets_for(transform_labels(inst, k, flip))

        correct_sum += inverse_transform(
            {"np": torch.zeros(1, 1, 256, 256), "hv": predicted,
             "tp": torch.zeros(1, 6, 256, 256)}, k, flip
        )["hv"]

        # Geometry undone, channels and signs left alone.
        undone = predicted
        if flip:
            undone = torch.flip(undone, dims=(-1,))
        if k:
            undone = torch.rot90(undone, -k, dims=(-2, -1))
        naive_sum += undone

    n_true = len(np.unique(inst)) - 1

    correct = decode_instances(np_map, (correct_sum / 8)[0].numpy())
    naive = decode_instances(np_map, (naive_sum / 8)[0].numpy())

    assert len(np.unique(correct)) - 1 == n_true
    assert len(np.unique(naive)) - 1 < n_true, "naive inversion should lose nuclei"
    assert get_fast_pq(inst, correct)[0][2] > get_fast_pq(inst, naive)[0][2]


def test_tta_of_an_equivariant_model_matches_the_single_view():
    # A model that ignores its input is trivially equivariant, so averaging the
    # eight views must reproduce the single-view answer exactly.
    class ConstantModel(torch.nn.Module):
        def forward(self, image, tissue=None):
            batch, _, height, width = image.shape
            return {
                "np": torch.full((batch, 1, height, width), 2.0),
                "hv": torch.zeros(batch, 2, height, width),
                "tp": torch.zeros(batch, 6, height, width),
            }

    averaged = predict_tta(ConstantModel(), torch.randn(1, 3, 64, 64))

    assert torch.allclose(averaged["np"], torch.sigmoid(torch.tensor(2.0)).expand(1, 1, 64, 64))
    assert torch.allclose(averaged["hv"], torch.zeros(1, 2, 64, 64))
    assert torch.allclose(averaged["tp"].sum(dim=1), torch.ones(1, 64, 64), atol=1e-5)


def test_tta_returns_probabilities_not_logits():
    class ConstantModel(torch.nn.Module):
        def forward(self, image, tissue=None):
            batch, _, height, width = image.shape
            return {
                "np": torch.full((batch, 1, height, width), 5.0),
                "hv": torch.zeros(batch, 2, height, width),
                "tp": torch.randn(batch, 6, height, width),
            }

    out = predict_tta(ConstantModel(), torch.randn(2, 3, 32, 32))

    assert float(out["np"].min()) >= 0.0 and float(out["np"].max()) <= 1.0
    assert torch.allclose(out["tp"].sum(dim=1), torch.ones(2, 32, 32), atol=1e-5)
