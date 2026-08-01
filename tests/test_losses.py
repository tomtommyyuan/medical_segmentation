"""
Tests for the multi-task loss.

The gradient term's axis convention gets its own test. The watershed reads the
horizontal map's derivative along x and the vertical map's along y; supervising
the other pairing would train a signal decoding never looks at, and nothing
downstream would fail loudly.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hover_targets import gen_hv_map
from losses import (
    ChromaLoss,
    LogitAdjustedCrossEntropy,
    compute_class_prior,
    consistency_loss,
    dice_loss,
    gradient_hv,
    msge_loss,
    sigmoid_rampup,
    sobel_kernels,
)
from preprocess import NUM_TYPES


def touching_pair(axis):
    """Two 28px discs sharing a boundary, side by side or stacked."""
    inst = np.zeros((128, 128), dtype=np.int32)
    yy, xx = np.ogrid[:128, :128]

    if axis == "x":
        centres = [(64, 52), (64, 76)]
    else:
        centres = [(52, 64), (76, 64)]

    for label, (cy, cx) in enumerate(centres, start=1):
        inst[(yy - cy) ** 2 + (xx - cx) ** 2 <= 196] = label

    return inst


def test_sobel_kernels_differentiate_along_the_expected_axes():
    kernel_x, kernel_y = sobel_kernels(5)

    # kernel_x varies across columns and is constant down a column.
    assert torch.allclose(kernel_x[:, 2], torch.zeros(5), atol=1e-6)
    assert kernel_x[2, 0] < 0 and kernel_x[2, 4] > 0
    # kernel_y is the transpose.
    assert torch.allclose(kernel_y, kernel_x.T, atol=1e-6)


def test_horizontal_map_gradient_ridges_on_a_side_by_side_join():
    inst = touching_pair("x")
    gradients = gradient_hv(torch.from_numpy(gen_hv_map(inst)).unsqueeze(0))[0].abs()

    # The join is the column midway between the two centres. The horizontal
    # channel must ridge there; the vertical channel's largest response is a
    # disc edge somewhere else entirely.
    assert abs(int(torch.argmax(gradients[0].sum(dim=0))) - 64) <= 2
    assert abs(int(torch.argmax(gradients[1].sum(dim=1))) - 64) > 5
    assert gradients[0].max() > gradients[1].max()


def test_vertical_map_gradient_ridges_on_a_stacked_join():
    inst = touching_pair("y")
    gradients = gradient_hv(torch.from_numpy(gen_hv_map(inst)).unsqueeze(0))[0].abs()

    assert abs(int(torch.argmax(gradients[1].sum(dim=1))) - 64) <= 2
    assert abs(int(torch.argmax(gradients[0].sum(dim=0))) - 64) > 5
    assert gradients[1].max() > gradients[0].max()


def test_transposing_the_patch_swaps_the_two_gradient_channels():
    # The sharpest statement that the axes are not crossed: a side-by-side pair
    # transposed is a stacked pair, and its gradients must be the transposed
    # channel swap of the original. A swapped kernel pairing breaks this.
    hv = torch.from_numpy(gen_hv_map(touching_pair("x"))).unsqueeze(0)

    # Transposing the patch swaps the roles of x and y, so the horizontal map
    # becomes the vertical one and vice versa.
    transposed = torch.stack([hv[0, 1].T, hv[0, 0].T]).unsqueeze(0)

    original = gradient_hv(hv)[0].abs()
    swapped = gradient_hv(transposed)[0].abs()

    assert torch.allclose(swapped[1], original[0].T, atol=1e-5)
    assert torch.allclose(swapped[0], original[1].T, atol=1e-5)


def test_msge_is_zero_for_a_perfect_prediction():
    inst = touching_pair("x")
    hv = torch.from_numpy(gen_hv_map(inst)).unsqueeze(0)
    focus = torch.from_numpy((inst > 0).astype(np.float32)).unsqueeze(0)

    assert float(msge_loss(hv, hv, focus)) == pytest.approx(0.0, abs=1e-8)


def test_msge_penalises_a_blurred_prediction():
    inst = touching_pair("x")
    hv = torch.from_numpy(gen_hv_map(inst)).unsqueeze(0)
    focus = torch.from_numpy((inst > 0).astype(np.float32)).unsqueeze(0)

    kernel = torch.ones(2, 1, 9, 9) / 81.0
    blurred = F.conv2d(hv, kernel, padding=4, groups=2)

    # Blurring barely moves the maps themselves but flattens the ridge the
    # watershed needs, which is exactly the failure the gradient term catches.
    assert float(msge_loss(blurred, hv, focus)) > 10 * float(F.mse_loss(blurred, hv))


def test_dice_loss_is_zero_for_a_perfect_prediction():
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 8:24, 8:24] = 1.0

    assert float(dice_loss(target, target)) == pytest.approx(0.0, abs=1e-3)


def test_dice_loss_is_maximal_for_an_inverted_prediction():
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 8:24, 8:24] = 1.0

    assert float(dice_loss(1.0 - target, target)) == pytest.approx(1.0, abs=1e-2)


def test_logit_adjustment_with_zero_tau_is_plain_cross_entropy():
    prior = np.array([0.9, 0.05, 0.03, 0.01, 0.005, 0.005])
    criterion = LogitAdjustedCrossEntropy(prior, tau=0.0)

    logits = torch.randn(2, 6, 8, 8)
    target = torch.randint(0, 6, (2, 8, 8))

    assert float(criterion(logits, target)) == pytest.approx(
        float(F.cross_entropy(logits, target)), abs=1e-5
    )


def test_logit_adjustment_pushes_harder_on_rare_classes():
    # Under plain cross entropy with uninformative logits, every class costs
    # the same. Adjustment must tilt that: getting a rare class wrong has to
    # cost more, relative to the common one, than it did before.
    prior = np.array([0.90, 0.06, 0.02, 0.015, 0.003, 0.002])
    criterion = LogitAdjustedCrossEntropy(prior, tau=1.0)

    logits = torch.zeros(1, 6, 4, 4)
    rare = torch.full((1, 4, 4), 4, dtype=torch.long)      # Dead, 0.3% of pixels
    common = torch.full((1, 4, 4), 0, dtype=torch.long)    # background, 90%

    plain_ratio = float(F.cross_entropy(logits, rare)) / float(F.cross_entropy(logits, common))
    adjusted_ratio = float(criterion(logits, rare)) / float(criterion(logits, common))

    assert plain_ratio == pytest.approx(1.0, abs=1e-5)
    assert adjusted_ratio > 10 * plain_ratio


def test_logit_adjustment_strength_scales_with_tau():
    prior = np.array([0.90, 0.06, 0.02, 0.015, 0.003, 0.002])
    logits = torch.zeros(1, 6, 4, 4)
    rare = torch.full((1, 4, 4), 4, dtype=torch.long)

    losses = [float(LogitAdjustedCrossEntropy(prior, tau=t)(logits, rare))
              for t in (0.0, 0.5, 1.0)]

    assert losses[0] < losses[1] < losses[2]


def test_class_prior_counts_pixels_and_normalises():
    type_maps = np.zeros((4, 10, 10), dtype=np.uint8)
    type_maps[:, :5, :] = 1  # half the pixels are class 1

    prior = compute_class_prior(type_maps)

    assert len(prior) == NUM_TYPES + 1
    assert prior.sum() == pytest.approx(1.0)
    assert prior[0] == pytest.approx(0.5)
    assert prior[1] == pytest.approx(0.5)


def test_class_prior_handles_chunking():
    rng = np.random.default_rng(0)
    type_maps = rng.integers(0, NUM_TYPES + 1, size=(9, 8, 8)).astype(np.uint8)

    whole = compute_class_prior(type_maps, chunk=100)
    chunked = compute_class_prior(type_maps, chunk=2)

    assert np.allclose(whole, chunked)


def test_consistency_is_zero_when_the_views_agree():
    outputs = {
        "np": torch.randn(2, 1, 16, 16),
        "hv": torch.randn(2, 2, 16, 16),
        "tp": torch.randn(2, 6, 16, 16),
    }

    assert float(consistency_loss(outputs, outputs)) == pytest.approx(0.0, abs=1e-6)


def test_consistency_is_positive_when_the_views_disagree():
    student = {
        "np": torch.randn(2, 1, 16, 16),
        "hv": torch.randn(2, 2, 16, 16),
        "tp": torch.randn(2, 6, 16, 16),
    }
    teacher = {k: torch.randn_like(v) for k, v in student.items()}

    assert float(consistency_loss(student, teacher)) > 0.0


def test_consistency_does_not_backpropagate_into_the_teacher():
    student = {
        "np": torch.randn(1, 1, 8, 8, requires_grad=True),
        "hv": torch.randn(1, 2, 8, 8, requires_grad=True),
        "tp": torch.randn(1, 6, 8, 8, requires_grad=True),
    }
    teacher = {k: torch.randn_like(v).requires_grad_(True) for k, v in student.items()}

    consistency_loss(student, teacher).backward()

    assert all(v.grad is None for v in teacher.values())
    assert all(v.grad is not None for v in student.values())


def test_rampup_rises_monotonically_from_zero_to_one():
    values = [sigmoid_rampup(step, 100) for step in range(0, 101, 10)]

    assert values[0] == pytest.approx(0.0, abs=1e-2)
    assert values[-1] == pytest.approx(1.0, abs=1e-6)
    assert all(b >= a for a, b in zip(values, values[1:]))
    assert sigmoid_rampup(5, 0) == 1.0


def test_full_loss_returns_a_scalar_and_its_terms():
    batch, size = 2, 32
    outputs = {
        "np": torch.randn(batch, 1, size, size, requires_grad=True),
        "hv": torch.randn(batch, 2, size, size, requires_grad=True),
        "tp": torch.randn(batch, NUM_TYPES + 1, size, size, requires_grad=True),
    }
    targets = {
        "np_map": torch.randint(0, 2, (batch, size, size)).float(),
        "hv_map": torch.rand(batch, 2, size, size) * 2 - 1,
        "tp_map": torch.randint(0, NUM_TYPES + 1, (batch, size, size)),
    }

    total, terms = ChromaLoss()(outputs, targets)
    total.backward()

    assert total.dim() == 0 and torch.isfinite(total)
    assert set(terms) == {"np_bce", "np_dice", "hv_mse", "hv_msge", "tp_ce", "tp_dice"}
    assert outputs["hv"].grad is not None


def test_full_loss_is_lower_for_a_better_prediction():
    batch, size = 2, 32
    targets = {
        "np_map": torch.zeros(batch, size, size),
        "hv_map": torch.zeros(batch, 2, size, size),
        "tp_map": torch.zeros(batch, size, size, dtype=torch.long),
    }
    targets["np_map"][:, 8:24, 8:24] = 1.0
    targets["tp_map"][:, 8:24, 8:24] = 1

    def outputs_for(scale):
        logits = torch.zeros(batch, NUM_TYPES + 1, size, size)
        logits[:, 0] = (1 - targets["np_map"]) * scale
        logits[:, 1] = targets["np_map"] * scale
        return {
            "np": ((targets["np_map"] * 2 - 1) * scale).unsqueeze(1),
            "hv": targets["hv_map"].clone(),
            "tp": logits,
        }

    criterion = ChromaLoss()
    good = float(criterion(outputs_for(8.0), targets)[0])
    poor = float(criterion(outputs_for(0.1), targets)[0])

    assert good < poor
