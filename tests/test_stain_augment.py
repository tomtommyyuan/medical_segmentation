"""
Tests for optical density stain augmentation.

The property the consistency loss depends on is that augmentation is a
pixel-wise function of colour: two views of a patch must stay exactly
registered, so student and teacher outputs can be compared pixel to pixel.
"""

import numpy as np
from synthetic import make_image, make_patch

from stain_augment import (
    REFERENCE_STAIN_MATRIX,
    augment_stain,
    estimate_stain_matrix,
    normalize_stain,
    od_to_rgb,
    rgb_to_od,
)


def test_optical_density_round_trips():
    image = make_image(make_patch()[0])
    recovered = od_to_rgb(rgb_to_od(image))

    # Quantisation and the +1 offset cost about a count.
    assert np.abs(recovered.astype(int) - image.astype(int)).max() <= 2


def test_estimated_stain_vectors_are_unit_and_non_negative():
    image = make_image(make_patch()[0])
    stain_matrix = estimate_stain_matrix(image)

    assert stain_matrix is not None
    assert stain_matrix.shape == (3, 2)
    assert np.allclose(np.linalg.norm(stain_matrix, axis=0), 1.0, atol=1e-5)
    assert np.all(stain_matrix > -1e-6)
    # Haematoxylin is placed first and absorbs more red.
    assert stain_matrix[0, 0] > stain_matrix[0, 1]


def test_macenko_recovers_the_stains_the_patch_was_rendered_with():
    # The fixture is built in optical density space from the reference vectors,
    # so estimation should return those vectors back.
    image = make_image(make_patch()[0])
    stain_matrix = estimate_stain_matrix(image)

    assert np.abs(stain_matrix - REFERENCE_STAIN_MATRIX).max() < 0.06


def test_blank_patch_has_no_estimable_stains():
    blank = np.full((256, 256, 3), 245, dtype=np.uint8)

    assert estimate_stain_matrix(blank) is None


def test_single_stain_patch_is_rejected_as_ill_conditioned():
    # A patch carrying only haematoxylin collapses the optical density cloud
    # onto a line, so the two recovered vectors would be near-parallel and the
    # concentration solve unstable. Estimation must report failure instead.
    rng = np.random.default_rng(0)
    concentrations = np.stack([rng.uniform(0.3, 1.6, 4096), np.zeros(4096)])
    od = (REFERENCE_STAIN_MATRIX @ concentrations).T.reshape(64, 64, 3)
    image = np.clip(240.0 * np.exp(-od), 0, 255).astype(np.uint8)

    assert estimate_stain_matrix(image) is None


def test_augmentation_falls_back_rather_than_failing_on_a_blank_patch():
    blank = np.full((64, 64, 3), 245, dtype=np.uint8)
    augmented = augment_stain(blank, rng=np.random.default_rng(0))

    assert augmented.shape == blank.shape
    assert augmented.dtype == np.uint8


def test_augmentation_preserves_shape_and_dtype():
    image = make_image(make_patch()[0])
    augmented = augment_stain(image, rng=np.random.default_rng(0))

    assert augmented.shape == image.shape
    assert augmented.dtype == np.uint8


def test_augmentation_changes_colour():
    image = make_image(make_patch()[0])
    augmented = augment_stain(image, sigma_scale=0.3, sigma_shift=0.05,
                              rng=np.random.default_rng(0))

    assert np.abs(augmented.astype(int) - image.astype(int)).mean() > 1.0


def test_augmentation_is_pixel_wise_so_views_stay_registered():
    # Every pixel of one colour must map to a single output colour, and the
    # spatial layout must be untouched. This is what lets the consistency loss
    # compare two views without any warping.
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :] = (220, 180, 210)
    image[16:48, 16:48] = (110, 70, 160)

    augmented = augment_stain(image, rng=np.random.default_rng(3))

    inside = augmented[16:48, 16:48].reshape(-1, 3)
    outside = augmented[0:16, 0:16].reshape(-1, 3)

    assert len(np.unique(inside, axis=0)) == 1
    assert len(np.unique(outside, axis=0)) == 1
    assert not np.array_equal(inside[0], outside[0])


def test_augmentation_is_reproducible_from_a_seed():
    image = make_image(make_patch()[0])

    first = augment_stain(image, rng=np.random.default_rng(7))
    second = augment_stain(image, rng=np.random.default_rng(7))
    third = augment_stain(image, rng=np.random.default_rng(8))

    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_normalisation_pulls_differently_stained_views_together():
    # The end-to-end check on the optical density machinery: two heavily
    # re-stained copies of one patch should be far apart, and much closer once
    # both are mapped onto the reference stain.
    image = make_image(make_patch()[0])

    view_a = augment_stain(image, sigma_scale=0.4, sigma_shift=0.1,
                           rng=np.random.default_rng(1))
    view_b = augment_stain(image, sigma_scale=0.4, sigma_shift=0.1,
                           rng=np.random.default_rng(2))

    before = np.abs(view_a.astype(float) - view_b.astype(float)).mean()
    after = np.abs(
        normalize_stain(view_a).astype(float) - normalize_stain(view_b).astype(float)
    ).mean()

    assert before > 5.0, f"test setup: views differ by only {before:.2f}"
    assert after < before / 2.0, f"normalisation left {after:.2f} of {before:.2f}"


def test_augmentation_accepts_a_precomputed_stain_matrix():
    image = make_image(make_patch()[0])
    augmented = augment_stain(image, rng=np.random.default_rng(0),
                              stain_matrix=REFERENCE_STAIN_MATRIX)

    assert augmented.shape == image.shape
