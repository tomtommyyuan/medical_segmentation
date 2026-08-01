"""
Tests for the distance-map targets and the watershed decoding.

The important one is the round trip: if the network predicted the targets
perfectly, decoding must return the instances they were built from. Targets and
decoding are two halves of the same contract, and a mismatch between them caps
PQ no matter how good the model is.
"""

import numpy as np
from synthetic import make_patch

from hover_targets import gen_hv_map, gen_targets, get_bounding_box
from postprocess import assign_types, decode_batch, decode_instances
from pq_metrics import get_fast_pq, pq_per_image


def test_bounding_box_is_exclusive_on_the_upper_bound():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[5:10, 7:14] = 1

    assert get_bounding_box(mask) == (5, 10, 7, 14)


def test_hv_map_is_bounded_and_zero_on_background():
    inst, _ = make_patch()
    hv = gen_hv_map(inst)

    assert hv.shape == (2,) + inst.shape
    assert hv.dtype == np.float32
    assert hv.min() >= -1.0 and hv.max() <= 1.0
    assert np.allclose(hv[:, inst == 0], 0.0)


def test_hv_map_signs_point_away_from_the_centre():
    inst = np.zeros((64, 64), dtype=np.int32)
    inst[20:40, 20:40] = 1
    hv = gen_hv_map(inst)

    centre_y, centre_x = 30, 30

    # Horizontal map: negative left of centre, positive right of centre.
    assert hv[0, centre_y, 21] < 0
    assert hv[0, centre_y, 38] > 0
    # Vertical map: negative above centre, positive below.
    assert hv[1, 21, centre_x] < 0
    assert hv[1, 38, centre_x] > 0

    # Each side spans the full range.
    assert np.isclose(hv[0, centre_y, 20], -1.0, atol=1e-5)
    assert np.isclose(hv[0, centre_y, 39], 1.0, atol=1e-5)


def test_hv_map_is_empty_for_an_empty_patch():
    inst = np.zeros((64, 64), dtype=np.int32)
    hv = gen_hv_map(inst)

    assert np.allclose(hv, 0.0)


def test_perfect_targets_decode_back_to_the_instances():
    inst, _ = make_patch()
    np_map, hv_map, _ = gen_targets(inst, np.zeros_like(inst, dtype=np.uint8))

    decoded = decode_instances(np_map, hv_map)

    n_true = len(np.unique(inst)) - 1
    n_decoded = len(np.unique(decoded)) - 1
    assert n_decoded == n_true, f"decoded {n_decoded} nuclei, expected {n_true}"

    (_, _, pq), _ = get_fast_pq(inst, decoded)
    assert pq > 0.95, f"round-trip PQ {pq:.4f} is too low"


def test_touching_nuclei_are_split_rather_than_merged():
    # Two discs sharing a boundary: a plain connected-components pass would
    # return one object, which is exactly what the distance maps prevent.
    inst = np.zeros((128, 128), dtype=np.int32)
    yy, xx = np.ogrid[:128, :128]
    inst[(yy - 64) ** 2 + (xx - 52) ** 2 <= 196] = 1
    inst[(yy - 64) ** 2 + (xx - 76) ** 2 <= 196] = 2

    np_map = (inst > 0).astype(np.float32)
    hv_map = gen_hv_map(inst)

    from scipy.ndimage import label

    assert label(np_map)[1] == 1, "test setup: the two discs should be connected"
    assert len(np.unique(decode_instances(np_map, hv_map))) - 1 == 2


def test_decoding_an_empty_prediction_returns_no_instances():
    np_map = np.zeros((64, 64), dtype=np.float32)
    hv_map = np.zeros((2, 64, 64), dtype=np.float32)

    assert decode_instances(np_map, hv_map).max() == 0


def test_assign_types_takes_a_majority_vote_per_instance():
    inst = np.zeros((32, 32), dtype=np.int32)
    inst[0:10, 0:10] = 1

    # 70 pixels vote class 2, 30 vote class 3.
    per_pixel = np.zeros((32, 32), dtype=np.uint8)
    per_pixel[0:7, 0:10] = 2
    per_pixel[7:10, 0:10] = 3

    assigned = assign_types(inst, per_pixel)

    assert set(np.unique(assigned[inst == 1]).tolist()) == {2}
    assert assigned[inst == 0].max() == 0


def test_assign_types_falls_back_when_every_pixel_votes_background():
    inst = np.zeros((32, 32), dtype=np.int32)
    inst[0:10, 0:10] = 1

    per_pixel = np.zeros((32, 32), dtype=np.uint8)
    per_pixel[0:3, 0:3] = 4  # a background majority with a minority real class

    assigned = assign_types(inst, per_pixel)

    assert set(np.unique(assigned[inst == 1]).tolist()) == {4}


def test_end_to_end_perfect_prediction_scores_pq_one():
    inst, type_map = make_patch()
    np_map, hv_map, _ = gen_targets(inst, type_map)

    inst_pred, type_pred = decode_batch(
        np_map[None], hv_map[None], type_map[None].astype(np.uint8)
    )

    bpq, class_pq = pq_per_image(inst, type_map, inst_pred[0], type_pred[0])

    assert bpq > 0.95, f"binary PQ {bpq:.4f}"
    present = ~np.isnan(class_pq)
    assert np.all(class_pq[present] > 0.9), f"class PQ {class_pq}"
