"""
Tests for the Panoptic Quality implementation.

Every case here is hand-computable, because a silently wrong PQ makes every
downstream number in the repo fiction.
"""

import numpy as np

from pq_metrics import aggregate_pq, get_fast_pq, pq_per_image


def square(canvas, y0, y1, x0, x1, value):
    canvas[y0:y1, x0:x1] = value
    return canvas


def test_perfect_prediction_scores_one():
    true = np.zeros((64, 64), dtype=np.int32)
    square(true, 10, 20, 10, 20, 1)
    square(true, 40, 50, 40, 50, 2)

    (dq, sq, pq), _ = get_fast_pq(true, true.copy())

    assert np.isclose(dq, 1.0, atol=1e-5)
    assert np.isclose(sq, 1.0, atol=1e-5)
    assert np.isclose(pq, 1.0, atol=1e-5)


def test_empty_prediction_scores_zero():
    true = np.zeros((64, 64), dtype=np.int32)
    square(true, 10, 20, 10, 20, 1)
    pred = np.zeros((64, 64), dtype=np.int32)

    (dq, sq, pq), _ = get_fast_pq(true, pred)

    assert np.isclose(pq, 0.0, atol=1e-5)
    assert np.isclose(dq, 0.0, atol=1e-5)


def test_one_hit_one_miss_one_false_positive():
    # tp = 1 (exact), fn = 1, fp = 1  ->  dq = 1 / (1 + 0.5 + 0.5) = 0.5
    true = np.zeros((128, 128), dtype=np.int32)
    square(true, 10, 20, 10, 20, 1)
    square(true, 50, 60, 50, 60, 2)

    pred = np.zeros((128, 128), dtype=np.int32)
    square(pred, 10, 20, 10, 20, 1)
    square(pred, 100, 110, 100, 110, 2)

    (dq, sq, pq), (paired_true, paired_pred, unpaired_true, unpaired_pred) = get_fast_pq(true, pred)

    assert len(paired_true) == 1
    assert len(unpaired_true) == 1
    assert len(unpaired_pred) == 1
    assert np.isclose(dq, 0.5, atol=1e-5)
    assert np.isclose(sq, 1.0, atol=1e-5)
    assert np.isclose(pq, 0.5, atol=1e-5)


def test_segmentation_quality_uses_iou():
    # 10x10 boxes offset by 2 columns: inter = 80, union = 120, iou = 2/3
    true = np.zeros((64, 64), dtype=np.int32)
    square(true, 0, 10, 0, 10, 1)

    pred = np.zeros((64, 64), dtype=np.int32)
    square(pred, 0, 10, 2, 12, 1)

    (dq, sq, pq), _ = get_fast_pq(true, pred)

    assert np.isclose(sq, 80.0 / 120.0, atol=1e-5)
    assert np.isclose(dq, 1.0, atol=1e-5)
    assert np.isclose(pq, 80.0 / 120.0, atol=1e-5)


def test_split_nucleus_matches_nothing():
    # One ground truth split into two halves: each half has IoU exactly 0.5,
    # which is not > 0.5, so nothing matches.
    true = np.zeros((64, 64), dtype=np.int32)
    square(true, 0, 10, 0, 20, 1)

    pred = np.zeros((64, 64), dtype=np.int32)
    square(pred, 0, 10, 0, 10, 1)
    square(pred, 0, 10, 10, 20, 2)

    (dq, sq, pq), (paired_true, _, unpaired_true, unpaired_pred) = get_fast_pq(true, pred)

    assert len(paired_true) == 0
    assert len(unpaired_true) == 1
    assert len(unpaired_pred) == 2
    assert np.isclose(pq, 0.0, atol=1e-5)


def test_non_contiguous_ids_are_handled():
    # PanNuke IDs are sparse; PQ must not depend on the numbering.
    true = np.zeros((64, 64), dtype=np.int32)
    square(true, 10, 20, 10, 20, 7)
    square(true, 40, 50, 40, 50, 913)

    pred = np.zeros((64, 64), dtype=np.int32)
    square(pred, 10, 20, 10, 20, 42)
    square(pred, 40, 50, 40, 50, 5)

    (_, _, pq), _ = get_fast_pq(true, pred)

    assert np.isclose(pq, 1.0, atol=1e-5)


def test_prediction_with_no_background_is_handled():
    # An over-confident model can label every pixel, leaving no id 0 at all.
    # Looking masks up by position rather than by id breaks here, and it breaks
    # during validation, hours into a run.
    true = np.zeros((32, 32), dtype=np.int32)
    square(true, 0, 16, 0, 32, 1)
    square(true, 16, 32, 0, 32, 2)

    pred = np.zeros((32, 32), dtype=np.int32)
    square(pred, 0, 16, 0, 32, 1)
    square(pred, 16, 32, 0, 32, 2)

    assert pred.min() > 0, "test setup: the prediction should cover every pixel"

    (_, _, pq), _ = get_fast_pq(true, pred)
    assert np.isclose(pq, 1.0, atol=1e-5)


def test_ids_starting_above_one_are_handled():
    true = np.zeros((32, 32), dtype=np.int32)
    square(true, 4, 12, 4, 12, 5)

    pred = np.zeros((32, 32), dtype=np.int32)
    square(pred, 4, 12, 4, 12, 900)

    (_, _, pq), (paired_true, paired_pred, _, _) = get_fast_pq(true, pred)

    assert np.isclose(pq, 1.0, atol=1e-5)
    assert list(paired_true) == [5]
    assert list(paired_pred) == [900]


def test_pq_per_image_survives_a_full_coverage_prediction():
    true_inst = np.zeros((32, 32), dtype=np.int32)
    true_type = np.zeros((32, 32), dtype=np.uint8)
    square(true_inst, 4, 12, 4, 12, 1)
    square(true_type, 4, 12, 4, 12, 1)

    pred_inst = np.ones((32, 32), dtype=np.int32)
    pred_type = np.ones((32, 32), dtype=np.uint8)

    bpq, class_pq = pq_per_image(true_inst, true_type, pred_inst, pred_type)

    assert np.isfinite(bpq)
    assert np.isfinite(class_pq[0])


def test_class_absent_from_both_is_nan_not_zero():
    true_inst = np.zeros((64, 64), dtype=np.int32)
    true_type = np.zeros((64, 64), dtype=np.uint8)
    square(true_inst, 10, 20, 10, 20, 1)
    square(true_type, 10, 20, 10, 20, 1)  # Neoplastic only

    bpq, class_pq = pq_per_image(true_inst, true_type, true_inst.copy(), true_type.copy())

    assert np.isclose(bpq, 1.0, atol=1e-5)
    assert np.isclose(class_pq[0], 1.0, atol=1e-5)
    # Classes 2-5 appear in neither map, so they are skipped rather than scored 0.
    assert np.all(np.isnan(class_pq[1:]))


def test_class_present_in_only_one_map_scores_zero():
    true_inst = np.zeros((64, 64), dtype=np.int32)
    true_type = np.zeros((64, 64), dtype=np.uint8)
    square(true_inst, 10, 20, 10, 20, 1)
    square(true_type, 10, 20, 10, 20, 1)

    # Same shape, wrong class: perfect binary PQ, zero for both classes.
    pred_inst = true_inst.copy()
    pred_type = np.zeros((64, 64), dtype=np.uint8)
    square(pred_type, 10, 20, 10, 20, 2)

    bpq, class_pq = pq_per_image(true_inst, true_type, pred_inst, pred_type)

    assert np.isclose(bpq, 1.0, atol=1e-5)
    assert np.isclose(class_pq[0], 0.0, atol=1e-5)
    assert np.isclose(class_pq[1], 0.0, atol=1e-5)
    assert np.all(np.isnan(class_pq[2:]))


def test_aggregate_averages_over_tissues_not_images():
    # 100 images of one tissue scoring 0 and 1 image of another scoring 1
    # must give 0.5, not 1/101.
    n_bad = 100
    bpq = [0.0] * n_bad + [1.0]
    class_pq = [[0.0, np.nan, np.nan, np.nan, np.nan]] * n_bad + [[1.0, np.nan, np.nan, np.nan, np.nan]]
    tissues = ["Breast"] * n_bad + ["Testis"]

    results = aggregate_pq(bpq, class_pq, tissues)

    assert np.isclose(results["bpq"], 0.5, atol=1e-6)
    assert np.isclose(results["mpq"], 0.5, atol=1e-6)
    assert results["per_tissue"]["Breast"]["n"] == n_bad
    assert np.isclose(results["per_tissue"]["Testis"]["bpq"], 1.0, atol=1e-6)


def test_aggregate_ignores_nan_classes_in_per_image_mpq():
    # An image where only one class is present should score that class's PQ,
    # not that PQ divided by 5.
    class_pq = [[0.8, np.nan, np.nan, np.nan, np.nan]]
    results = aggregate_pq([0.8], class_pq, ["Breast"])

    assert np.isclose(results["mpq_per_image"][0], 0.8, atol=1e-6)
    assert np.isclose(results["mpq"], 0.8, atol=1e-6)
