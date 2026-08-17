"""
Panoptic Quality metrics for PanNuke, following the official evaluation.

PQ = DQ * SQ, where DQ (detection quality) is the F1 over instances matched at
IoU > 0.5 and SQ (segmentation quality) is the mean IoU of those matches. At an
IoU threshold above 0.5 the matching is unique, so no assignment solver is
needed; a Hungarian match is used for lower thresholds.

The two headline numbers are:
    bPQ - PQ over all nuclei, ignoring class
    mPQ - PQ computed per nucleus class, then averaged over the 5 classes

Both are averaged per image, then within each tissue, then over the 19 tissue
types, so that a large tissue cannot dominate the score. This is the same
aggregation used by the PanNuke baselines, HoVer-Net and CellViT, and is what
makes the numbers here directly comparable to the published tables.

A class that is absent from both the ground truth and the prediction of an
image contributes NaN (it is skipped), not zero. A class present in only one of
the two contributes zero. Getting this wrong shifts mPQ by several points.

Reference: Graham et al., "HoVer-Net", Medical Image Analysis 2019;
Gamper et al., "PanNuke Dataset Extension, Insights and Baselines", 2020.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from preprocess import TYPE_NAMES, remap_label

NUM_TYPES = len(TYPE_NAMES)


def get_fast_pq(true, pred, match_iou=0.5):
    """
    Compute DQ, SQ and PQ between two instance maps.

    Instance IDs may be arbitrary and need not be contiguous or start at 1;
    masks are looked up by ID rather than by position. The reference
    implementation indexes a list positionally and assumes ID k sits at slot k,
    which silently requires that background be present and the IDs be dense. A
    prediction covering every pixel has no background, shifting every index.

    Args:
        true: (H, W) instance map, 0 = background
        pred: (H, W) instance map, 0 = background
        match_iou: IoU threshold for a match, 0.5 by default

    Returns:
        [dq, sq, pq], [paired_true, paired_pred, unpaired_true, unpaired_pred]
    """
    if match_iou < 0.0:
        raise ValueError("match_iou must be non-negative")

    true = np.asarray(true)
    pred = np.asarray(pred)

    true_ids = [int(i) for i in np.unique(true) if i != 0]
    pred_ids = [int(i) for i in np.unique(pred) if i != 0]

    if not true_ids and not pred_ids:
        # Nothing in either map. PQ is undefined; callers decide how to treat it.
        return [0.0, 0.0, 0.0], [[], [], [], []]

    true_masks = {i: np.array(true == i, np.uint8) for i in true_ids}
    pred_masks = {i: np.array(pred == i, np.uint8) for i in pred_ids}

    true_position = {i: k for k, i in enumerate(true_ids)}
    pred_position = {i: k for k, i in enumerate(pred_ids)}

    # Pairwise IoU, only over pairs that actually overlap.
    pairwise_iou = np.zeros([len(true_ids), len(pred_ids)], dtype=np.float64)

    for true_id in true_ids:
        t_mask = true_masks[true_id]
        for pred_id in np.unique(pred[t_mask > 0]):
            if pred_id == 0:
                continue
            p_mask = pred_masks[int(pred_id)]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_iou[true_position[true_id], pred_position[int(pred_id)]] = (
                inter / (total - inter)
            )

    if match_iou >= 0.5:
        # Above 0.5 at most one prediction can match a given ground truth.
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        true_slots, pred_slots = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[true_slots, pred_slots]
    else:
        true_slots, pred_slots = linear_sum_assignment(-pairwise_iou)
        paired_iou = pairwise_iou[true_slots, pred_slots]
        keep = paired_iou > match_iou
        true_slots, pred_slots, paired_iou = true_slots[keep], pred_slots[keep], paired_iou[keep]

    paired_true = [true_ids[slot] for slot in true_slots]
    paired_pred = [pred_ids[slot] for slot in pred_slots]

    unpaired_true = [i for i in true_ids if i not in set(paired_true)]
    unpaired_pred = [i for i in pred_ids if i not in set(paired_pred)]

    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)

    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1.0e-6)
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


def pq_per_image(true_inst, true_type, pred_inst, pred_type, num_types=NUM_TYPES):
    """
    Compute binary PQ and per-class PQ for a single patch.

    Args:
        true_inst: (H, W) ground truth instance map
        true_type: (H, W) ground truth type map, classes 1..num_types
        pred_inst: (H, W) predicted instance map
        pred_type: (H, W) predicted type map, classes 1..num_types
        num_types: number of nuclei classes

    Returns:
        bpq: float, binary PQ
        class_pq: (num_types,) float array, NaN where the class is absent from
                  both ground truth and prediction
        bdq: float, binary detection quality (F1 over matched instances)
        bsq: float, binary segmentation quality (mean IoU of those matches)

    bdq and bsq factorise bpq, and the split says which half of the pipeline is
    weak: low bdq means nuclei are missed, hallucinated, merged or split, and
    points at the watershed decoding; low bsq means the matches are found but
    their boundaries are loose, and points at the decoder's resolution. The two
    have different remedies, so reporting only their product hides which one
    applies.
    """
    true_inst = remap_label(true_inst)
    pred_inst = remap_label(pred_inst)

    bdq, bsq, bpq = get_fast_pq(true_inst, pred_inst)[0]

    class_pq = np.full(num_types, np.nan, dtype=np.float64)

    for t in range(1, num_types + 1):
        true_t = remap_label(true_inst * (true_type == t))
        pred_t = remap_label(pred_inst * (pred_type == t))

        true_empty = true_t.max() == 0
        pred_empty = pred_t.max() == 0

        if true_empty and pred_empty:
            class_pq[t - 1] = np.nan
        elif true_empty or pred_empty:
            class_pq[t - 1] = 0.0
        else:
            class_pq[t - 1] = get_fast_pq(true_t, pred_t)[0][2]

    return bpq, class_pq, bdq, bsq


def _safe_nanmean(values):
    """np.nanmean that returns NaN instead of warning on an all-NaN slice."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmean(values))


def aggregate_pq(bpq_list, class_pq_list, tissues, bdq_list=None, bsq_list=None):
    """
    Aggregate per-image PQ into the PanNuke headline numbers.

    Averages within each tissue first, then over tissues, so that Breast (the
    largest tissue) does not dominate. This matches the official protocol.

    Args:
        bpq_list: (N,) per-image binary PQ
        class_pq_list: (N, num_types) per-image per-class PQ, may contain NaN
        tissues: (N,) tissue type string per image
        bdq_list: (N,) per-image binary detection quality, optional
        bsq_list: (N,) per-image binary segmentation quality, optional

    Returns:
        dict with keys:
            mpq, bpq                overall scores averaged over tissues
            per_tissue              {tissue: {"mpq", "bpq", "n"}}
            per_class               (num_types,) PQ per nucleus class
            mpq_per_image           (N,) per-image mPQ
            bpq_per_image           (N,) per-image bPQ
    """
    bpq_per_image = np.asarray(bpq_list, dtype=np.float64)
    class_pq_arr = np.asarray(class_pq_list, dtype=np.float64)
    tissues = np.asarray(tissues)

    mpq_per_image = np.array([_safe_nanmean(row) for row in class_pq_arr])

    bdq_per_image = np.asarray(bdq_list, dtype=np.float64) if bdq_list is not None else None
    bsq_per_image = np.asarray(bsq_list, dtype=np.float64) if bsq_list is not None else None

    per_tissue = {}
    for tissue in sorted(set(tissues.tolist())):
        idx = tissues == tissue
        row = {
            "mpq": _safe_nanmean(mpq_per_image[idx]),
            "bpq": _safe_nanmean(bpq_per_image[idx]),
            "n": int(idx.sum()),
        }
        if bdq_per_image is not None:
            row["bdq"] = _safe_nanmean(bdq_per_image[idx])
        if bsq_per_image is not None:
            row["bsq"] = _safe_nanmean(bsq_per_image[idx])
        per_tissue[tissue] = row

    per_class = np.array([_safe_nanmean(class_pq_arr[:, t]) for t in range(class_pq_arr.shape[1])])

    results = {
        "mpq": _safe_nanmean([v["mpq"] for v in per_tissue.values()]),
        "bpq": _safe_nanmean([v["bpq"] for v in per_tissue.values()]),
        "per_tissue": per_tissue,
        "per_class": per_class,
        "mpq_per_image": mpq_per_image,
        "bpq_per_image": bpq_per_image,
    }

    # Averaged over tissues the same way as bPQ, so the three are comparable.
    # Note bdq * bsq only approximates bpq here: each is averaged separately,
    # and a mean of products is not the product of means.
    if bdq_per_image is not None:
        results["bdq"] = _safe_nanmean([v["bdq"] for v in per_tissue.values()])
    if bsq_per_image is not None:
        results["bsq"] = _safe_nanmean([v["bsq"] for v in per_tissue.values()])

    return results


def print_pq_report(name, results, type_names=TYPE_NAMES):
    """Print the tissue and class breakdown in the layout used by the papers."""
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    print(f"  mPQ: {results['mpq']:.4f}    bPQ: {results['bpq']:.4f}")

    print(f"\n  {'Tissue':<18} {'N':>5} {'mPQ':>10} {'bPQ':>10}")
    print(f"  {'-' * 18} {'-' * 5} {'-' * 10} {'-' * 10}")
    for tissue, row in results["per_tissue"].items():
        print(f"  {tissue:<18} {row['n']:>5} {row['mpq']:>10.4f} {row['bpq']:>10.4f}")

    print(f"\n  {'Class':<18} {'PQ':>10}")
    print(f"  {'-' * 18} {'-' * 10}")
    for cls_name, value in zip(type_names, results["per_class"]):
        print(f"  {cls_name:<18} {value:>10.4f}")
