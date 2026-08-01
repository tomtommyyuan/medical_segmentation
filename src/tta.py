"""
Test-time augmentation over the dihedral group.

Averaging predictions over the eight rotations and reflections of a patch is
close to free accuracy, but the distance maps make it easy to get wrong. They
are not passive images: the horizontal map stores a signed offset along x, so
mirroring a patch left to right does not just mirror that map, it negates it.
Rotating by 90 degrees swaps the two maps and negates one of them.

Get this wrong and nothing crashes. The distance maps quietly cancel when
averaged, the watershed loses its ridges, touching nuclei merge, and TTA makes
PQ worse instead of better while looking like it works.

The correction is derived once here and pinned by tests that compare against
distance maps regenerated from transformed instance maps, which is the only
statement of correctness that does not just restate the implementation.
"""

import torch

# The eight dihedral transforms: rotate by k * 90 degrees, then optionally
# mirror left to right.
TRANSFORMS = [(k, flip) for k in range(4) for flip in (False, True)]


def transform_image(image, k, flip):
    """
    Apply a dihedral transform to a batch of images.

    Args:
        image: (B, C, H, W) tensor
        k: number of 90 degree counter-clockwise rotations
        flip: mirror left to right after rotating

    Returns:
        (B, C, H, W) transformed tensor
    """
    if k:
        image = torch.rot90(image, k, dims=(-2, -1))
    if flip:
        image = torch.flip(image, dims=(-1,))
    return image


def _undo_flip_hv(h, v):
    """
    Undo a left-right mirror on the distance maps.

    Mirroring sends x to W-1-x, so a pixel's offset from its instance centre
    changes sign along x and is unchanged along y.
    """
    return -torch.flip(h, dims=(-1,)), torch.flip(v, dims=(-1,))


def _undo_rot90_hv(h, v):
    """
    Undo one 90 degree counter-clockwise rotation on the distance maps.

    The rotation carries the x axis onto the y axis, so the map that was
    horizontal becomes vertical and the one that was vertical becomes
    horizontal with its sign reversed.
    """
    return (
        -torch.rot90(v, -1, dims=(-2, -1)),
        torch.rot90(h, -1, dims=(-2, -1)),
    )


def inverse_transform(outputs, k, flip):
    """
    Map predictions made on a transformed patch back to the original frame.

    The nuclei and class branches are plain images and only need the geometric
    inverse. The distance maps need the sign and channel corrections above.

    Args:
        outputs: dict with np (B, 1, H, W), hv (B, 2, H, W), tp (B, C, H, W)
        k: rotations that were applied
        flip: whether a mirror was applied

    Returns:
        dict of the same shape, in the original orientation
    """
    np_out = outputs["np"]
    tp_out = outputs["tp"]
    h, v = outputs["hv"][:, 0:1], outputs["hv"][:, 1:2]

    # The forward transform rotated then flipped, so undo the flip first.
    if flip:
        np_out = torch.flip(np_out, dims=(-1,))
        tp_out = torch.flip(tp_out, dims=(-1,))
        h, v = _undo_flip_hv(h, v)

    for _ in range(k):
        np_out = torch.rot90(np_out, -1, dims=(-2, -1))
        tp_out = torch.rot90(tp_out, -1, dims=(-2, -1))
        h, v = _undo_rot90_hv(h, v)

    return {"np": np_out, "hv": torch.cat([h, v], dim=1), "tp": tp_out}


@torch.no_grad()
def predict_tta(model, image, tissue=None, transforms=TRANSFORMS):
    """
    Average a model's predictions over the dihedral group.

    Probabilities are averaged rather than logits: the average of eight
    sigmoids is the quantity the 0.5 threshold is calibrated against, whereas
    averaging logits lets one over-confident view dominate.

    Args:
        model: a CHROMA-Net
        image: (B, 3, H, W) normalised batch
        tissue: (B,) tissue indices, or None
        transforms: which dihedral transforms to average over

    Returns:
        dict with np (B, 1, H, W) probability, hv (B, 2, H, W) distance maps,
        tp (B, C, H, W) class probability
    """
    np_total = None
    hv_total = None
    tp_total = None

    for k, flip in transforms:
        outputs = model(transform_image(image, k, flip), tissue)
        outputs = inverse_transform(outputs, k, flip)

        np_prob = torch.sigmoid(outputs["np"])
        tp_prob = torch.softmax(outputs["tp"], dim=1)

        if np_total is None:
            np_total, hv_total, tp_total = np_prob, outputs["hv"], tp_prob
        else:
            np_total = np_total + np_prob
            hv_total = hv_total + outputs["hv"]
            tp_total = tp_total + tp_prob

    n = len(transforms)
    return {"np": np_total / n, "hv": hv_total / n, "tp": tp_total / n}


@torch.no_grad()
def predict_plain(model, image, tissue=None):
    """Single-view prediction, returned in the same probability form as TTA."""
    outputs = model(image, tissue)
    return {
        "np": torch.sigmoid(outputs["np"]),
        "hv": outputs["hv"],
        "tp": torch.softmax(outputs["tp"], dim=1),
    }
