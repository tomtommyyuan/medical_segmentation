"""
Turn the network's three output maps into labelled nuclei.

The nuclei branch gives a foreground mask, but touching nuclei come out as one
blob. The distance maps fix that: their gradient spikes where two nuclei meet,
so taking Sobel derivatives of the horizontal map along x and the vertical map
along y produces an energy ridge at every join. Subtracting that ridge from the
foreground leaves one marker per nucleus, and a watershed grows those markers
back out to the blob boundary.

Each resulting instance is then given a single class by majority vote over the
type branch inside it. Voting per instance rather than per pixel matters for
mPQ: a nucleus whose pixels are split between two classes would otherwise be
torn into two instances when the metric filters by class.

Reference: Graham et al., "HoVer-Net", Medical Image Analysis 2019.
"""

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, label
from skimage.segmentation import watershed

from preprocess import NUM_TYPES

# Cross-shaped kernel used to clean up markers before labelling.
MARKER_KERNEL = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)


def _remove_small_objects(labels, min_size):
    """
    Drop labelled objects smaller than min_size pixels.

    Written out rather than calling skimage.morphology.remove_small_objects,
    whose threshold parameter changed name and became inclusive in skimage
    0.26. Clusters run whatever skimage the site module ships, and a silently
    off-by-one size filter would move PQ.
    """
    labels = np.asarray(labels)
    if labels.max() == 0:
        return labels

    sizes = np.bincount(labels.ravel())
    too_small = sizes < min_size
    too_small[0] = False

    out = labels.copy()
    out[too_small[labels]] = 0
    return out


def _minmax(x):
    """Rescale to [0, 1]; returns zeros for a constant input."""
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def decode_instances(np_map, hv_map, np_threshold=0.5, marker_threshold=0.4,
                     min_size=10, sobel_ksize=21):
    """
    Recover an instance map from the nuclei probability and distance maps.

    Args:
        np_map: (H, W) foreground probability in [0, 1]
        hv_map: (2, H, W) predicted horizontal and vertical distance maps
        np_threshold: foreground cutoff
        marker_threshold: energy cutoff that carves markers out of the blobs
        min_size: drop blobs and markers below this many pixels
        sobel_ksize: Sobel aperture; large because the distance maps are smooth

    Returns:
        (H, W) int32 instance map, 0 = background
    """
    np_map = np.asarray(np_map, dtype=np.float32)
    h_dir = np.asarray(hv_map[0], dtype=np.float32)
    v_dir = np.asarray(hv_map[1], dtype=np.float32)

    blob = np.array(np_map >= np_threshold, dtype=np.int32)
    blob = label(blob)[0]
    blob = _remove_small_objects(blob, min_size)
    blob[blob > 0] = 1

    if blob.max() == 0:
        return np.zeros(np_map.shape, dtype=np.int32)

    h_dir = _minmax(h_dir)
    v_dir = _minmax(v_dir)

    # The horizontal map only separates nuclei along x, the vertical only along
    # y, so each is differentiated along its own axis and the two are combined.
    sobel_h = cv2.Sobel(h_dir, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
    sobel_v = cv2.Sobel(v_dir, cv2.CV_32F, 0, 1, ksize=sobel_ksize)
    sobel_h = 1.0 - _minmax(sobel_h)
    sobel_v = 1.0 - _minmax(sobel_v)

    energy = np.maximum(sobel_h, sobel_v)
    energy = energy - (1 - blob)
    energy[energy < 0] = 0

    # Watershed descends, so the surface is negated: nucleus centres become
    # basins and the ridges between touching nuclei become walls.
    distance = (1.0 - energy) * blob
    distance = -cv2.GaussianBlur(distance, (3, 3), 0)

    energy = np.array(energy >= marker_threshold, dtype=np.int32)

    markers = blob - energy
    markers[markers < 0] = 0
    markers = binary_fill_holes(markers).astype(np.uint8)
    markers = cv2.morphologyEx(markers, cv2.MORPH_OPEN, MARKER_KERNEL)
    markers = label(markers)[0]
    markers = _remove_small_objects(markers, min_size)

    if markers.max() == 0:
        return np.zeros(np_map.shape, dtype=np.int32)

    inst_map = watershed(distance, markers=markers, mask=blob.astype(bool))

    return inst_map.astype(np.int32)


def assign_types(inst_map, type_map, num_types=NUM_TYPES):
    """
    Give every instance one class by majority vote of the per-pixel type map.

    Background votes are ignored unless an instance has nothing else, in which
    case the next most common class wins.

    Args:
        inst_map: (H, W) instance map, 0 = background
        type_map: (H, W) per-pixel predicted class, 0 = background
        num_types: number of nuclei classes

    Returns:
        (H, W) uint8 type map consistent with inst_map
    """
    inst_map = np.asarray(inst_map)
    type_map = np.asarray(type_map).astype(np.int64)
    out = np.zeros(inst_map.shape, dtype=np.uint8)

    for inst_id in np.unique(inst_map):
        if inst_id == 0:
            continue

        mask = inst_map == inst_id
        counts = np.bincount(type_map[mask], minlength=num_types + 1)
        order = np.argsort(-counts)

        cls = int(order[0])
        if cls == 0:
            # Every pixel voted background; take the best real class instead.
            cls = int(order[1]) if counts[order[1]] > 0 else 1

        out[mask] = cls

    return out


def decode_batch(np_prob, hv_pred, tp_pred, **kwargs):
    """
    Run decoding over a batch.

    Args:
        np_prob: (B, H, W) foreground probability
        hv_pred: (B, 2, H, W) distance maps
        tp_pred: (B, H, W) per-pixel predicted class

    Returns:
        inst_maps: (B, H, W) int32
        type_maps: (B, H, W) uint8
    """
    batch = np_prob.shape[0]
    inst_maps = np.zeros(np_prob.shape, dtype=np.int32)
    type_maps = np.zeros(np_prob.shape, dtype=np.uint8)

    for i in range(batch):
        inst = decode_instances(np_prob[i], hv_pred[i], **kwargs)
        inst_maps[i] = inst
        type_maps[i] = assign_types(inst, tp_pred[i])

    return inst_maps, type_maps
