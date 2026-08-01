"""
Horizontal/vertical distance maps, the regression target that separates nuclei.

For every nucleus, the horizontal map holds each pixel's signed offset from the
instance centre of mass along x, rescaled to [-1, 0] on the left and [0, 1] on
the right; the vertical map does the same along y. Two nuclei that touch have
distance maps that run in opposite directions across the join, so the gradient
of the maps spikes exactly at the boundary between them. That spike is what
postprocess.py turns into a watershed energy landscape.

This is the mechanism that lets a semantic segmentation network do instance
segmentation without a detection head, and it is why the repo can move from
binary Dice to the mPQ/bPQ the PanNuke benchmark is scored on.

Targets are generated on the fly from the instance map rather than precomputed,
so geometric augmentation only has to be applied to the instance map. Rotating
or flipping a precomputed HV map also requires negating and swapping its
channels, which is an easy thing to get silently wrong (see tta.py, where the
same correction is unavoidable and is unit tested).

Reference: Graham et al., "HoVer-Net", Medical Image Analysis 2019.
"""

import numpy as np
from scipy.ndimage import center_of_mass


def get_bounding_box(mask):
    """Return (y0, y1, x0, x1) bounds of the non-zero region, y1/x1 exclusive."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return y0, y1 + 1, x0, x1 + 1


def gen_hv_map(inst_map):
    """
    Build the horizontal and vertical distance maps for an instance map.

    Args:
        inst_map: (H, W) instance map, 0 = background

    Returns:
        (2, H, W) float32 in [-1, 1]; channel 0 horizontal, channel 1 vertical.
        Background pixels are 0.
    """
    inst_map = np.asarray(inst_map)
    h_map = np.zeros(inst_map.shape, dtype=np.float32)
    v_map = np.zeros(inst_map.shape, dtype=np.float32)

    for inst_id in np.unique(inst_map):
        if inst_id == 0:
            continue

        inst = (inst_map == inst_id).astype(np.uint8)
        y0, y1, x0, x1 = get_bounding_box(inst)
        inst = inst[y0:y1, x0:x1]

        # A 1px sliver has no meaningful centre; leaving it at 0 keeps it in the
        # foreground map but contributes nothing to the watershed energy.
        if inst.shape[0] < 2 or inst.shape[1] < 2:
            continue

        com = center_of_mass(inst)
        com_y = int(com[0] + 0.5)
        com_x = int(com[1] + 0.5)

        x_range = np.arange(1, inst.shape[1] + 1) - com_x
        y_range = np.arange(1, inst.shape[0] + 1) - com_y
        grid_x, grid_y = np.meshgrid(x_range, y_range)

        grid_x = grid_x.astype(np.float32)
        grid_y = grid_y.astype(np.float32)
        grid_x[inst == 0] = 0
        grid_y[inst == 0] = 0

        # Each side is normalised independently so an off-centre nucleus still
        # spans the full [-1, 1] range and its gradient stays comparable.
        if grid_x.min() < 0:
            grid_x[grid_x < 0] /= -grid_x[grid_x < 0].min()
        if grid_y.min() < 0:
            grid_y[grid_y < 0] /= -grid_y[grid_y < 0].min()
        if grid_x.max() > 0:
            grid_x[grid_x > 0] /= grid_x[grid_x > 0].max()
        if grid_y.max() > 0:
            grid_y[grid_y > 0] /= grid_y[grid_y > 0].max()

        h_map[y0:y1, x0:x1][inst > 0] = grid_x[inst > 0]
        v_map[y0:y1, x0:x1][inst > 0] = grid_y[inst > 0]

    return np.stack([h_map, v_map]).astype(np.float32)


def gen_targets(inst_map, type_map):
    """
    Build the full multi-task target set for one patch.

    Args:
        inst_map: (H, W) instance map, 0 = background
        type_map: (H, W) type map, classes 1..5, 0 = background

    Returns:
        np_map: (H, W) float32 binary nuclei map
        hv_map: (2, H, W) float32 distance maps
        tp_map: (H, W) int64 class map for the cross-entropy branch
    """
    np_map = (inst_map > 0).astype(np.float32)
    hv_map = gen_hv_map(inst_map)
    tp_map = np.asarray(type_map).astype(np.int64)
    return np_map, hv_map, tp_map
