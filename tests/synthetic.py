"""
Synthetic nuclei patches, so tests run without the PanNuke download.

Builds discs on a canvas: well-separated ones, touching pairs along each axis
and a chain of three, which is the case the distance maps exist to handle.
"""

import numpy as np

# Macenko's reference H and E vectors, hardcoded rather than imported from
# stain_augment so the fixture does not depend on the code it is used to test.
_STAIN_MATRIX = np.array(
    [[0.5626, 0.2159],
     [0.7201, 0.8012],
     [0.4062, 0.5581]]
)
_IO = 240.0


def draw_disc(canvas, cy, cx, radius, value):
    """Stamp a filled disc onto a canvas, overwriting whatever is there."""
    yy, xx = np.ogrid[: canvas.shape[0], : canvas.shape[1]]
    canvas[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius] = value
    return canvas


def make_patch(size=256):
    """
    Build a synthetic instance map and matching type map.

    Returns:
        inst: (size, size) int32 instance map with 9 nuclei
        type_map: (size, size) uint8 class map, classes 1-5
    """
    inst = np.zeros((size, size), dtype=np.int32)
    type_map = np.zeros((size, size), dtype=np.uint8)

    # (cy, cx, radius, instance id, class)
    nuclei = [
        (40, 40, 12, 1, 1),      # isolated
        (40, 100, 12, 2, 2),     # isolated
        (140, 60, 14, 3, 1),     # touching pair, side by side
        (140, 84, 14, 4, 3),
        (190, 180, 14, 5, 1),    # touching pair, stacked
        (214, 180, 14, 6, 4),
        (60, 190, 13, 7, 5),     # chain of three
        (60, 212, 13, 8, 5),
        (60, 234, 13, 9, 1),
    ]

    for cy, cx, radius, inst_id, cls in nuclei:
        draw_disc(inst, cy, cx, radius, inst_id)
        draw_disc(type_map, cy, cx, radius, cls)

    return inst, type_map


def make_image(inst, seed=0):
    """
    Build an H&E-like RGB patch by construction in optical density space.

    Nuclei get a high haematoxylin concentration and low eosin, stroma the
    reverse, and the patch is rendered through the reference stain vectors.
    Building it this way rather than picking two RGB colours gives a patch whose
    stain vectors are actually recoverable by Macenko's method, which is what
    the stain tests need in order to exercise anything.

    Args:
        inst: (H, W) instance map, used only to place nuclei
        seed: seed for the per-pixel concentration noise

    Returns:
        (H, W, 3) uint8 RGB patch
    """
    rng = np.random.default_rng(seed)
    foreground = inst > 0

    haematoxylin = np.where(foreground, 1.4, 0.15) + rng.normal(0, 0.05, inst.shape)
    eosin = np.where(foreground, 0.25, 0.9) + rng.normal(0, 0.05, inst.shape)

    concentrations = np.stack([haematoxylin.ravel(), eosin.ravel()])
    od = (_STAIN_MATRIX @ concentrations).T.reshape(inst.shape + (3,))

    return np.clip(_IO * np.exp(-od), 0, 255).astype(np.uint8)
