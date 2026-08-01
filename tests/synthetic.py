"""
Synthetic nuclei patches, so tests run without the PanNuke download.

Builds discs on a canvas: well-separated ones, touching pairs along each axis
and a chain of three, which is the case the distance maps exist to handle.
"""

import numpy as np


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
    """Build a plausible H&E-looking RGB patch: purple nuclei on pink stroma."""
    rng = np.random.default_rng(seed)
    image = np.zeros(inst.shape + (3,), dtype=np.float32)

    image[..., 0] = 220.0
    image[..., 1] = 180.0
    image[..., 2] = 210.0

    foreground = inst > 0
    image[foreground, 0] = 110.0
    image[foreground, 1] = 70.0
    image[foreground, 2] = 160.0

    image += rng.normal(0, 4.0, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)
