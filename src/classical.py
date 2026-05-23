"""
Classical CV baseline: Otsu thresholding + watershed segmentation.

Converts H&E patches to grayscale, applies Otsu thresholding to find nuclei
regions, then uses distance transform + watershed to separate touching nuclei.
"""

import numpy as np
import cv2
from scipy import ndimage


def segment_single(image):
    """
    Segment nuclei in a single 256x256 RGB image.

    Args:
        image: (256, 256, 3) uint8 RGB image

    Returns:
        binary_mask: (256, 256) uint8, 1 = nuclei, 0 = background
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Invert so nuclei (dark in H&E) become bright
    gray_inv = 255 - gray

    # Otsu threshold
    _, thresh = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background via dilation
    sure_bg = cv2.dilate(opened, kernel, iterations=3)

    # Sure foreground via distance transform
    dist_transform = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling for watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed needs 3-channel image
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    markers = cv2.watershed(img_bgr, markers)

    # Build binary mask: watershed boundaries are -1, background is 1
    binary_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    binary_mask[markers > 1] = 1

    return binary_mask


def segment_batch(images):
    """
    Segment a batch of images.

    Args:
        images: (N, 256, 256, 3) uint8

    Returns:
        masks: (N, 256, 256) uint8
    """
    masks = np.zeros((len(images), images.shape[1], images.shape[2]), dtype=np.uint8)
    for i in range(len(images)):
        masks[i] = segment_single(images[i])
    return masks
