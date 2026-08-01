"""
Stain augmentation and normalisation in optical density space.

H&E slides vary in colour by lab, scanner and batch far more than they vary in
nuclear morphology, and PanNuke spans 19 tissue types collected from several
sources. A model trained on raw RGB partly learns the colour statistics of the
tissues it saw most, which is why per-tissue Dice in the original run varied so
much more than the overall number suggested.

Beer-Lambert says transmitted intensity falls exponentially with stain
concentration, so the physically meaningful space is optical density,
OD = -log(I / Io), where stains combine linearly. Macenko's method recovers the
two stain vectors of a patch as the robust extremes of the OD point cloud
projected onto its principal plane. Perturbing the per-stain concentrations
there simulates a different staining protocol; perturbing RGB directly does not.

The key property for the consistency loss in train_chroma.py: this is a
pixel-wise function of colour alone. Two views of a patch stay exactly
registered, so the student and teacher outputs can be compared pixel to pixel
with no warping and no correspondence estimation.

References: Macenko et al., "A method for normalizing histology slides for
quantitative analysis", ISBI 2009; Tellez et al., "Quantifying the effects of
data augmentation and stain color normalization", Medical Image Analysis 2019.
"""

import numpy as np

# Background intensity of a bright field scanner, in 8-bit counts.
IO = 240.0
# Optical density below this is transparent glass rather than tissue.
BETA = 0.15
# Percentile used for the robust extremes of the stain angle.
ALPHA = 1.0
# Reject an estimate whose two stain vectors are this close to parallel. Real
# H&E vectors sit around 0.93; anything tighter means the patch does not carry
# two separable stains and the concentration solve is ill-conditioned.
MAX_STAIN_COLLINEARITY = 0.99

# Macenko's reference haematoxylin and eosin vectors, columns of a (3, 2)
# matrix in RGB optical density space, and their reference concentrations.
REFERENCE_STAIN_MATRIX = np.array(
    [[0.5626, 0.2159],
     [0.7201, 0.8012],
     [0.4062, 0.5581]], dtype=np.float32
)
REFERENCE_MAX_CONCENTRATION = np.array([1.9705, 1.0308], dtype=np.float32)


def rgb_to_od(image):
    """Convert an 8-bit RGB image to optical density."""
    image = np.asarray(image, dtype=np.float32)
    return np.maximum(-np.log((image + 1.0) / IO), 1e-6)


def od_to_rgb(od):
    """Convert optical density back to an 8-bit RGB image."""
    image = IO * np.exp(-np.asarray(od, dtype=np.float32))
    return np.clip(image, 0, 255).astype(np.uint8)


def estimate_stain_matrix(image, beta=BETA, alpha=ALPHA):
    """
    Macenko estimate of a patch's haematoxylin and eosin vectors.

    Args:
        image: (H, W, 3) uint8 RGB patch
        beta: optical density threshold separating tissue from glass
        alpha: percentile for the robust extremes of the stain angle

    Returns:
        (3, 2) float32 matrix whose columns are the unit H and E vectors, or
        None when the patch holds too little tissue to estimate from.
    """
    od = rgb_to_od(image).reshape(-1, 3)
    tissue = od[np.all(od > beta, axis=1)]

    if tissue.shape[0] < 100:
        return None

    covariance = np.cov(tissue.T)
    if not np.all(np.isfinite(covariance)):
        return None

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # eigh returns ascending eigenvalues; the stains span the top two.
    plane = eigenvectors[:, [2, 1]]

    projected = tissue @ plane
    angles = np.arctan2(projected[:, 1], projected[:, 0])

    low = np.percentile(angles, alpha)
    high = np.percentile(angles, 100 - alpha)

    v_low = plane @ np.array([np.cos(low), np.sin(low)])
    v_high = plane @ np.array([np.cos(high), np.sin(high)])

    # Eigenvector signs are arbitrary; optical density is non-negative.
    if v_low.sum() < 0:
        v_low = -v_low
    if v_high.sum() < 0:
        v_high = -v_high

    # Haematoxylin is the bluer stain, so it absorbs more red light and has the
    # larger red component in optical density.
    if v_low[0] > v_high[0]:
        stain_matrix = np.stack([v_low, v_high], axis=1)
    else:
        stain_matrix = np.stack([v_high, v_low], axis=1)

    norms = np.linalg.norm(stain_matrix, axis=0, keepdims=True)
    if np.any(norms < 1e-6):
        return None

    stain_matrix = stain_matrix / norms

    # A patch dominated by one stain, or by background, collapses the optical
    # density cloud onto a line. The two "stain" vectors then come out nearly
    # parallel, the least-squares solve becomes ill-conditioned, and the
    # unphysical negative concentrations it returns blow up the reconstruction.
    # Reporting failure lets callers fall back to the reference stains.
    if abs(float(stain_matrix[:, 0] @ stain_matrix[:, 1])) > MAX_STAIN_COLLINEARITY:
        return None

    return stain_matrix.astype(np.float32)


def get_concentrations(image, stain_matrix):
    """
    Least-squares stain concentrations per pixel.

    Args:
        image: (H, W, 3) uint8 RGB patch
        stain_matrix: (3, 2) stain vectors

    Returns:
        (2, H * W) float32 concentrations for haematoxylin and eosin
    """
    od = rgb_to_od(image).reshape(-1, 3)
    return np.linalg.lstsq(stain_matrix, od.T, rcond=None)[0].astype(np.float32)


def augment_stain(image, sigma_scale=0.25, sigma_shift=0.05, rng=None, stain_matrix=None):
    """
    Jitter a patch's stain concentrations, leaving geometry untouched.

    Each stain channel is independently scaled by 1 + U(-sigma_scale,
    sigma_scale) and shifted by U(-sigma_shift, sigma_shift), then the patch is
    reconstructed through the same stain vectors. Because this is a pixel-wise
    function of colour, the result stays exactly registered with the input.

    Args:
        image: (H, W, 3) uint8 RGB patch
        sigma_scale: multiplicative jitter per stain
        sigma_shift: additive jitter per stain
        rng: numpy Generator, for reproducible augmentation
        stain_matrix: precomputed stain vectors, estimated if not given

    Returns:
        (H, W, 3) uint8 augmented patch
    """
    if rng is None:
        rng = np.random.default_rng()

    if stain_matrix is None:
        stain_matrix = estimate_stain_matrix(image)
    if stain_matrix is None:
        # Nearly blank patch: fall back to the reference stains so the
        # augmentation still perturbs colour rather than silently no-op-ing.
        stain_matrix = REFERENCE_STAIN_MATRIX

    concentrations = get_concentrations(image, stain_matrix)

    scale = 1.0 + rng.uniform(-sigma_scale, sigma_scale, size=(2, 1))
    shift = rng.uniform(-sigma_shift, sigma_shift, size=(2, 1))
    concentrations = np.maximum(concentrations * scale + shift, 0.0)

    od = (stain_matrix @ concentrations).T.reshape(np.asarray(image).shape)
    return od_to_rgb(od)


def normalize_stain(image, target_matrix=REFERENCE_STAIN_MATRIX,
                    target_max=REFERENCE_MAX_CONCENTRATION):
    """
    Macenko normalisation onto a fixed reference stain appearance.

    Used as an optional deterministic preprocessing step at evaluation time;
    training instead relies on augment_stain, which covers a wider range of
    appearances than normalisation can collapse.

    Args:
        image: (H, W, 3) uint8 RGB patch
        target_matrix: (3, 2) reference stain vectors
        target_max: (2,) reference 99th-percentile concentrations

    Returns:
        (H, W, 3) uint8 normalised patch, or the input when estimation fails
    """
    image = np.asarray(image)
    stain_matrix = estimate_stain_matrix(image)
    if stain_matrix is None:
        return image.astype(np.uint8)

    concentrations = get_concentrations(image, stain_matrix)

    max_concentration = np.percentile(concentrations, 99, axis=1).reshape(-1, 1)
    max_concentration = np.maximum(max_concentration, 1e-6)
    concentrations = concentrations * (np.asarray(target_max).reshape(-1, 1) / max_concentration)

    od = (target_matrix @ concentrations).T.reshape(image.shape)
    return od_to_rgb(od)
