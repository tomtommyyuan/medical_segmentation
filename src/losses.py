"""
Multi-task loss for CHROMA-Net, plus the stain consistency term.

Six supervised terms over three branches:

    nuclei    binary cross entropy + soft Dice
    distance  MSE on the maps + MSE on their gradients
    class     logit-adjusted cross entropy + soft Dice

The gradient term is the one that earns its place. The watershed reads the
derivative of the horizontal map along x and the vertical map along y, so a
model that fits the maps but blurs their gradients still merges touching
nuclei. Supervising the gradient directly, restricted to nuclei pixels,
sharpens exactly the signal decoding depends on.

The class branch uses logit adjustment because mPQ averages PQ over the five
nuclei classes, giving Dead nuclei - around 1% of PanNuke's instances - the
same weight as Neoplastic. Most published methods score close to zero on Dead,
which costs them mPQ out of all proportion to the pixels involved. Adding
tau * log(prior) to the logits during training moves the decision boundary
toward the rare classes with no change to the architecture; inference uses the
plain logits.

The consistency term compares two stain-jittered views of the same patch. Stain
augmentation never moves a pixel, so the views are exactly registered and the
branches can be compared pixel to pixel with no warping.

References: Graham et al., "HoVer-Net", Medical Image Analysis 2019;
Menon et al., "Long-tail learning via logit adjustment", ICLR 2021;
Tarvainen and Valpola, "Mean teachers", NeurIPS 2017.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocess import NUM_TYPES

# HoVer-Net's term weights, which this follows so ablations are interpretable.
DEFAULT_WEIGHTS = {
    "np_bce": 1.0,
    "np_dice": 1.0,
    "hv_mse": 1.0,
    "hv_msge": 1.0,
    "tp_ce": 1.0,
    "tp_dice": 1.0,
}


def sobel_kernels(size=5):
    """
    Distance-weighted derivative kernels, as used by HoVer-Net.

    Returns:
        kernel_x, kernel_y: (size, size) tensors differentiating along columns
        and along rows respectively
    """
    if size % 2 == 0:
        raise ValueError("Sobel kernel size must be odd")

    coords = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    denominator = grid_x * grid_x + grid_y * grid_y + 1e-15

    return grid_x / denominator, grid_y / denominator


def gradient_hv(hv, size=5):
    """
    Differentiate each distance map along its own axis.

    The horizontal map is differentiated along x and the vertical along y,
    which is the exact pair of derivatives postprocess.py feeds to the
    watershed. Differentiating along the other axis would supervise a signal
    decoding never reads.

    Args:
        hv: (B, 2, H, W) distance maps
        size: kernel size

    Returns:
        (B, 2, H, W) gradients
    """
    kernel_x, kernel_y = sobel_kernels(size)
    kernel_x = kernel_x.view(1, 1, size, size).to(hv.device, hv.dtype)
    kernel_y = kernel_y.view(1, 1, size, size).to(hv.device, hv.dtype)
    padding = size // 2

    d_horizontal = F.conv2d(hv[:, 0:1], kernel_x, padding=padding)
    d_vertical = F.conv2d(hv[:, 1:2], kernel_y, padding=padding)

    return torch.cat([d_horizontal, d_vertical], dim=1)


def dice_loss(probabilities, target, smooth=1e-3):
    """
    Soft Dice, computed per channel over the whole batch.

    Aggregating across the batch rather than per sample avoids the failure the
    original evaluate.py had, where an empty prediction on an empty patch
    scored a free 1.0 through the smoothing term.

    Args:
        probabilities: (B, C, H, W) in [0, 1]
        target: (B, C, H, W) in {0, 1}

    Returns:
        scalar loss, summed over channels
    """
    dims = (0, 2, 3)
    intersection = (probabilities * target).sum(dims)
    cardinality = probabilities.sum(dims) + target.sum(dims)

    return (1.0 - (2.0 * intersection + smooth) / (cardinality + smooth)).sum()


def msge_loss(pred_hv, true_hv, focus):
    """
    MSE between predicted and true distance-map gradients, over nuclei only.

    Background gradients are meaningless (both maps are flat zero there) and
    would dominate the average, since nuclei cover well under half the pixels.

    Args:
        pred_hv: (B, 2, H, W) predicted maps
        true_hv: (B, 2, H, W) target maps
        focus: (B, H, W) binary nuclei mask

    Returns:
        scalar loss
    """
    focus = focus.unsqueeze(1).to(pred_hv.dtype)
    focus = torch.cat([focus, focus], dim=1)

    difference = gradient_hv(pred_hv) - gradient_hv(true_hv)

    return (focus * difference * difference).sum() / (focus.sum() + 1e-8)


def compute_class_prior(type_maps, num_types=NUM_TYPES, chunk=256):
    """
    Pixel frequency of every class, including background, over a fold.

    Args:
        type_maps: (N, H, W) array of class labels, may be memory-mapped
        num_types: number of nuclei classes
        chunk: patches per pass, so memory-mapped folds are not materialised

    Returns:
        (num_types + 1,) float64 array summing to 1
    """
    counts = np.zeros(num_types + 1, dtype=np.float64)

    for start in range(0, len(type_maps), chunk):
        block = np.asarray(type_maps[start : start + chunk]).astype(np.int64).ravel()
        counts += np.bincount(block, minlength=num_types + 1)[: num_types + 1]

    return counts / counts.sum()


class LogitAdjustedCrossEntropy(nn.Module):
    """
    Cross entropy with a class-prior offset added to the logits.

    Args:
        class_prior: (C,) frequencies, background first
        tau: strength of the adjustment; 0 recovers plain cross entropy
    """

    def __init__(self, class_prior, tau=1.0):
        super().__init__()
        prior = torch.as_tensor(np.asarray(class_prior), dtype=torch.float32)
        prior = prior / prior.sum()

        self.register_buffer("log_prior", torch.log(prior.clamp_min(1e-12)))
        self.tau = tau

    def forward(self, logits, target):
        adjusted = logits + self.tau * self.log_prior.view(1, -1, 1, 1)
        return F.cross_entropy(adjusted, target)


class ChromaLoss(nn.Module):
    """
    The full supervised objective.

    Args:
        class_prior: (num_types + 1,) pixel frequencies for logit adjustment,
            or None for plain cross entropy
        tau: logit adjustment strength
        weights: per-term weights, defaults to HoVer-Net's
        num_types: number of nuclei classes
    """

    def __init__(self, class_prior=None, tau=1.0, weights=None, num_types=NUM_TYPES):
        super().__init__()
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

        self.num_types = num_types

        if class_prior is not None and tau > 0:
            self.type_criterion = LogitAdjustedCrossEntropy(class_prior, tau=tau)
        else:
            self.type_criterion = None

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with np (B, 1, H, W), hv (B, 2, H, W), tp (B, C, H, W)
            targets: dict with np_map (B, H, W), hv_map (B, 2, H, W),
                tp_map (B, H, W)

        Returns:
            total: scalar loss
            terms: dict of the individual weighted terms, for logging
        """
        np_target = targets["np_map"].unsqueeze(1).to(outputs["np"].dtype)
        hv_target = targets["hv_map"].to(outputs["hv"].dtype)
        tp_target = targets["tp_map"].long()

        np_bce = F.binary_cross_entropy_with_logits(outputs["np"], np_target)
        np_dice = dice_loss(torch.sigmoid(outputs["np"]), np_target)

        hv_mse = F.mse_loss(outputs["hv"], hv_target)
        hv_msge = msge_loss(outputs["hv"], hv_target, targets["np_map"])

        if self.type_criterion is not None:
            tp_ce = self.type_criterion(outputs["tp"], tp_target)
        else:
            tp_ce = F.cross_entropy(outputs["tp"], tp_target)

        tp_onehot = F.one_hot(tp_target, self.num_types + 1)
        tp_onehot = tp_onehot.permute(0, 3, 1, 2).to(outputs["tp"].dtype)
        tp_dice = dice_loss(torch.softmax(outputs["tp"], dim=1), tp_onehot)

        terms = {
            "np_bce": self.weights["np_bce"] * np_bce,
            "np_dice": self.weights["np_dice"] * np_dice,
            "hv_mse": self.weights["hv_mse"] * hv_mse,
            "hv_msge": self.weights["hv_msge"] * hv_msge,
            "tp_ce": self.weights["tp_ce"] * tp_ce,
            "tp_dice": self.weights["tp_dice"] * tp_dice,
        }

        return sum(terms.values()), terms


def consistency_loss(student, teacher):
    """
    Agreement between the student's strong view and the teacher's weak view.

    Both views come from the same patch after the same geometric transform, so
    they are registered pixel for pixel and the branches compare directly.
    The class branch uses KL divergence rather than MSE, which behaves better
    when the teacher is confident, and is summed over classes then averaged over
    pixels so it stays on the same scale as the other two terms.

    Args:
        student: dict of the student's outputs on the strongly jittered view
        teacher: dict of the teacher's outputs on the weakly jittered view

    Returns:
        scalar loss
    """
    teacher_np = torch.sigmoid(teacher["np"]).detach()
    teacher_hv = teacher["hv"].detach()
    teacher_tp = torch.softmax(teacher["tp"], dim=1).detach()

    np_term = F.mse_loss(torch.sigmoid(student["np"]), teacher_np)
    hv_term = F.mse_loss(student["hv"], teacher_hv)

    log_student_tp = F.log_softmax(student["tp"], dim=1)
    tp_term = F.kl_div(log_student_tp, teacher_tp, reduction="none").sum(dim=1).mean()

    return np_term + hv_term + tp_term


def sigmoid_rampup(step, length):
    """
    Consistency weight ramp, from Tarvainen and Valpola.

    The teacher is a copy of a randomly initialised student at step 0, so its
    targets are noise. Ramping the weight in over the first epochs stops that
    noise from steering the student before the teacher is worth listening to.
    """
    if length <= 0:
        return 1.0

    progress = float(np.clip(step / length, 0.0, 1.0))
    return float(np.exp(-5.0 * (1.0 - progress) ** 2))
