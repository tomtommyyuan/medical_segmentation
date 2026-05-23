"""
Simple fully convolutional CNN baseline for binary nuclei segmentation.

A shallow encoder (no skip connections, no decoder) that directly outputs
a 256x256 prediction. Serves as a weak neural baseline to compare against U-Net.
"""

import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    def __init__(self, in_channels=3, base_filters=32):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),

            # Block 2
            nn.Conv2d(base_filters, base_filters * 2, 3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 2, base_filters * 2, 3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),

            # Block 3
            nn.Conv2d(base_filters * 2, base_filters * 4, 3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, 3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),

            # 1x1 conv to single channel
            nn.Conv2d(base_filters * 4, 1, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, 256, 256) float tensor
        Returns:
            logits: (B, 1, 256, 256) raw logits
        """
        return self.net(x)
