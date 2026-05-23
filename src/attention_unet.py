"""
Attention U-Net for binary nuclei segmentation.

Same encoder-decoder structure as vanilla U-Net but with attention gates
on skip connections to suppress irrelevant features and focus on nuclei regions.

Reference: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas", 2018.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Attention gate: uses the gating signal (from decoder) to weight
    the skip connection (from encoder), letting the network learn
    which spatial regions are relevant.
    """

    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        attention = self.psi(self.relu(g + s))
        return skip * attention


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, base_filters=64):
        super().__init__()
        f = base_filters

        # Encoder
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f, f * 2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.enc4 = ConvBlock(f * 4, f * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(f * 8, f * 16)

        # Decoder with attention gates
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.att4 = AttentionGate(gate_ch=f * 8, skip_ch=f * 8, inter_ch=f * 4)
        self.dec4 = ConvBlock(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.att3 = AttentionGate(gate_ch=f * 4, skip_ch=f * 4, inter_ch=f * 2)
        self.dec3 = ConvBlock(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.att2 = AttentionGate(gate_ch=f * 2, skip_ch=f * 2, inter_ch=f)
        self.dec2 = ConvBlock(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.att1 = AttentionGate(gate_ch=f, skip_ch=f, inter_ch=f // 2)
        self.dec1 = ConvBlock(f * 2, f)

        self.out_conv = nn.Conv2d(f, 1, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with attention-gated skip connections
        up4 = self.up4(b)
        e4 = self.att4(gate=up4, skip=e4)
        d4 = self.dec4(torch.cat([up4, e4], dim=1))

        up3 = self.up3(d4)
        e3 = self.att3(gate=up3, skip=e3)
        d3 = self.dec3(torch.cat([up3, e3], dim=1))

        up2 = self.up2(d3)
        e2 = self.att2(gate=up2, skip=e2)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))

        up1 = self.up1(d2)
        e1 = self.att1(gate=up1, skip=e1)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))

        return self.out_conv(d1)
