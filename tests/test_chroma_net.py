"""
Tests for the CHROMA-Net architecture.

Runs on the convolutional encoder, and on a small randomly initialised ViT for
the transformer plumbing, so nothing here needs a model-hub download.
"""

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from chroma_net import ChromaNet, ConvEncoder, FiLM, LoRALinear, apply_lora
from dataset import TISSUE_TYPES
from preprocess import NUM_TYPES


def test_conv_encoder_returns_a_stride_2_to_16_pyramid():
    features = ConvEncoder()(torch.randn(1, 3, 256, 256))

    assert [tuple(f.shape) for f in features] == [
        (1, 64, 128, 128),
        (1, 128, 64, 64),
        (1, 256, 32, 32),
        (1, 512, 16, 16),
    ]


def test_forward_returns_all_three_branches_at_full_resolution():
    model = ChromaNet(encoder="conv")
    out = model(torch.randn(2, 3, 256, 256), torch.tensor([3, 7]))

    assert tuple(out["np"].shape) == (2, 1, 256, 256)
    assert tuple(out["hv"].shape) == (2, 2, 256, 256)
    assert tuple(out["tp"].shape) == (2, NUM_TYPES + 1, 256, 256)


def test_distance_maps_are_bounded_to_the_target_range():
    model = ChromaNet(encoder="conv")
    with torch.no_grad():
        hv = model(torch.randn(2, 3, 256, 256), torch.tensor([0, 1]))["hv"]

    assert float(hv.min()) >= -1.0
    assert float(hv.max()) <= 1.0


def test_model_runs_without_a_tissue_label():
    model = ChromaNet(encoder="conv", tissue_film=False)
    out = model(torch.randn(1, 3, 256, 256))

    assert tuple(out["np"].shape) == (1, 1, 256, 256)


def test_film_conditioning_starts_as_the_identity():
    # An untrained FiLM layer must not perturb the model, so enabling tissue
    # conditioning can only help from the starting point onwards.
    model = ChromaNet(encoder="conv", tissue_film=True).eval()
    x = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        first = model(x, torch.zeros(2, dtype=torch.long))["np"]
        last = model(x, torch.full((2,), len(TISSUE_TYPES) - 1, dtype=torch.long))["np"]

    assert torch.allclose(first, last, atol=1e-6)


def test_film_can_actually_modulate_once_trained():
    film = FiLM(8, 4)
    nn.init.normal_(film.to_scale.weight, std=0.5)
    nn.init.normal_(film.to_shift.weight, std=0.5)

    x = torch.randn(2, 4, 6, 6)
    out = film(x, torch.randn(2, 8))

    assert not torch.allclose(out, x)


def test_lora_starts_as_the_identity_and_freezes_the_base():
    base = nn.Linear(32, 64)
    adapted = LoRALinear(copy.deepcopy(base), rank=4)
    x = torch.randn(5, 32)

    assert torch.allclose(base(x), adapted(x), atol=1e-6)
    assert not any(p.requires_grad for p in adapted.base.parameters())
    assert adapted.lora_a.requires_grad and adapted.lora_b.requires_grad


def test_lora_changes_the_output_once_its_weights_move():
    base = nn.Linear(32, 64)
    adapted = LoRALinear(copy.deepcopy(base), rank=4)
    nn.init.normal_(adapted.lora_b, std=0.1)

    x = torch.randn(5, 32)
    assert not torch.allclose(base(x), adapted(x), atol=1e-4)


def test_apply_lora_only_wraps_the_attention_projections():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(16, 48)
            self.mlp = nn.Linear(16, 16)

    module = nn.Sequential(Block(), Block())
    wrapped = apply_lora(module, rank=4)

    assert wrapped == 2
    assert isinstance(module[0].qkv, LoRALinear)
    assert isinstance(module[0].mlp, nn.Linear) and not isinstance(module[0].mlp, LoRALinear)


def test_lora_leaves_most_of_the_encoder_frozen():
    timm = pytest.importorskip("timm")

    model = ChromaNet(encoder="vit_small_patch16_224", pretrained=False, lora_rank=8)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)

    assert model.lora_layers > 0
    assert encoder_trainable / encoder_params < 0.05


def test_vit_encoder_tokens_reshape_to_a_square_grid():
    timm = pytest.importorskip("timm")

    from chroma_net import TimmViTEncoder

    encoder = TimmViTEncoder("vit_small_patch16_224", img_size=256,
                             blocks=(2, 5, 8, 11), pretrained=False)
    features = encoder(torch.randn(1, 3, 256, 256))

    assert len(features) == 4
    # 256 / 16 = 16 tokens a side.
    assert all(tuple(f.shape) == (1, 384, 16, 16) for f in features)


def test_vit_backbone_reaches_full_resolution_through_the_decoder():
    timm = pytest.importorskip("timm")

    model = ChromaNet(encoder="vit_small_patch16_224", pretrained=False)
    model.encoder.blocks = (2, 5, 8, 11)
    out = model(torch.randn(1, 3, 256, 256), torch.tensor([1]))

    assert tuple(out["np"].shape) == (1, 1, 256, 256)


def test_gradients_reach_the_decoder_and_the_adapters_only():
    model = ChromaNet(encoder="conv")
    out = model(torch.randn(1, 3, 256, 256), torch.tensor([0]))
    (out["np"].mean() + out["hv"].mean() + out["tp"].mean()).backward()

    assert model.decoder.bottleneck.block[0].weight.grad is not None
    assert model.np_head[0].weight.grad is not None


def test_parameter_summary_counts_only_trainable_weights():
    frozen = ChromaNet(encoder="conv")
    for param in frozen.encoder.parameters():
        param.requires_grad = False

    trainable, total = frozen.parameter_summary()

    assert trainable < total
    assert trainable == sum(p.numel() for p in frozen.parameters() if p.requires_grad)
