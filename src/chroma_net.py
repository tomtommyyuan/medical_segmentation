"""
CHROMA-Net: a multi-task nuclei network on a pathology foundation encoder.

Three ideas stacked on the U-Net baseline already in this repo.

1. The encoder is a pathology foundation model (UNI or Phikon-v2, ViT-L/16
   trained on order 100M histology tiles) rather than weights learned from
   PanNuke's 7901 patches. It is adapted with LoRA on the attention
   projections, so under 5% of the encoder's parameters move and the whole
   model fits comfortably on one GPU.

   A plain ViT is a poor fit for nuclei: its tokens are stride 16, so a 256x256
   patch becomes a 16x16 grid while a nucleus is roughly 20px across. Tokens
   are therefore taken from four blocks spread through the depth, and a
   full-resolution convolutional stem on the raw image supplies the fine detail
   the transformer never represents. Without that stem, boundaries between
   touching nuclei are lost and PQ collapses.

2. The decoder is shared and the three tasks sit on light heads: nuclei
   foreground, horizontal/vertical distance maps, and nucleus class. Separate
   full decoders per branch (as in HoVer-Net and CellViT) triple the decoder
   parameters; the shared trunk keeps the model small enough to train three
   splits in a day.

3. Decoder features are FiLM-modulated by an embedding of the patch's tissue
   type. PanNuke ships that label as part of the benchmark input, so using it
   is within protocol, but it is reported separately and can be disabled with
   --no-tissue-film so the unconditioned number is always available.

A convolutional encoder is included as a fallback for environments with no
model-hub access, and doubles as the "no foundation model" ablation row.

References: Chen et al., "UNI", Nature Medicine 2024; Filiot et al.,
"Phikon-v2", 2024; Horst et al., "CellViT", Medical Image Analysis 2024;
Hu et al., "LoRA", ICLR 2022; Perez et al., "FiLM", AAAI 2018.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import TISSUE_TYPES
from preprocess import NUM_TYPES

# Encoder blocks whose tokens feed the decoder skips, shallow to deep. Spread
# through a 24-block ViT-L so the skips carry different levels of abstraction.
VIT_L_BLOCKS = (5, 11, 17, 23)

# Decoder widths: bottleneck first, then one per upsampling stage.
DECODER_CHANNELS = (512, 256, 128, 64, 32)
STEM_CHANNELS = 32


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with batch norm, as in unet.py."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LoRALinear(nn.Module):
    """
    Low-rank adapter around a frozen linear layer.

    The frozen weight keeps whatever the foundation model learned from millions
    of slides; the rank-r update is the only thing PanNuke gets to move. B is
    initialised to zero so the adapted model starts exactly at the pretrained
    one.
    """

    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False

        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        self.scaling = alpha / rank

    def forward(self, x):
        update = F.linear(F.linear(x, self.lora_a), self.lora_b)
        return self.base(x) + update * self.scaling


def apply_lora(module, rank=8, alpha=16, targets=("qkv", "query", "key", "value")):
    """
    Replace the attention projections of a transformer with LoRA-wrapped ones.

    Args:
        module: the encoder to adapt, modified in place
        rank: adapter rank
        alpha: adapter scaling
        targets: substrings of child names to wrap

    Returns:
        number of layers wrapped
    """
    wrapped = 0

    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(t in name for t in targets):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            wrapped += 1
        else:
            wrapped += apply_lora(child, rank=rank, alpha=alpha, targets=targets)

    return wrapped


class ConvEncoder(nn.Module):
    """
    Convolutional encoder, for environments without model-hub access.

    Also serves as the "no foundation model" ablation: same decoder, same
    losses, same training recipe, encoder learned from PanNuke alone.
    """

    feature_channels = (64, 128, 256, 512)

    def __init__(self, in_channels=3, base_filters=64):
        super().__init__()
        f = base_filters
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f, f * 2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.enc4 = ConvBlock(f * 4, f * 8)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(self.pool(x))        # stride 2
        e2 = self.enc2(self.pool(e1))       # stride 4
        e3 = self.enc3(self.pool(e2))       # stride 8
        e4 = self.enc4(self.pool(e3))       # stride 16
        return [e1, e2, e3, e4]


class TimmViTEncoder(nn.Module):
    """
    A timm vision transformer, returning tokens from four blocks as feature maps.

    Used for UNI (hf-hub:MahmoodLab/UNI), which is gated but free for academic
    use, and for any timm ViT as a stand-in.
    """

    def __init__(self, model_name, img_size=256, blocks=VIT_L_BLOCKS, pretrained=True):
        super().__init__()
        import timm

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            dynamic_img_size=True,
        )
        self.blocks = tuple(blocks)
        self.patch_size = self.model.patch_embed.patch_size[0]

        embed_dim = self.model.embed_dim
        self.feature_channels = (embed_dim,) * len(self.blocks)

    def forward(self, x):
        return list(
            self.model.get_intermediate_layers(x, n=self.blocks, reshape=True, norm=True)
        )


class HFViTEncoder(nn.Module):
    """
    A HuggingFace transformers vision model, returning four hidden states.

    Used for Phikon-v2 (owkin/phikon-v2), which needs no access request and is
    the default so the repo runs end to end without a gated download.
    """

    def __init__(self, model_name, blocks=VIT_L_BLOCKS, pretrained=True):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            self.model = AutoModel.from_config(AutoConfig.from_pretrained(model_name))

        self.blocks = tuple(blocks)
        self.patch_size = self.model.config.patch_size

        embed_dim = self.model.config.hidden_size
        self.feature_channels = (embed_dim,) * len(self.blocks)

    def forward(self, x):
        height = x.shape[2] // self.patch_size
        width = x.shape[3] // self.patch_size

        outputs = self.model(
            pixel_values=x,
            output_hidden_states=True,
            interpolate_pos_encoding=True,
        )

        features = []
        for index in self.blocks:
            # hidden_states[0] is the embedding output, so block i is at i + 1.
            tokens = outputs.hidden_states[index + 1]

            # Drop CLS and any register tokens, whose count varies by model.
            n_prefix = tokens.shape[1] - height * width
            tokens = tokens[:, n_prefix:, :]

            features.append(
                tokens.transpose(1, 2).reshape(tokens.shape[0], -1, height, width)
            )

        return features


class FiLM(nn.Module):
    """
    Feature-wise linear modulation from a tissue embedding.

    Predicts a per-channel scale and shift, so the decoder can specialise its
    features for skin versus liver without a separate model per tissue. The
    scale is centred on 1 and the shift on 0, so an untrained FiLM layer is the
    identity and adding it cannot hurt the starting point.
    """

    def __init__(self, embed_dim, num_channels):
        super().__init__()
        self.to_scale = nn.Linear(embed_dim, num_channels)
        self.to_shift = nn.Linear(embed_dim, num_channels)
        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)

    def forward(self, x, embedding):
        scale = 1.0 + self.to_scale(embedding).unsqueeze(-1).unsqueeze(-1)
        shift = self.to_shift(embedding).unsqueeze(-1).unsqueeze(-1)
        return x * scale + shift


class ChromaDecoder(nn.Module):
    """
    Shared decoder: bottleneck, four upsampling stages, encoder and stem skips.

    Encoder features are resized to each stage's resolution before being
    projected, which lets the same decoder sit on a ViT (every feature at
    stride 16) or on a convolutional encoder (features at strides 2 to 16).
    """

    def __init__(self, feature_channels, channels=DECODER_CHANNELS,
                 stem_channels=STEM_CHANNELS, tissue_embed_dim=0):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = ConvBlock(feature_channels[3], channels[0])

        # Skip source per stage: three encoder features, then the stem at full
        # resolution. Stage i upsamples by 2, so stage 3 lands back at 1:1.
        skip_channels = [feature_channels[2], feature_channels[1],
                         feature_channels[0], stem_channels]

        self.ups = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.films = nn.ModuleList()

        for i in range(4):
            out_ch = channels[i + 1]
            self.ups.append(nn.ConvTranspose2d(channels[i], out_ch, 2, stride=2))
            self.skips.append(nn.Conv2d(skip_channels[i], out_ch, 1, bias=False))
            self.decoders.append(ConvBlock(out_ch * 2, out_ch))
            self.films.append(FiLM(tissue_embed_dim, out_ch) if tissue_embed_dim else None)

        self.out_channels = channels[-1]

    def forward(self, image, features, tissue_embedding=None):
        skips = [features[2], features[1], features[0], self.stem(image)]

        x = self.bottleneck(features[3])

        for i in range(4):
            x = self.ups[i](x)

            skip = skips[i]
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear",
                                     align_corners=False)

            x = self.decoders[i](torch.cat([x, self.skips[i](skip)], dim=1))

            if self.films[i] is not None and tissue_embedding is not None:
                x = self.films[i](x, tissue_embedding)

        return x


class ChromaNet(nn.Module):
    """
    Foundation-model encoder, shared decoder, three task heads.

    Args:
        encoder: "phikon", "uni", "conv", or an explicit hub identifier
        img_size: input resolution
        lora_rank: rank of the attention adapters, 0 to leave the encoder frozen
        tissue_film: condition the decoder on the patch's tissue type
        tissue_embed_dim: width of the tissue embedding
        num_types: number of nuclei classes
        pretrained: load pretrained encoder weights

    Returns from forward a dict with:
        np: (B, 1, H, W) nuclei logits
        hv: (B, 2, H, W) horizontal and vertical distance maps
        tp: (B, num_types + 1, H, W) class logits, channel 0 is background
    """

    ENCODER_ALIASES = {
        "phikon": ("hf", "owkin/phikon-v2"),
        "uni": ("timm", "hf-hub:MahmoodLab/UNI"),
        "conv": ("conv", None),
    }

    def __init__(self, encoder="phikon", img_size=256, lora_rank=8, lora_alpha=16,
                 tissue_film=True, tissue_embed_dim=64, num_types=NUM_TYPES,
                 pretrained=True, freeze_encoder=False):
        super().__init__()

        kind, name = self.ENCODER_ALIASES.get(encoder, ("timm", encoder))

        if kind == "conv":
            self.encoder = ConvEncoder()
        elif kind == "hf":
            self.encoder = HFViTEncoder(name, pretrained=pretrained)
        else:
            self.encoder = TimmViTEncoder(name, img_size=img_size, pretrained=pretrained)

        self.encoder_kind = kind
        self.lora_layers = 0

        if kind != "conv":
            if freeze_encoder or lora_rank > 0:
                for param in self.encoder.parameters():
                    param.requires_grad = False
            if lora_rank > 0:
                self.lora_layers = apply_lora(self.encoder, rank=lora_rank, alpha=lora_alpha)

        self.tissue_embedding = (
            nn.Embedding(len(TISSUE_TYPES), tissue_embed_dim) if tissue_film else None
        )

        self.decoder = ChromaDecoder(
            self.encoder.feature_channels,
            tissue_embed_dim=tissue_embed_dim if tissue_film else 0,
        )

        width = self.decoder.out_channels
        self.np_head = self._make_head(width, 1)
        self.hv_head = self._make_head(width, 2)
        self.tp_head = self._make_head(width, num_types + 1)

    @staticmethod
    def _make_head(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
        )

    def forward(self, image, tissue=None):
        features = self.encoder(image)

        embedding = None
        if self.tissue_embedding is not None and tissue is not None:
            embedding = self.tissue_embedding(tissue)

        decoded = self.decoder(image, features, embedding)

        return {
            "np": self.np_head(decoded),
            "hv": torch.tanh(self.hv_head(decoded)),
            "tp": self.tp_head(decoded),
        }

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_summary(self):
        """Return (trainable, total) parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total
