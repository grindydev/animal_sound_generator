"""
unet.py — Tiny UNet for Latent Diffusion.

Input: [B, 16, 4, 35] + timestep + class label → predicted noise [B, 16, 4, 35].

Architecture (~3M params):
  Encoder: 3 levels (16→64→128→256)
  Bottleneck: self-attention
  Decoder: 3 levels (256→128→64→16)

Reuses ResBlock and SelfAttention2D from src/diffusion/unet.py.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.latent_diff.config import config as cfg
from src.diffusion.unet import ResBlock, SelfAttention2D, SinusoidalTimeEmbedding


class Downsample2D(nn.Module):
    """2D downsampling via strided conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2D(nn.Module):
    """2D upsampling via nearest neighbor + conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class LatentUNet(nn.Module):
    """
    Tiny UNet for latent diffusion.

    Input:  [B, latent_channels, H, W] = [B, 16, 4, 35]
    Output: [B, 16, 4, 35] (predicted noise)
    """

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = cfg

        base_ch = config.unet_base_channels
        ch_mults = config.unet_channel_multipliers  # (1, 2, 2)
        n_levels = len(ch_mults)
        time_out_dim = config.time_emb_dim * 4
        n_classes = config.num_classes + 1  # +1 for null label (CFG)

        # Input projection
        self.input_proj = nn.Conv2d(config.latent_channels, base_ch, kernel_size=3, padding=1)

        # Time + class embeddings
        self.time_embed = SinusoidalTimeEmbedding(config.time_emb_dim)
        self.class_embed = nn.Embedding(n_classes, config.class_emb_dim)
        self.null_class_idx = config.num_classes

        # ── Encoder ──────────────────────────────────
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = base_ch

        for i, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            self.encoder_blocks.append(ResBlock(
                in_ch, out_ch, time_out_dim, config.class_emb_dim, config.dropout
            ))
            if i < n_levels - 1:
                self.downsamples.append(Downsample2D(out_ch))
            in_ch = out_ch

        # ── Bottleneck ───────────────────────────────
        self.bottleneck = ResBlock(in_ch, in_ch, time_out_dim, config.class_emb_dim, config.dropout)
        self.bottleneck_attn = SelfAttention2D(in_ch, num_heads=8)

        # ── Decoder ─────────────────────────────────
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # Reverse encoder channel list
        ch_list = [base_ch * m for m in ch_mults]

        for i in reversed(range(n_levels)):
            out_ch = ch_list[i]
            skip_ch = ch_list[i]

            if i < n_levels - 1:
                self.upsamples.append(Upsample2D(in_ch))

            block_in = in_ch + skip_ch  # concat skip
            self.decoder_blocks.append(ResBlock(
                block_in, out_ch, time_out_dim, config.class_emb_dim, config.dropout
            ))
            in_ch = out_ch

        # ── Output ──────────────────────────────────
        self.output_proj = nn.Sequential(
            nn.GroupNorm(min(16, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, config.latent_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        labels = labels.clamp(0, self.null_class_idx)
        c_emb = self.class_embed(labels)

        h = self.input_proj(x)
        n_levels = len(self.encoder_blocks)
        skips = []

        # Encoder
        for i in range(n_levels):
            h = self.encoder_blocks[i](h, t_emb, c_emb)
            skips.append(h)
            if i < n_levels - 1:
                h = self.downsamples[i](h)

        # Bottleneck
        h = self.bottleneck(h, t_emb, c_emb)
        h = self.bottleneck_attn(h)

        # Decoder
        for dec_idx in range(n_levels):
            if dec_idx > 0:
                h = self.upsamples[dec_idx - 1](h)

            skip = skips[n_levels - 1 - dec_idx]
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode='nearest')

            h = torch.cat([h, skip], dim=1)
            h = self.decoder_blocks[dec_idx](h, t_emb, c_emb)

        return self.output_proj(h)


# ═══════════════════════════════════════════════════════════
#  Quick test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🔧 LatentUNet — architecture test\n")

    model = LatentUNet()
    params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {params:,} ({params/1e6:.1f}M)")

    # Test forward
    x = torch.randn(2, 16, 4, 35)
    t = torch.randint(0, 1000, (2,))
    labels = torch.randint(0, 8, (2,))

    with torch.no_grad():
        out = model(x, t, labels)

    ok = "✅" if x.shape == out.shape else "❌"
    print(f"   Input:  {tuple(x.shape)}")
    print(f"   Output: {tuple(out.shape)} {ok}")

    # Test null label
    null_labels = torch.full((2,), 8, dtype=torch.long)
    out_null = model(x, t, null_labels)
    diff = (out_null - out).abs().mean().item()
    print(f"   Cond vs null diff: {diff:.4f} (should be >0) {'✅' if diff > 0 else '⚠️'}")
