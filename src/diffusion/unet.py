"""
unet.py — U-Net for Spectrogram Diffusion (Phase 7b).

A small U-Net (~20M params) that predicts noise for diffusion-based
spectrogram refinement. Conditioned on timestep (sinusoidal embedding)
and animal class (learned embedding).

Architecture (standard U-Net with skip connections):
  - Input projection: 1ch → base_ch
  - Encoder: 4 levels, each: ResBlocks → save skip → Downsample(2×)
  - Bottleneck: ResBlocks + attention at deepest resolution
  - Decoder: 4 levels, each: concat skip → ResBlocks → Upsample(2×) [except last]
  - Output projection: base_ch → 1ch

  Skip connections: encoder level i output → decoder level i input
  (same resolution — skip is saved BEFORE downsampling, decoder
   receives it at the same resolution after previous level's upsample)

Input:  [B, 1, 64, W] noisy spectrogram + timestep t + class label
Output: [B, 1, 64, W] predicted noise
"""
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.diffusion.config import config as cfg


# ═══════════════════════════════════════════════════════════════
#  Sinusoidal Time Embedding
# ═══════════════════════════════════════════════════════════════

class SinusoidalTimeEmbedding(nn.Module):
    """Maps integer timestep t to a continuous vector."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return self.mlp(emb)


# ═══════════════════════════════════════════════════════════════
#  Self-Attention (bottleneck levels)
# ═══════════════════════════════════════════════════════════════

class SelfAttention2D(nn.Module):
    """Lightweight multi-head self-attention for 2D feature maps."""
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        scale = self.head_dim ** -0.5
        attn = (q.transpose(-2, -1) @ k) * scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        out = out.reshape(B, C, H, W)
        return self.proj(out)


# ═══════════════════════════════════════════════════════════════
#  ResBlock with Time + Class Conditioning (FiLM-style)
# ═══════════════════════════════════════════════════════════════

def _num_groups(n_channels: int, max_groups: int = 32) -> int:
    """Find largest divisor of n_channels ≤ max_groups for GroupNorm."""
    for g in range(min(max_groups, n_channels), 0, -1):
        if n_channels % g == 0:
            return g
    return 1

class ResBlock(nn.Module):
    """Residual block with GroupNorm, SiLU, and FiLM conditioning."""
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, class_emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # FiLM: time/class → scale + shift for each channel
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch * 2))
        self.class_mlp = nn.Sequential(nn.SiLU(), nn.Linear(class_emb_dim, out_ch * 2))

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # FiLM: scale and shift
        t_out = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        t_scale, t_shift = t_out.chunk(2, dim=1)
        c_out = self.class_mlp(c_emb).unsqueeze(-1).unsqueeze(-1)
        c_scale, c_shift = c_out.chunk(2, dim=1)
        h = h * (1 + t_scale + c_scale) + t_shift + c_shift

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


# ═══════════════════════════════════════════════════════════════
#  Downsample / Upsample
# ═══════════════════════════════════════════════════════════════

class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ═══════════════════════════════════════════════════════════════
#  U-Net
# ═══════════════════════════════════════════════════════════════

class SpectrogramUNet(nn.Module):
    """
    U-Net for spectrogram diffusion denoising.

    Encoder:  ResBlocks → save skip → Downsample
    Decoder:  concat skip → ResBlocks → Upsample (except last)
    """
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = cfg

        base_ch = config.base_channels
        ch_mults = config.channel_multipliers   # e.g. (1, 2, 4, 4)
        n_levels = len(ch_mults)
        time_out_dim = config.time_emb_dim * 4

        # Input projection
        self.input_proj = nn.Conv2d(config.spec_channels, base_ch, kernel_size=3, padding=1)

        # Time + class embeddings
        self.time_embed = SinusoidalTimeEmbedding(config.time_emb_dim)
        self.class_embed = nn.Embedding(config.num_classes + 1, config.class_emb_dim)  # +1 for null (uncond)
        self.null_class_idx = config.num_classes

        # ── Encoder ──────────────────────────────────────
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()   # attention at selected encoder levels
        self.downsamples = nn.ModuleList()
        in_ch = base_ch

        for i, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            level = nn.ModuleList()
            for _ in range(config.res_blocks_per_level):
                level.append(ResBlock(in_ch, out_ch, time_out_dim, config.class_emb_dim, config.dropout))
                in_ch = out_ch
            self.encoder_blocks.append(level)
            # Attention at this encoder level if configured
            if i in config.attention_levels:
                self.encoder_attns.append(SelfAttention2D(out_ch, num_heads=8))
            else:
                self.encoder_attns.append(nn.Identity())
            if i < n_levels - 1:
                self.downsamples.append(Downsample(out_ch))

        # ── Bottleneck ──────────────────────────────────
        self.bottleneck = nn.ModuleList()
        for _ in range(config.res_blocks_per_level):
            self.bottleneck.append(ResBlock(in_ch, in_ch, time_out_dim, config.class_emb_dim, config.dropout))
        self.bottleneck_attn = SelfAttention2D(in_ch, num_heads=8)

        # ── Decoder ─────────────────────────────────────
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()   # attention at selected decoder levels
        self.upsamples = nn.ModuleList()

        ch_list = [base_ch * m for m in ch_mults]  # encoder channel list
        for i in reversed(range(n_levels)):
            out_ch = ch_list[i]

            if i < n_levels - 1:
                self.upsamples.append(Upsample(in_ch))

            level = nn.ModuleList()
            skip_ch = ch_list[i]
            for j in range(config.res_blocks_per_level):
                block_in = (in_ch + skip_ch) if j == 0 else out_ch
                level.append(ResBlock(block_in, out_ch, time_out_dim, config.class_emb_dim, config.dropout))
            self.decoder_blocks.append(level)
            # Attention at this decoder level if configured
            if i in config.attention_levels:
                self.decoder_attns.append(SelfAttention2D(out_ch, num_heads=8))
            else:
                self.decoder_attns.append(nn.Identity())
            in_ch = out_ch

        # ── Output ──────────────────────────────────────
        self.output_proj = nn.Sequential(
            nn.GroupNorm(_num_groups(in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, config.spec_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        # Handle null label (unconditional) — map to null embedding
        labels = labels.clamp(0, self.null_class_idx)
        c_emb = self.class_embed(labels)
        n_levels = len(self.encoder_blocks)

        h = self.input_proj(x)

        # ── Encoder ──────────────────────────────────────
        skips = []
        for i in range(n_levels):
            for block in self.encoder_blocks[i]:
                h = block(h, t_emb, c_emb)
            h = self.encoder_attns[i](h)   # attention at this level
            skips.append(h)
            if i < n_levels - 1:
                h = self.downsamples[i](h)

        # ── Bottleneck ──────────────────────────────────
        for block in self.bottleneck:
            h = block(h, t_emb, c_emb)
        h = self.bottleneck_attn(h)

        # ── Decoder ─────────────────────────────────────
        for dec_idx in range(n_levels):
            if dec_idx > 0:
                h = self.upsamples[dec_idx - 1](h)

            skip = skips[n_levels - 1 - dec_idx]
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode='nearest')

            h = torch.cat([h, skip], dim=1)

            for block in self.decoder_blocks[dec_idx]:
                h = block(h, t_emb, c_emb)
            h = self.decoder_attns[dec_idx](h)  # attention at this level

        # ── Output ──────────────────────────────────────
        return self.output_proj(h)


# ═══════════════════════════════════════════════════════════════
#  Quick test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🔧 SpectrogramUNet — architecture test\n")

    model = SpectrogramUNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # Test forward pass
    for w in [552, 400, 300]:
        B, C, H = 2, 1, 64
        x = torch.randn(B, C, H, w)
        t = torch.randint(0, 1000, (B,))
        labels = torch.randint(0, 8, (B,))

        with torch.no_grad():
            out = model(x, t, labels)

        ok = "✅" if x.shape == out.shape else "❌"
        print(f"   [{B}, {C}, {H:2d}, {w:3d}] → {tuple(out.shape)} {ok}")

    # Parameter breakdown
    enc_params = sum(p.numel() for n, p in model.named_parameters() if 'encoder' in n or 'downsample' in n)
    dec_params = sum(p.numel() for n, p in model.named_parameters() if 'decoder' in n or 'upsample' in n)
    bot_params = sum(p.numel() for n, p in model.named_parameters() if 'bottleneck' in n)
    print(f"\n   Encoder:     {enc_params:,}")
    print(f"   Bottleneck:  {bot_params:,}")
    print(f"   Decoder:     {dec_params:,}")
