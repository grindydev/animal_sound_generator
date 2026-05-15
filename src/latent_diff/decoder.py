"""
decoder.py — Small Upsampling Decoder (No Skip Connections).

Takes spatial latent [B, 512, 4, 35] → upsamples 4× → mel [B, 1, 64, 560].
Pure generator — never had skip connections, so can't depend on them.

Design:
  Block: Upsample 2× → Conv → GroupNorm → SiLU
  4 blocks: 512→256→128→64→32
  Output: Conv 32→1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.latent_diff.config import config as cfg


class UpsampleBlock(nn.Module):
    """Upsample 2× + Conv → GN → SiLU. Simple, no skips."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(min(16, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(min(16, out_ch), out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        residual = self.skip(x)

        h = self.conv1(x)
        h = self.gn1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.gn2(h)
        h = F.silu(h + residual)

        return h


class LatentDecoder(nn.Module):
    """
    Upsampling decoder for latent diffusion.

    Input:  [B, bottleneck_ch, 4, 35]
    Output: [B, 1, 64, 560]

    No skip connections. 4 upsample levels.
    Channel multipliers from config control width at each level.
    """

    def __init__(self, bottleneck_ch: int = 256, config=None):
        super().__init__()
        if config is None:
            config = cfg

        # Compute channel sizes: multipliers × bottleneck_ch
        mults = config.decoder_multipliers  # [1.0, 0.5, 0.25, 0.125]
        ch = [max(32, int(bottleneck_ch * m)) for m in mults]

        self.blocks = nn.ModuleList()
        current = bottleneck_ch
        for out_ch in ch:
            self.blocks.append(UpsampleBlock(current, out_ch))
            current = out_ch

        self.output_conv = nn.Conv2d(current, config.spec_channels, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
        """
        Args:
            x: spatial latent [B, 512, H, W]
            target_size: (n_mels, time_frames) — default (64, 552)
        Returns:
            mel [B, 1, target_size]
        """
        h = x
        for block in self.blocks:
            h = block(h)

        h = self.output_conv(h)  # [B, 1, 64, 560]

        if target_size is not None:
            h = F.interpolate(h, size=target_size, mode='bilinear', align_corners=False)

        return h


class ChannelReducer(nn.Module):
    """Reduce encoder bottleneck channels → latent channels."""

    def __init__(self, in_ch: int, out_ch: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ChannelExpander(nn.Module):
    """Expand latent channels back → bottleneck channels."""

    def __init__(self, in_ch: int = 16, out_ch: int = 256):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ═══════════════════════════════════════════════════════════
#  Quick test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🔧 LatentDecoder — architecture test\n")

    decoder = LatentDecoder()
    reducer = ChannelReducer()
    expander = ChannelExpander()

    params = sum(p.numel() for p in decoder.parameters())
    print(f"   Decoder params: {params:,} ({params/1e6:.1f}M)")
    print(f"   Reducer params: {sum(p.numel() for p in reducer.parameters()):,}")
    print(f"   Expander params: {sum(p.numel() for p in expander.parameters()):,}")

    # Test forward
    x = torch.randn(2, 512, 4, 35)
    reduced = reducer(x)
    print(f"\n   Input:  {tuple(x.shape)}")
    print(f"   Latent: {tuple(reduced.shape)}  (for diffusion)")

    expanded = expander(reduced)
    mel = decoder(expanded, target_size=(64, 552))
    print(f"   Output: {tuple(mel.shape)}  ✅" if mel.shape == (2, 1, 64, 552) else f"   ❌ {tuple(mel.shape)}")
