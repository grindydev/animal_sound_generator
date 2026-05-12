"""
autoencoder_v2.py — Improved Autoencoder with Skip Connections.

Key improvements over SimpleAudioAutoencoder:
  1. Residual encoder blocks — better gradient flow
  2. Skip connections from encoder to decoder — detail preservation
  3. Self-attention at bottleneck — temporal coherence
  4. 2× larger latent (2048 vs 1024) — less compression
  5. 2× deeper (512 max channels vs 256)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from vae.blocks import ResEncoderBlock


# ═══════════════════════════════════════════════════════════════
#  Decoder Stage (one upsampling level)
# ═══════════════════════════════════════════════════════════════

class DecoderStage(nn.Module):
    """
    One decoder level: upsample 2× → 2 convs → concat encoder skip → project.
    """
    def __init__(self, in_ch: int, out_ch: int, enc_skip_ch: int):
        super().__init__()
        self.out_ch = out_ch
        self.enc_skip_ch = enc_skip_ch

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(min(32, out_ch), out_ch)

        # Residual connection (1×1 conv on upsampled input)
        self.skip_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        # Projection after concat with encoder skip
        self.proj = nn.Conv2d(out_ch + enc_skip_ch, out_ch, kernel_size=1)

    def forward(self, h: torch.Tensor, enc_skip: torch.Tensor = None) -> torch.Tensor:
        # Upsample
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        residual = self.skip_conv(h)

        h = self.conv1(h)
        h = self.gn1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.gn2(h)
        h = F.silu(h + residual)

        # Concat encoder skip
        if enc_skip is not None:
            if h.shape[-2:] != enc_skip.shape[-2:]:
                enc_skip = F.interpolate(enc_skip, size=h.shape[-2:], mode='nearest')
            h = torch.cat([h, enc_skip], dim=1)
            h = self.proj(h)

        return h


# ═══════════════════════════════════════════════════════════════
#  Improved Autoencoder
# ═══════════════════════════════════════════════════════════════

class ImprovedAutoencoder(nn.Module):
    """
    Improved autoencoder with residual blocks, skip connections, and attention.
    
    Encoder: 1→64→128→256→512 (4 ResEncoderBlocks)
    Bottleneck: SelfAttention → Flatten → Linear → latent_dim
    Decoder: Linear → 4 DecoderStages (512→256→128→64→32) → Conv → 1ch
    """

    def __init__(self, latent_dim: int = 2048, base_channels: int = 32):
        super().__init__()

        # Channel progression: base → 2*base → 4*base → 8*base
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.c4 = c4

        # ── Encoder ──────────────────────────────────────
        self.enc1 = ResEncoderBlock(1, c1)
        self.enc2 = ResEncoderBlock(c1, c2)
        self.enc3 = ResEncoderBlock(c2, c3)
        self.enc4 = ResEncoderBlock(c3, c4)

        # ── Bottleneck ───────────────────────────────────
        # After 4 stride-2: 552→276→138→69→35 → [B, c4, 4, 35]
        self.flat_dim = c4 * 4 * 35

        # Self-attention
        from vae.blocks import SelfAttention1D
        self.attn = SelfAttention1D(c4, num_heads=4)

        self.fc_encode = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)

        # ── Decoder ──────────────────────────────────────
        # Levels: (in_ch, out_ch, enc_skip_ch)
        # enc_skip_ch matches the encoder output channels at the corresponding level
        self.dec4 = DecoderStage(c4, c3, c3)   # skip from enc3
        self.dec3 = DecoderStage(c3, c2, c2)   # skip from enc2
        self.dec2 = DecoderStage(c2, c1, c1)   # skip from enc1
        self.dec1 = DecoderStage(c1, base_channels // 2, 0)  # final level, no skip

        # Final output conv
        self.output_conv = nn.Conv2d(base_channels // 2, 1, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if m is self.fc_encode:
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor):
        """Returns (z, [skip0, skip1, skip2, skip3])."""
        s0 = self.enc1(x)       # [B, 64, 32, 276]
        s1 = self.enc2(s0)      # [B, 128, 16, 138]
        s2 = self.enc3(s1)      # [B, 256, 8, 69]
        s3 = self.enc4(s2)      # [B, 512, 4, 35]

        h = self.attn(s3)
        h = h.flatten(start_dim=1)  # [B, 71680]
        z = self.fc_encode(h)       # [B, latent_dim]

        return z, [s0, s1, s2, s3]

    def decode(self, z: torch.Tensor, skips: list, target_size: tuple):
        """skips = [s0 (64ch), s1 (128ch), s2 (256ch), s3 (512ch)]"""
        B = z.shape[0]

        h = self.fc_decode(z)             # [B, 71680]
        h = h.view(B, self.c4, 4, 35)         # [B, c4, 4, 35]

        h = self.dec4(h, skips[2])        # [B, 256, 8, 70]
        h = self.dec3(h, skips[1])        # [B, 128, 16, 140]
        h = self.dec2(h, skips[0])        # [B, 64, 32, 280]
        h = self.dec1(h, None)            # [B, 32, 64, 560]

        h = self.output_conv(h)           # [B, 1, 64, 560]
        h = F.interpolate(h, size=target_size, mode='bilinear')

        return h

    def forward(self, x: torch.Tensor):
        target_size = x.shape[2:]
        z, skips = self.encode(x)
        return self.decode(z, skips, target_size)


# ═══════════════════════════════════════════════════════════════
#  Quick test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🔧 ImprovedAutoencoder — architecture test\n")
    model = ImprovedAutoencoder(latent_dim=2048)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    x = torch.randn(2, 1, 64, 552)
    with torch.no_grad():
        z, skips = model.encode(x)
        out = model.decode(z, skips, (64, 552))

    ok = "✅" if x.shape == out.shape else "❌"
    print(f"   Input:  {tuple(x.shape)}")
    print(f"   Latent: {tuple(z.shape)}")
    print(f"   Output: {tuple(out.shape)} {ok}")
    print(f"   Skips:  {[tuple(s.shape) for s in skips]}")
