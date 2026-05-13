"""
vae_v2.py — Improved VAE with FiLM Class Conditioning.

Inherits the improved autoencoder architecture and adds:
  1. Probabilistic bottleneck (μ, log_var → reparameterize)
  2. FiLM class conditioning in EVERY decoder block
  3. Class embedding → FiLM parameters for strong class influence

The class embedding is NOT concatenated to z. Instead, it modulates
every decoder block via scale+shift. This gives the class 4× more
influence than the old single-concat approach.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from vae.blocks import FiLM


# ═══════════════════════════════════════════════════════════════
#  FiLM Decoder Stage (like DecoderStage but with FiLM conditioning)
# ═══════════════════════════════════════════════════════════════

class FiLMDecoderStage(nn.Module):
    """
    One decoder level with FiLM class conditioning.
    
    Flow:
      h → upsample 2× → conv1 → GN → FiLM(γ,β) → SiLU → conv2 → GN
        → + residual
        → concat encoder_skip
        → project
    """
    def __init__(self, in_ch: int, out_ch: int, enc_skip_ch: int, cond_dim: int):
        super().__init__()
        self.out_ch = out_ch
        self.enc_skip_ch = enc_skip_ch

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.film = FiLM(cond_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(min(32, out_ch), out_ch)

        self.skip_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.proj = nn.Conv2d(out_ch + enc_skip_ch, out_ch, kernel_size=1)

    def forward(self, h: torch.Tensor, cond: torch.Tensor,
                enc_skip: torch.Tensor = None) -> torch.Tensor:
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        residual = self.skip_conv(h)

        h = self.conv1(h)
        h = self.gn1(h)
        gamma, beta = self.film(cond, n_spatial_dims=2)
        h = h * (1.0 + gamma) + beta
        h = F.silu(h)
        h = self.conv2(h)
        h = self.gn2(h)
        h = F.silu(h + residual)

        if enc_skip is not None:
            if h.shape[-2:] != enc_skip.shape[-2:]:
                enc_skip = F.interpolate(enc_skip, size=h.shape[-2:], mode='nearest')
            h = torch.cat([h, enc_skip], dim=1)
            h = self.proj(h)

        return h


# ═══════════════════════════════════════════════════════════════
#  Improved VAE
# ═══════════════════════════════════════════════════════════════

class ImprovedVAE(nn.Module):
    """
    Improved VAE with FiLM class conditioning and skip connections.
    
    Encoder: 1→64→128→256→512 (4 ResEncoderBlocks, SAME as autoencoder)
    Bottleneck: SelfAttention → Flatten → μ/σ → sample z
    Class: Embedding(8, 128) → FiLM in every decoder block
    Decoder: 4 FiLMDecoderStages (512→256→128→64→32) → Conv → 1ch
    """

    def __init__(self, latent_dim: int = 2048, num_classes: int = 8, embed_dim: int = 128,
                 base_channels: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.c4 = c4  # store for decode methods

        # ── Encoder ──────────────────────────────────────
        from vae.blocks import ResEncoderBlock, SelfAttention1D
        self.enc1 = ResEncoderBlock(1, c1)
        self.enc2 = ResEncoderBlock(c1, c2)
        self.enc3 = ResEncoderBlock(c2, c3)
        self.enc4 = ResEncoderBlock(c3, c4)

        self.flat_dim = c4 * 4 * 35
        self.attn = SelfAttention1D(c4, num_heads=4)

        # ── VAE Bottleneck ───────────────────────────────
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flat_dim, latent_dim)

        # ── Class Conditioning ───────────────────────────
        self.class_embed = nn.Embedding(num_classes, embed_dim)

        # ── Decoder with FiLM ────────────────────────────
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)

        self.dec4 = FiLMDecoderStage(c4, c3, c3, embed_dim)
        self.dec3 = FiLMDecoderStage(c3, c2, c2, embed_dim)
        self.dec2 = FiLMDecoderStage(c2, c1, c1, embed_dim)
        self.dec1 = FiLMDecoderStage(c1, base_channels // 2, 0, embed_dim)

        self.output_conv = nn.Conv2d(base_channels // 2, 1, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if m is self.fc_mu or m is self.fc_log_var:
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.zeros_(m.bias)
                else:
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def encode_to_params(self, x: torch.Tensor):
        """Encode spectrogram → (μ, log_var)."""
        s0 = self.enc1(x)
        s1 = self.enc2(s0)
        s2 = self.enc3(s1)
        s3 = self.enc4(s2)

        h = self.attn(s3)
        h = h.flatten(start_dim=1)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        log_var = torch.clamp(log_var, min=-10, max=10)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor, class_emb: torch.Tensor, target_size: tuple):
        """
        Decode latent z + class embedding → spectrogram.
        
        Args:
            z:          [B, latent_dim] — sampled latent
            class_emb:  [B, embed_dim] — class embedding for FiLM
            target_size: (H, W) output spectrogram size
        """
        B = z.shape[0]

        # Need encoder skips. But at generation time, we don't have them!
        # Solution: pass None for all skips → decoder works without them.
        # At TRAINING time (forward pass), we DO have skips from the encoder.

        h = self.fc_decode(z)
        h = h.view(B, self.c4, 4, 35)

        h = self.dec4(h, class_emb)           # [B, c3, 8, 70]
        h = self.dec3(h, class_emb)           # [B, c2, 16, 140]
        h = self.dec2(h, class_emb)           # [B, c1, 32, 280]
        h = self.dec1(h, class_emb)           # [B, c1/2, 64, 560]

        h = self.output_conv(h)
        h = F.interpolate(h, size=target_size, mode='bilinear')
        return h

    def decode_with_skips(self, z: torch.Tensor, class_emb: torch.Tensor,
                          skips: list, target_size: tuple):
        """
        Decode WITH encoder skip connections (used during training/reconstruction).
        skips = [s0 (c1ch), s1 (c2ch), s2 (c3ch)]
        """
        B = z.shape[0]
        h = self.fc_decode(z)
        h = h.view(B, self.c4, 4, 35)

        h = self.dec4(h, class_emb, skips[2])   # enc3 skip (256ch)
        h = self.dec3(h, class_emb, skips[1])   # enc2 skip (128ch)
        h = self.dec2(h, class_emb, skips[0])   # enc1 skip (64ch)
        h = self.dec1(h, class_emb, None)       # no skip

        h = self.output_conv(h)
        h = F.interpolate(h, size=target_size, mode='bilinear')
        return h

    def forward(self, x: torch.Tensor, labels: torch.Tensor, skip_dropout: float = 0.0):
        """
        Full forward pass WITH skip connections (training / reconstruction).
        
        Args:
            skip_dropout: probability of dropping ALL decoder skip connections.
                          Forces decoder to work without encoder skips (generation mode).
                          Ramp from 0.0 (warmup) → 0.5 (full training).

        Returns: (reconstructed, mu, log_var)
        """
        target_size = x.shape[2:]

        # Encode
        s0 = self.enc1(x)
        s1 = self.enc2(s0)
        s2 = self.enc3(s1)
        s3 = self.enc4(s2)
        skips = [s0, s1, s2]  # 64ch, 128ch, 256ch

        h = self.attn(s3)
        h = h.flatten(start_dim=1)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)

        z = self.reparameterize(mu, log_var)
        class_emb = self.class_embed(labels)

        # Randomly drop skip connections during training
        # This forces the decoder to learn generation-capable outputs
        if self.training and skip_dropout > 0 and torch.rand(1).item() < skip_dropout:
            skips = [None, None, None]

        reconstructed = self.decode_with_skips(z, class_emb, skips, target_size)
        return reconstructed, mu, log_var

    @torch.no_grad()
    def sample(self, label, num_samples: int = 1, device='cpu', temperature: float = 1.0):
        """
        Generate new spectrograms from random noise + class.
        No encoder needed — pure generation.
        
        Note: at generation time, we DON'T have encoder skip connections.
        The decoder works without them (just passes None to FiLMDecoderStage).
        """
        self.eval()

        if isinstance(label, int):
            labels = torch.full((num_samples,), label, dtype=torch.long, device=device)
        else:
            labels = torch.tensor(label, dtype=torch.long, device=device)
            num_samples = len(labels)

        # Sample latent from prior
        z = torch.randn(num_samples, self.fc_mu.out_features, device=device) * temperature
        class_emb = self.class_embed(labels)

        generated = self.decode(z, class_emb, target_size=(64, 552))
        return generated


# ═══════════════════════════════════════════════════════════════
#  Quick test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🔧 ImprovedVAE — architecture test\n")
    model = ImprovedVAE(latent_dim=2048, num_classes=8, embed_dim=128)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # Test forward (training)
    x = torch.randn(2, 1, 64, 552)
    labels = torch.tensor([0, 1])
    with torch.no_grad():
        recon, mu, log_var = model(x, labels)
    print(f"   Forward: input={tuple(x.shape)} → output={tuple(recon.shape)} ✅")

    # Test sample (generation)
    gen = model.sample(0, num_samples=1, temperature=1.0)
    print(f"   Sample:  output={tuple(gen.shape)} ✅")
