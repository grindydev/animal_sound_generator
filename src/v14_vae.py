"""
v14_vae.py — V14 Lightweight VAE (no skip connections)
=======================================================

Ultra-compact VAE designed for 640 training samples:
  - ~1.5M params (appropriate for small dataset)
  - NO encoder skip connections → decoder works for generation
  - Self-attention at bottleneck only
  - 256-dim latent

Architecture:
  Encoder: mel [1,64,552] → 4 Conv2d → [64,4,35] → attn → flatten → μ,σ → z [256]
  Decoder: z [256] + class → Linear → [64,4,35] → 4 ConvTranspose2d → mel [1,64,552]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttn(nn.Module):
    """1D self-attention along spatial dims (treats [H*W] as sequence)."""
    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads = heads
        self.dim = ch // heads
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.heads, self.dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = F.softmax((q.transpose(-2, -1) @ k) * (self.dim ** -0.5), dim=-1)
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        return self.proj(out.reshape(B, C, H, W))


class ResBlk(nn.Module):
    """Residual conv block."""
    def __init__(self, ch):
        super().__init__()
        g = ch if ch < 32 else 32
        while ch % g != 0: g -= 1
        self.n1 = nn.GroupNorm(g, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(g, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = F.silu(self.n1(x))
        h = self.c1(h)
        h = F.silu(self.n2(h))
        h = self.c2(h)
        return x + h


class V14VAE(nn.Module):
    """
    Tiny VAE for mel compression. ~1.5M params.
    
    NO encoder → decoder skip connections.
    Decoder uses FiLM class conditioning + self-attention.
    """
    def __init__(self, latent_dim=256, num_classes=7, class_emb_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # ── Encoder (4 downsample stages) ─────────────────
        # Input: [B, 1, 64, 552]
        self.enc_conv_in = nn.Conv2d(1, 16, 3, padding=1)
        # Stage 1: [B, 16, 64, 552] → [B, 32, 32, 276]
        self.enc1 = nn.Sequential(
            ResBlk(16), ResBlk(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1)
        )
        # Stage 2: [B, 32, 32, 276] → [B, 48, 16, 138]
        self.enc2 = nn.Sequential(
            ResBlk(32), ResBlk(32),
            nn.Conv2d(32, 48, 3, stride=2, padding=1)
        )
        # Stage 3: [B, 48, 16, 138] → [B, 64, 8, 69]
        self.enc3 = nn.Sequential(
            ResBlk(48), ResBlk(48),
            nn.Conv2d(48, 64, 3, stride=2, padding=1)
        )
        # Stage 4: [B, 64, 8, 69] → [B, 64, 4, 35]
        self.enc4 = nn.Sequential(
            ResBlk(64), ResBlk(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1)
        )
        self.enc_attn = SelfAttn(64)

        # Bottleneck
        self.flat_dim = 64 * 4 * 35  # 8960
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        # ── Class Embedding ───────────────────────────────
        self.class_embed = nn.Embedding(num_classes, class_emb_dim)

        # ── Decoder ──────────────────────────────────────
        self.fc_decode = nn.Linear(latent_dim + class_emb_dim, self.flat_dim)
        
        # Stage 1: [B, 64, 4, 35] → [B, 64, 8, 70]
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            ResBlk(64), ResBlk(64), SelfAttn(64),
        )
        # Stage 2: [B, 64, 8, 70] → [B, 48, 16, 140]
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 48, 4, stride=2, padding=1),
            ResBlk(48), ResBlk(48),
        )
        # Stage 3: [B, 48, 16, 140] → [B, 32, 32, 280]
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(48, 32, 4, stride=2, padding=1),
            ResBlk(32), ResBlk(32),
        )
        # Stage 4: [B, 32, 32, 280] → [B, 16, 64, 560]
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            ResBlk(16), ResBlk(16),
        )
        self.output_conv = nn.Conv2d(16, 1, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if m in (self.fc_mu, self.fc_logvar):
                    nn.init.normal_(m.weight, std=0.001)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def encode(self, x):
        h = self.enc_conv_in(x)
        h = self.enc1(h)
        h = self.enc2(h)
        h = self.enc3(h)
        h = self.enc4(h)
        h = self.enc_attn(h)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, -10, 10)
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z, labels):
        class_emb = self.class_embed(labels)
        h = torch.cat([z, class_emb], dim=-1)
        h = self.fc_decode(h)
        h = h.view(-1, 64, 4, 35)
        h = self.dec1(h)
        h = self.dec2(h)
        h = self.dec3(h)
        h = self.dec4(h)
        h = self.output_conv(h)  # [B, 1, 64, 560]
        return F.interpolate(h, size=(64, 552), mode='bilinear')

    def forward(self, x, labels):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, labels)
        return recon, mu, logvar

    @torch.no_grad()
    def encode_to_latent(self, x):
        mu, _ = self.encode(x)
        return mu


if __name__ == "__main__":
    m = V14VAE()
    print(f"V14VAE: {sum(p.numel() for p in m.parameters()):,} params")
    x = torch.randn(2, 1, 64, 552)
    r, mu, lv = m(x, torch.tensor([0,1]))
    print(f"In:{x.shape} → z:{mu.shape} → Out:{r.shape}")
    # Generation test
    z = torch.randn(1, 256)
    g = m.decode(z, torch.tensor([0]))
    print(f"Gen: z({z.shape}) → mel({g.shape})")
