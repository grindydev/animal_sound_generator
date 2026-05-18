"""
GAN Generator — FiLM-conditioned synthesis from noise

Input:  noise z [B, latent_dim] + class label [B]
Output: mel spectrogram [B, 1, 64, 552]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: h = h * (1 + γ) + β"""
    def __init__(self, cond_dim, num_features):
        super().__init__()
        self.linear = nn.Linear(cond_dim, num_features * 2)

    def forward(self, h, cond):
        γ, β = self.linear(cond).chunk(2, dim=1)
        return h * (1.0 + γ.unsqueeze(-1).unsqueeze(-1)) + β.unsqueeze(-1).unsqueeze(-1)


class FiLMUpBlock(nn.Module):
    """Upsample → Conv → FiLM → LeakyReLU"""
    def __init__(self, in_ch, out_ch, cond_dim, upsample=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else nn.Identity()
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )
        self.film = FiLM(cond_dim, out_ch)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, cond):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.film(x, cond)
        x = self.act(x)
        return x


class Generator(nn.Module):
    """Class-conditional mel spectrogram generator.

    Architecture:
        noise [B, latent_dim] + class label [B]
          → class embedding + concat → MLP → conditioning [cond_dim]
          → Dense → [base_ch, 4, 18]
          → FiLMUpBlock × 4 → [1, 64, 576]
          → crop → [1, 64, 552] (output)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.cond_dim = config.cond_dim
        self.num_classes = config.num_classes
        self.mel_pad_width = config.mel_pad_width
        self.mel_width = config.mel_width

        # Class embedding
        self.class_embed = nn.Embedding(config.num_classes, config.latent_dim)

        # Conditioning MLP: [noise + class_embed] → [cond_dim]
        self.cond_mlp = nn.Sequential(
            nn.Linear(config.latent_dim * 2, config.cond_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(config.cond_dim, config.cond_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(config.cond_dim, config.cond_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(config.cond_dim, config.cond_dim),
        )

        # Initial dense: cond → spatial features
        # Starting spatial: [4, 18] (18 = 576 / 2^5... wait, lets trace)
        # Upsamples: [4,18] →×2→ [8,36] →×2→ [16,72] →×2→ [32,144] →×2→ [64,288]
        # We need one more: →×2→ [128,576] ← too tall. Need final conv to fix height.
        #
        # Better: start at [4, 36]
        # →×2→ [8,72] →×2→ [16,144] →×2→ [32,288] →×2→ [64,576] ✓
        self.init_h, self.init_w = 4, 36
        base_ch = config.gen_base_ch
        self.init_dense = nn.Linear(config.cond_dim, base_ch * self.init_h * self.init_w)

        # Channel progression: base_ch → base_ch//2 → base_ch//4 → base_ch//8 → 1
        ch = base_ch
        self.blocks = nn.ModuleList()
        out_channels = [
            min(ch // 2, config.gen_max_ch),  # 128
            min(ch // 4, config.gen_max_ch),  # 64
            min(ch // 8, config.gen_max_ch),  # 32
            min(ch // 16, config.gen_max_ch), # 16
        ]
        in_ch = ch
        for out_ch in out_channels:
            self.blocks.append(FiLMUpBlock(in_ch, out_ch, config.cond_dim, upsample=True))
            in_ch = out_ch

        # Final conv: in_ch → 1 (no upsample, stays at [64, 576])
        self.final_conv = nn.utils.spectral_norm(
            nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)
        )

    def forward(self, z, labels):
        """
        Args:
            z: noise tensor [B, latent_dim]
            labels: class indices [B]
        Returns:
            mel: [B, 1, 64, mel_width]
        """
        B = z.shape[0]

        # Conditioning
        class_emb = self.class_embed(labels)           # [B, latent_dim]
        cond_input = torch.cat([z, class_emb], dim=1)   # [B, latent_dim*2]
        cond = self.cond_mlp(cond_input)                # [B, cond_dim]

        # Initial spatial features
        x = self.init_dense(cond)                       # [B, base_ch * 4 * 36]
        x = x.view(B, self.config.gen_base_ch, self.init_h, self.init_w)

        # Synthesis blocks
        for block in self.blocks:
            x = block(x, cond)

        # Final conv
        x = self.final_conv(x)  # [B, 1, 64, 576]

        # Crop to target width
        if x.shape[-1] > self.mel_width:
            x = x[..., :self.mel_width]

        return torch.tanh(x)  # output in [-1, 1]
