"""
GAN Generator v17 — Per-bin z-score output + simple FiLM

Input:  noise z [B, 256] + class label [B]
Output: mel spectrogram [B, 1, 64, 552] (per-bin z-score normalized, no activation)
"""
import torch
import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, cond_dim, num_features):
        super().__init__()
        self.linear = nn.Linear(cond_dim, num_features * 2)

    def forward(self, h, cond):
        γ, β = self.linear(cond).chunk(2, dim=1)
        return h * (1.0 + γ.unsqueeze(-1).unsqueeze(-1)) + β.unsqueeze(-1).unsqueeze(-1)


class FiLMUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.film = FiLM(cond_dim, out_ch)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, cond):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.film(x, cond)
        return self.act(x)


class Generator(nn.Module):
    """Simple class-conditional mel generator.

    noise [256] + class → MLP → cond [256]
    → Dense → [128, 4, 36] → FiLMUpBlocks → [1, 64, 576] → crop → [1, 64, 552]
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mel_width = config.mel_width
        self.num_classes = config.num_classes

        self.class_embed = nn.Embedding(config.num_classes, config.latent_dim)

        # Conditioning MLP: [noise + class_emb] → cond
        self.cond_mlp = nn.Sequential(
            nn.Linear(config.latent_dim * 2, config.latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(config.latent_dim, config.latent_dim),
        )

        # Initial dense
        self.init_h, self.init_w = 4, 36
        base_ch = config.gen_base_ch
        self.init_dense = nn.Linear(config.latent_dim, base_ch * self.init_h * self.init_w)

        # FiLM upsampling blocks
        in_ch = base_ch
        out_channels = [base_ch // 2, base_ch // 4, base_ch // 8, base_ch // 16]
        self.blocks = nn.ModuleList()
        for out_ch in out_channels:
            self.blocks.append(FiLMUpBlock(in_ch, out_ch, config.latent_dim))
            in_ch = out_ch

        # Final conv — NO activation (per-bin z-score output)
        self.final_conv = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, z, labels):
        B = z.shape[0]
        class_emb = self.class_embed(labels)
        cond = self.cond_mlp(torch.cat([z, class_emb], dim=1))

        x = self.init_dense(cond)
        x = x.view(B, self.config.gen_base_ch, self.init_h, self.init_w)

        for block in self.blocks:
            x = block(x, cond)

        x = self.final_conv(x)
        if x.shape[-1] > self.mel_width:
            x = x[..., :self.mel_width]
        return x  # z-score output, no tanh
