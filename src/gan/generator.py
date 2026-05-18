"""
GAN Generator v17 — No tanh + hierarchical class conditioning

Input:  noise z [B, latent_dim] + class label [B]
Output: mel spectrogram [B, 1, 64, 552] (per-bin z-score normalized)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.film1 = FiLM(cond_dim, out_ch)
        self.film2 = FiLM(cond_dim, out_ch)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, cond):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.film1(x, cond)
        x = self.act(x)
        x = self.conv2(x)
        x = self.film2(x, cond)
        x = self.act(x)
        return x


class Generator(nn.Module):
    """v17: Per-bin z-score output (no tanh). Hierarchical class conditioning.

    Architecture:
        noise [256] + class → style_mlp → w [256]
        w + class_emb[128] → cond per block [384]
        Dense → [128, 4, 36] → FiLMUpBlocks → [1, 64, 576] → crop → [1, 64, 552]
        No output activation (linear) — output is z-score normalized per bin
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.num_classes = config.num_classes
        self.mel_width = config.mel_width

        # Style mapping network: [noise + class] → w
        self.class_embed = nn.Embedding(config.num_classes, config.latent_dim)
        self.style_mlp = nn.Sequential(
            nn.Linear(config.latent_dim * 2, config.latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(config.latent_dim, config.latent_dim),
        )

        # Per-block conditioning: w + class_emb → block condition
        # Each block gets its own conditioning (hierarchical)
        self.block_cond_mlps = nn.ModuleList()

        # Initial spatial
        self.init_h, self.init_w = 4, 36
        base_ch = config.gen_base_ch
        self.init_dense = nn.Linear(config.latent_dim, base_ch * self.init_h * self.init_w)
        self.block_cond_mlps.append(nn.Linear(config.latent_dim + config.latent_dim, config.latent_dim))

        # Channel progression
        in_ch = base_ch
        out_channels = [min(base_ch // 2, config.gen_max_ch),
                        min(base_ch // 4, config.gen_max_ch),
                        min(base_ch // 8, config.gen_max_ch),
                        min(base_ch // 16, config.gen_max_ch)]
        self.blocks = nn.ModuleList()
        for out_ch in out_channels:
            self.blocks.append(FiLMUpBlock(in_ch, out_ch, config.latent_dim))
            self.block_cond_mlps.append(nn.Linear(config.latent_dim + config.latent_dim, config.latent_dim))
            in_ch = out_ch

        # Final conv: NO activation (linear output)
        self.final_conv = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, z, labels):
        B = z.shape[0]

        # Style vector
        class_emb = self.class_embed(labels)
        w = self.style_mlp(torch.cat([z, class_emb], dim=1))  # [B, latent_dim]

        # Per-block conditioning
        block_input = torch.cat([w, class_emb], dim=1)  # [B, latent_dim*2]
        cond0 = self.block_cond_mlps[0](block_input)

        # Initial spatial
        x = self.init_dense(w)
        x = x.view(B, self.config.gen_base_ch, self.init_h, self.init_w)

        # Synthesis blocks
        for i, block in enumerate(self.blocks):
            cond = self.block_cond_mlps[i + 1](block_input)
            x = block(x, cond)

        # Final conv (no activation)
        x = self.final_conv(x)

        if x.shape[-1] > self.mel_width:
            x = x[..., :self.mel_width]

        return x  # z-score output, no tanh
