"""
v14_ldm.py — V14 Latent Diffusion Model

Small MLP-based diffusion on 256-dim VAE latent vectors.
Much easier than pixel-level diffusion (256 values vs 35,328).

Architecture:
  - Input: noisy latent z_t [256] + timestep t + class label
  - Output: predicted clean latent z_0 [256]
  - ~300K params — appropriate for 640 training samples
  
  Timestep conditioning: sinusoidal embedding → FiLM in every layer
  Class conditioning: learned embedding → concatenated with input

Training:
  1. VAE encodes real mel → z_clean [256]
  2. Add noise at random timestep t → z_t
  3. Model(z_t, t, class) → predicted z_0
  4. Loss = L1(pred_z0, z_clean) + β_kl * KL(VAE prior)

Inference:
  1. Random noise z_T [256]
  2. DDIM 50 steps with classifier-free guidance
  3. VAE decoder(z_0, class) → mel spectrogram
  4. Griffin-Lim → audio waveform
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal time embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return emb


class FiLMLayer(nn.Module):
    """Linear layer with FiLM conditioning (time embedding modulates)."""
    def __init__(self, in_dim, out_dim, time_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_dim * 2),
        )

    def forward(self, x, t_emb):
        h = self.linear(x)
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=-1)
        return h * (1 + scale) + shift


class FiLMBlock(nn.Module):
    """Residual FiLM block with time and class conditioning."""
    def __init__(self, dim, time_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.film = FiLMLayer(dim, dim, time_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, t_emb):
        h = self.norm(x)
        h = self.film(h, t_emb)
        h = F.silu(h)
        h = self.dropout(h)
        x = x + h

        h = self.norm2(x)
        h = self.ff(h)
        return x + h


class LatentDiffusionModel(nn.Module):
    """
    Small diffusion model for VAE latent space.
    
    Architecture: MLP-ResNet with FiLM time conditioning + class embedding.
    
    Args:
        latent_dim: VAE latent dimension (256)
        class_emb_dim: class embedding dimension (32)
        time_emb_dim: time embedding dimension (64)
        hidden_dim: hidden layer dimension (512)
        num_blocks: number of FiLM blocks (4)
        num_classes: number of animal classes (7)
    """
    def __init__(
        self, latent_dim=256, class_emb_dim=32, time_emb_dim=64,
        hidden_dim=512, num_blocks=4, num_classes=7, dropout=0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.null_label = num_classes  # for CFG

        # Embeddings
        self.time_embed = SinusoidalEmbedding(time_emb_dim)
        self.class_embed = nn.Embedding(num_classes + 1, class_emb_dim)  # +1 for null

        # Input projection
        in_dim = latent_dim + class_emb_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # FiLM blocks
        self.blocks = nn.ModuleList([
            FiLMBlock(hidden_dim, time_emb_dim, dropout)
            for _ in range(num_blocks)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x, t, labels):
        """
        Predict clean latent z_0 from noisy latent z_t.
        
        Args:
            x:      noisy latent [B, latent_dim]
            t:      timesteps [B]
            labels: class labels [B] (use null_label for unconditional)
        Returns:
            predicted clean latent [B, latent_dim]
        """
        # Time embedding
        t_emb = self.time_embed(t)  # [B, time_emb_dim]

        # Class embedding
        labels = labels.clamp(0, self.null_label)
        c_emb = self.class_embed(labels)  # [B, class_emb_dim]

        # Combine
        h = torch.cat([x, c_emb], dim=-1)  # [B, latent_dim + class_emb_dim]
        h = self.input_proj(h)  # [B, hidden_dim]

        # FiLM blocks
        for block in self.blocks:
            h = block(h, t_emb)

        # Output
        return self.output_proj(h)


class LatentDiffusion:
    """
    Diffusion process for VAE latent space.
    
    Cosine noise schedule, DDIM sampling, classifier-free guidance.
    """
    def __init__(self, timesteps=1000, cosine_s=0.008, device='cpu'):
        self.timesteps = timesteps
        self.device = device
        self.build_schedule(timesteps, cosine_s)

    def build_schedule(self, T, s):
        """Cosine schedule (Improved DDPM)."""
        steps = T + 1
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0.0001, 0.9999)

        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def to(self, device):
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: x_0 → x_t."""
        if noise is None:
            noise = torch.randn_like(x_0)
        alpha = self._gather(self.sqrt_alphas_cumprod, t)
        sigma = self._gather(self.sqrt_one_minus_alphas_cumprod, t)
        return alpha * x_0 + sigma * noise

    def _gather(self, arr, t):
        """Index into [T] array with [B] indices, expand dims."""
        out = arr[t]
        while out.dim() < 2:
            out = out.unsqueeze(-1)
        return out

    @torch.no_grad()
    def ddim_sample(self, model, shape, labels, num_steps=50, cfg_scale=2.0, device='cpu'):
        """DDIM sampling with classifier-free guidance."""
        b = shape[0]
        x_t = torch.randn(shape, device=device)
        null_labels = torch.full((b,), model.null_label, device=device, dtype=torch.long)

        times = torch.linspace(self.timesteps - 1, 0, num_steps, device=device).long()

        for i in range(len(times) - 1):
            t = times[i]
            t_next = times[i + 1]
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)

            # Predict x_0 with CFG
            pred_x0 = model(x_t, t_batch, labels)
            if cfg_scale != 1.0:
                pred_uncond = model(x_t, t_batch, null_labels)
                pred_x0 = pred_uncond + cfg_scale * (pred_x0 - pred_uncond)

            # DDIM step
            alpha_t = self.alphas_cumprod[t]
            alpha_next = self.alphas_cumprod[t_next]

            # Recover noise
            noise_pred = (x_t - torch.sqrt(alpha_t) * pred_x0) / torch.sqrt(1.0 - alpha_t)

            # Step to next timestep
            direction = torch.sqrt(1.0 - alpha_next) * noise_pred
            x_t = torch.sqrt(alpha_next) * pred_x0 + direction

        return x_t


if __name__ == "__main__":
    model = LatentDiffusionModel()
    n = sum(p.numel() for p in model.parameters())
    print(f"LatentDiffusionModel: {n:,} params ({n/1e6:.2f}M)")

    x = torch.randn(4, 256)
    t = torch.randint(0, 1000, (4,))
    labels = torch.tensor([0, 1, 2, 3])
    pred = model(x, t, labels)
    print(f"Input: {x.shape} → Output: {pred.shape}")

    diffusion = LatentDiffusion()
    sampled = diffusion.ddim_sample(model, (2, 256), torch.tensor([0, 1]), num_steps=50, device='cpu')
    print(f"DDIM sample: {sampled.shape}")
