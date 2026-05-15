"""
diffusion.py — DDPM/DDIM Forward & Reverse Processes (Phase 7b).

Implements:
  - Cosine noise schedule (Improved DDPM)
  - Forward diffusion: q(x_t | x_0) — add noise to clean data
  - Reverse diffusion: p(x_{t-1} | x_t) — denoise step by step
  - DDIM accelerated sampling (fewer steps at inference)
  - Img2img refinement: add noise to VAE output, then denoise

References:
  - Ho et al. "Denoising Diffusion Probabilistic Models" (DDPM)
  - Song et al. "Denoising Diffusion Implicit Models" (DDIM)
  - Nichol & Dhariwal "Improved Denoising Diffusion Probabilistic Models" (cosine schedule)
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.diffusion.config import config as cfg


class DiffusionProcess(nn.Module):
    """
    Manages the noise schedule and forward/reverse diffusion steps.
    """

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = cfg
        self.config = config
        self.timesteps = config.timesteps
        self.num_classes = config.num_classes

        # Build noise schedule
        if getattr(config, 'use_linear_schedule', False):
            betas = torch.linspace(config.beta_start, config.beta_end, config.timesteps)
        else:
            betas = self._cosine_beta_schedule(config.timesteps, getattr(config, 'cosine_s', 0.008))
        self.register_schedule_buffers(betas)

    def register_schedule_buffers(self, betas: torch.Tensor):
        """Compute and register all schedule-derived tensors as persistent buffers."""
        self.register_buffer('betas', betas)                         # [T]
        alphas = 1.0 - betas                                         # [T]
        alphas_cumprod = torch.cumprod(alphas, dim=0)                # [T]
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('posterior_variance', betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine noise schedule from "Improved DDPM".
        Produces a smoother noise decay than the linear schedule,
        especially important for small T.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, min=0.0001, max=0.9999)

    # ═════════════════════════════════════════════════════════
    #  Forward Process: q(x_t | x_0)
    # ═════════════════════════════════════════════════════════

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Forward diffusion: sample x_t from x_0 and timestep t.
        
        x_t = sqrt(α_cumprod[t]) * x_0 + sqrt(1 - α_cumprod[t]) * noise

        Args:
            x_0:   clean spectrogram [B, C, H, W]
            t:     timesteps [B] (0 to T-1)
            noise: optional pre-sampled noise [B, C, H, W]
        Returns:
            noisy x_t [B, C, H, W]
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self._get(self.sqrt_alphas_cumprod, t, x_0)
        sqrt_one_minus = self._get(self.sqrt_one_minus_alphas_cumprod, t, x_0)

        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    # ═════════════════════════════════════════════════════════
    #  Reverse Process: p(x_{t-1} | x_t) — DDPM sampling
    # ═════════════════════════════════════════════════════════

    def p_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, labels: torch.Tensor,
                 null_labels=None, cfg_scale=1.0) -> torch.Tensor:
        """
        Single DDPM denoising step: x_t → x_{t-1}.
        Supports classifier-free guidance.
        """
        # Predict noise (with optional CFG)
        pred_noise = model(x_t, t, labels)
        if null_labels is not None and cfg_scale != 1.0:
            pred_uncond = model(x_t, t, null_labels)
            pred_noise = pred_uncond + cfg_scale * (pred_noise - pred_uncond)

        # Simplified posterior mean: (x_t - beta_t/sqrt(1-alpha_cumprod) * pred_noise) / sqrt(alpha_t)
        beta_t = self._get(self.betas, t, x_t)
        alpha_t = 1.0 - beta_t
        sqrt_one_minus_cumprod = self._get(self.sqrt_one_minus_alphas_cumprod, t, x_t)
        posterior_mean = (x_t - beta_t / sqrt_one_minus_cumprod * pred_noise) / torch.sqrt(alpha_t)

        # Add noise (posterior_variance[0] == 0, so t=0 adds nothing automatically)
        noise = torch.randn_like(x_t)
        posterior_var = self._get(self.posterior_variance, t, x_t)
        return posterior_mean + torch.sqrt(posterior_var) * noise

    # ═════════════════════════════════════════════════════════
    #  DDIM Sampling (accelerated, deterministic)
    # ═════════════════════════════════════════════════════════

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        x_t: torch.Tensor,
        labels: torch.Tensor,
        num_steps: int = 50,
        eta: float = 0.0,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """DDIM sampling with optional classifier-free guidance."""
        device = x_t.device
        b = x_t.shape[0]
        null_labels = torch.full((b,), self.num_classes, device=device, dtype=torch.long) if cfg_scale != 1.0 else None
        times = torch.linspace(self.timesteps - 1, 0, num_steps, device=device).long()

        for i in range(len(times) - 1):
            t = times[i]
            t_next = times[i + 1]
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)

            # Predict noise (with optional CFG)
            pred_noise = model(x_t, t_batch, labels)
            if null_labels is not None:
                pred_uncond = model(x_t, t_batch, null_labels)
                pred_noise = pred_uncond + cfg_scale * (pred_noise - pred_uncond)

            alpha_cumprod_t = self.alphas_cumprod[t]
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_cumprod_t)
            x_0_pred = (x_t - sqrt_one_minus_alpha_t * pred_noise) / torch.sqrt(alpha_cumprod_t)
            x_0_pred = torch.clamp(x_0_pred, -4.0, 4.0)

            alpha_cumprod_next = self.alphas_cumprod[t_next]
            sigma = eta * torch.sqrt((1.0 - alpha_cumprod_next) / (1.0 - alpha_cumprod_t) * (1.0 - alpha_cumprod_t / alpha_cumprod_next))
            direction = torch.sqrt(1.0 - alpha_cumprod_next - sigma ** 2) * pred_noise
            noise = torch.randn_like(x_t) if eta > 0 else 0.0
            x_t = torch.sqrt(alpha_cumprod_next) * x_0_pred + direction + sigma * noise

        return x_t

    # ═════════════════════════════════════════════════════════
    #  V7: x₀-Prediction DDIM Sampling
    # ═════════════════════════════════════════════════════════

    @torch.no_grad()
    def ddim_sample_x0(
        self, model, x_t: torch.Tensor, labels: torch.Tensor,
        num_steps: int = 100, eta: float = 0.0, cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """DDIM sampling for x₀-prediction model."""
        device = x_t.device
        b = x_t.shape[0]
        null_labels = torch.full((b,), self.num_classes, device=device, dtype=torch.long) if cfg_scale != 1.0 else None
        times = torch.linspace(self.timesteps - 1, 0, num_steps, device=device).long()

        for i in range(len(times) - 1):
            t = times[i]
            t_next = times[i + 1]
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)

            # Model predicts x₀ directly
            pred_x0 = model(x_t, t_batch, labels)
            if null_labels is not None:
                pred_uncond = model(x_t, t_batch, null_labels)
                pred_x0 = pred_uncond + cfg_scale * (pred_x0 - pred_uncond)

            pred_x0 = torch.clamp(pred_x0, -4.0, 4.0)

            # Recover noise from x₀ prediction
            alpha_t = self.alphas_cumprod[t]
            noise_pred = (x_t - torch.sqrt(alpha_t) * pred_x0) / torch.sqrt(1.0 - alpha_t)

            # DDIM step
            alpha_next = self.alphas_cumprod[t_next]
            sigma = eta * torch.sqrt((1.0 - alpha_next) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_next))
            direction = torch.sqrt(1.0 - alpha_next - sigma ** 2) * noise_pred
            noise = torch.randn_like(x_t) if eta > 0 else 0.0
            x_t = torch.sqrt(alpha_next) * pred_x0 + direction + sigma * noise

        return x_t

    @torch.no_grad()
    def p_sample_loop_x0(self, model, shape, labels: torch.Tensor, device,
                         progress: bool = False, cfg_scale: float = 1.0) -> torch.Tensor:
        """Full DDPM reverse process for x₀-prediction model."""
        # DDPM for x₀: use p_sample but instead of predicting noise,
        # predict x₀ and recover noise for the step
        # Simpler: just use ddim_sample_x0 with eta=1.0 (fully stochastic)
        return self.ddim_sample_x0(model, torch.randn(shape, device=device),
                                   labels, num_steps=self.timesteps,
                                   eta=1.0, cfg_scale=cfg_scale)
    # ═════════════════════════════════════════════════════════

    @torch.no_grad()
    def p_sample_loop(self, model, shape, labels: torch.Tensor, device,
                      progress: bool = False, cfg_scale: float = 1.0) -> torch.Tensor:
        """Full DDPM reverse process with optional CFG."""
        b = shape[0]
        x_t = torch.randn(shape, device=device)
        null_labels = torch.full((b,), self.num_classes, device=device, dtype=torch.long) if cfg_scale != 1.0 else None

        timesteps = list(reversed(range(self.timesteps)))
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="  Denoising")

        for t_step in timesteps:
            t = torch.full((b,), t_step, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t, labels, null_labels, cfg_scale)

        return x_t

    # ═════════════════════════════════════════════════════════
    #  Img2img Refinement (VAE output → sharpen)
    # ═════════════════════════════════════════════════════════

    @torch.no_grad()
    def refine(
        self,
        model,
        vae_output: torch.Tensor,
        labels: torch.Tensor,
        num_steps: int = 50,
        strength: float = 0.6,
        use_ddim: bool = True,
    ) -> torch.Tensor:
        """
        Refine a VAE-generated blurry spectrogram via img2img.

        1. Add noise to vae_output (strength controls how much)
        2. Denoise with DDIM/DDPM to get sharp output

        Args:
            model:       U-Net noise predictor
            vae_output:  blurry VAE spectrogram [B, C, H, W]
            labels:      class labels [B]
            num_steps:   denoising steps
            strength:    how much noise to add (0.0 = no change, 1.0 = full noise)
            use_ddim:    use DDIM if True, DDPM if False
        Returns:
            sharpened spectrogram [B, C, H, W]
        """
        device = vae_output.device
        b = vae_output.shape[0]

        # Starting timestep from strength
        start_t = int(strength * (self.timesteps - 1))
        start_t = max(start_t, 1)  # at least 1 step

        # Add noise to VAE output
        noise = torch.randn_like(vae_output)
        t_start = torch.full((b,), start_t, device=device, dtype=torch.long)
        x_t = self.q_sample(vae_output, t_start, noise)

        if use_ddim:
            # DDIM: sample only from start_t down
            # Build timesteps from start_t down to 0
            times = torch.linspace(start_t, 0, min(num_steps, start_t + 1), device=device).long()
            for i in range(len(times) - 1):
                t = times[i]
                t_next = times[i + 1]
                t_batch = torch.full((b,), t, device=device, dtype=torch.long)

                pred_noise = model(x_t, t_batch, labels)

                alpha_t = self.alphas_cumprod[t]
                alpha_next = self.alphas_cumprod[t_next]

                x_0_pred = (x_t - torch.sqrt(1.0 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
                x_0_pred = torch.clamp(x_0_pred, -4.0, 4.0)

                direction = torch.sqrt(1.0 - alpha_next) * pred_noise
                x_t = torch.sqrt(alpha_next) * x_0_pred + direction

            return x_t
        else:
            # DDPM: step from start_t down to 0
            for t_step in reversed(range(start_t)):
                t = torch.full((b,), t_step, device=device, dtype=torch.long)
                x_t = self.p_sample(model, x_t, t, labels)
            return x_t

    # ═════════════════════════════════════════════════════════
    #  Utility
    # ═════════════════════════════════════════════════════════

    def _get(self, buffer: torch.Tensor, t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Index into a schedule buffer and broadcast to match `ref` shape.
        Buffer has shape [T], t has shape [B] → output [B, 1, 1, 1]
        """
        out = buffer[t].float()
        while out.dim() < ref.dim():
            out = out.unsqueeze(-1)
        return out


# ═══════════════════════════════════════════════════════════════
#  Quick test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🔧 DiffusionProcess — schedule test\n")

    diff = DiffusionProcess()
    print(f"   Timesteps: {diff.timesteps}")
    print(f"   Beta range: [{diff.betas[0]:.6f}, {diff.betas[-1]:.6f}]")
    print(f"   Alpha_cumprod[0]: {diff.alphas_cumprod[0]:.4f}")
    print(f"   Alpha_cumprod[-1]: {diff.alphas_cumprod[-1]:.6f}")

    # Test q_sample
    x_0 = torch.randn(2, 1, 64, 552)
    t = torch.tensor([100, 500])
    x_t = diff.q_sample(x_0, t)
    print(f"\n   q_sample: x_0 var={x_0.var():.4f} → x_t var={x_t.var():.4f}")

    # Signal-to-noise ratio at t=500
    t_mid = torch.tensor([500])
    alpha = diff.alphas_cumprod[500]
    snr = alpha / (1.0 - alpha)
    print(f"   SNR at t=500: {snr:.2f} ({10*torch.log10(snr):.1f} dB)")
