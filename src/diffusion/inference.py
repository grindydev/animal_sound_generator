"""
inference.py — V12: Diffusion Generation & Refinement Inference.

Provides the high-level API for generating spectrograms from noise
and refining VAE-generated spectrograms:

  1. generate_from_noise(): pure noise → DDIM 200 steps → sharp mel
  2. match_spectral_stats(): fix mel statistics before HiFi-GAN (V12)
  3. refine_spectrogram(): VAE output → diffusion sharpen → sharp mel
  4. generate_refined(): full pipeline: VAE → Diffusion → HiFi-GAN → audio

Usage:
    from src.diffusion.inference import generate_from_noise, match_spectral_stats

    # Option A: Generate from scratch
    mel = generate_from_noise(label_idx=0, num_steps=200)
    audio = mel_to_waveform(mel)  # then convert to audio

    # Option B: Refine a VAE spectrogram
    sharp_mel = refine_spectrogram(vae_mel, label_idx=0, strength=0.4)
"""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.diffusion.config import config as cfg
from src.diffusion.unet import SpectrogramUNet
from src.diffusion.diffusion import DiffusionProcess


# ═══════════════════════════════════════════════════════════════
#  Model loading (cached in memory after first call)
# ═══════════════════════════════════════════════════════════════

_unet: SpectrogramUNet = None
_diffusion: DiffusionProcess = None
_loaded_device: torch.device = None

# Class name → index mapping (7 animal classes — Noise removed in v12)
CLASSES = ['Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}


def get_diffusion_model(device: torch.device = None):
    """Load diffusion U-Net + diffusion process (cached per device)."""
    global _unet, _diffusion, _loaded_device

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if _unet is not None and _loaded_device == device:
        return _unet, _diffusion

    _unet = SpectrogramUNet(cfg).to(device)
    _diffusion = DiffusionProcess(cfg).to(device)

    # Try loading checkpoint
    paths_to_try = [
        os.path.join(cfg.model_dir, "diffusion_unet_train_best.pth"),
        os.path.join(cfg.model_dir, "diffusion_unet_train.pth"),
        os.path.join(cfg.model_dir, "diffusion_unet.pth"),
    ]

    loaded = False
    for path in paths_to_try:
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device, weights_only=True)
            if "unet" in ckpt:
                _unet.load_state_dict(ckpt["unet"])
            elif "model" in ckpt:
                _unet.load_state_dict(ckpt["model"])
            else:
                _unet.load_state_dict(ckpt)
            loaded = True
            print(f"✅ Diffusion U-Net loaded from: {path}")
            break

    if not loaded:
        print("⚠️  No diffusion checkpoint found. Using untrained model (output = noise).")

    _unet.eval()
    _loaded_device = device
    return _unet, _diffusion


# ═══════════════════════════════════════════════════════════════
#  Pure Generation API (from noise — no VAE needed)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_from_noise(
    label_idx: int = 0,
    num_samples: int = 1,
    num_steps: int = 100,
    device: torch.device = None,
    spec_shape: tuple = (64, 552),
    use_ddpm: bool = False,
    cfg_scale: float = 2.0,
) -> torch.Tensor:
    """
    Generate mel spectrogram from pure Gaussian noise using diffusion.

    Args:
        label_idx:   animal class index (0=Dog, 1=Cat, ...)
        num_samples: how many spectrograms to generate
        num_steps:   DDIM sampling steps (100 recommended)
        device:      torch device
        spec_shape:  (n_mels, time_frames) — default (64, 552)
        use_ddpm:    use full DDPM sampling (slower, more diverse)
        cfg_scale:   classifier-free guidance (>1.0 sharpens output)
    Returns:
        generated mel [num_samples, 1, 64, T]
    """
    model, diffusion = get_diffusion_model(device)
    if device is None:
        device = next(model.parameters()).device

    labels = torch.full((num_samples,), label_idx, device=device, dtype=torch.long)

    if use_ddpm:
        if getattr(cfg, 'predict_x0', False):
            generated = diffusion.p_sample_loop_x0(
                model, (num_samples, cfg.spec_channels, spec_shape[0], spec_shape[1]),
                labels, device, progress=True, cfg_scale=cfg_scale
            )
        else:
            generated = diffusion.p_sample_loop(
                model, (num_samples, cfg.spec_channels, spec_shape[0], spec_shape[1]),
                labels, device, progress=True, cfg_scale=cfg_scale
            )
    else:
        x_t = torch.randn(num_samples, cfg.spec_channels, spec_shape[0], spec_shape[1], device=device)
        if getattr(cfg, 'predict_x0', False):
            generated = diffusion.ddim_sample_x0(
                model, x_t, labels, num_steps=num_steps, eta=0.0, cfg_scale=cfg_scale
            )
        else:
            generated = diffusion.ddim_sample(
                model, x_t, labels, num_steps=num_steps, eta=0.0, cfg_scale=cfg_scale
            )

    return generated.cpu()


# ═══════════════════════════════════════════════════════════════
#  V12: Spectral Stats Matching (prevents crackle/electric sound)
# ═══════════════════════════════════════════════════════════════

def match_spectral_stats(mel: torch.Tensor) -> torch.Tensor:
    """
    Rescale generated mel spectrogram to match real data statistics.

    Problem: Generated mels have wildly different statistics than real mels.
      Real:     mean≈0.0, std≈0.5  (normalized)
      Generated: mean≈-1.5, std≈1.4  (erratic, extreme outliers)

    HiFi-GAN was trained on real data statistics. Feeding it wrong-stats
    mels produces electric/crackle artifacts.

    Args:
        mel: generated mel spectrogram [B, C, F, T]
    Returns:
        rescaled mel with proper statistics, clipped to [-2, 2]
    """
    target_mean = getattr(cfg, 'target_mel_mean', 0.0)
    target_std = getattr(cfg, 'target_mel_std', 0.5)
    clip_range = getattr(cfg, 'mel_clip_range', 2.0)

    # Per-sample normalization
    mel_mean = mel.mean(dim=(1, 2, 3), keepdim=True)
    mel_std = mel.std(dim=(1, 2, 3), keepdim=True)

    # Rescale
    mel = (mel - mel_mean) / (mel_std + 1e-8) * target_std + target_mean

    # Clip extreme outliers
    return torch.clamp(mel, -clip_range, clip_range)


# ═══════════════════════════════════════════════════════════════
#  Refinement API
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def refine_spectrogram(
    vae_output: torch.Tensor,
    label_idx: int = 0,
    strength: float = 0.6,
    num_steps: int = 50,
    device: torch.device = None,
    use_ddim: bool = True,
) -> torch.Tensor:
    """
    Sharpen a VAE-generated blurry spectrogram using diffusion.

    This is the "img2img" refinement step:
      1. Add noise to VAE output (strength controls how much)
      2. Denoise with U-Net → sharp edges, clear harmonics

    Args:
        vae_output: VAE-generated spectrogram [1, 1, 64, T]
        label_idx:  animal class index (0=Dog, 1=Cat, ...)
        strength:   how much refinement (0.0=none, 0.6=balanced, 1.0=full)
        num_steps:  denoising steps (more = sharper but slower)
        device:     torch device
        use_ddim:   use DDIM accelerated sampling
    Returns:
        sharpened spectrogram [1, 1, 64, T]
    """
    model, diffusion = get_diffusion_model(device)

    if device is None:
        device = next(model.parameters()).device

    # Ensure correct shape
    if vae_output.dim() == 3:
        vae_output = vae_output.unsqueeze(1)  # [B, 64, T] → [B, 1, 64, T]

    x = vae_output.to(device)
    labels = torch.full((x.shape[0],), label_idx, device=device, dtype=torch.long)

    refined = diffusion.refine(
        model, x, labels,
        num_steps=num_steps,
        strength=strength,
        use_ddim=use_ddim,
    )

    return refined.cpu()


# ═══════════════════════════════════════════════════════════════
#  Full Pipeline: VAE → Diffusion → HiFi-GAN → Audio
# ═══════════════════════════════════════════════════════════════

def generate_refined(
    vae_model,
    label: str = "Dog",
    num_samples: int = 1,
    temperature: float = 0.7,
    strength: float = 0.6,
    num_diffusion_steps: int = 50,
    device: str = None,
    use_diffusion: bool = True,
    use_griffin_lim: bool = True,
    griffin_lim_iters: int = 5,
):
    """
    Full generation pipeline: VAE → Diffusion → HiFi-GAN → Audio.

    Args:
        vae_model:          trained SimpleAudioVAE instance
        label:              animal class name ("Dog", "Cat", etc.)
        num_samples:        how many sounds to generate
        temperature:        VAE sampling temperature (0.5=consistent, 1.5=wild)
        strength:           diffusion refinement strength
        num_diffusion_steps: DDIM sampling steps
        device:             'cpu', 'cuda', 'mps', or None (auto)
        use_diffusion:      enable/disable diffusion refinement
        use_griffin_lim:    enable Griffin-Lim phase refinement
        griffin_lim_iters:  Griffin-Lim iterations
    Returns:
        waveform: [1, samples] mono audio in [-1, 1]
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    label_idx = CLASS_TO_IDX.get(label, 0)

    # Step 1: VAE generates rough spectrogram
    vae_model.eval()
    vae_device = next(vae_model.parameters()).device
    vae_mel = vae_model.sample(label_idx, num_samples=num_samples,
                                device=vae_device, temperature=temperature)
    # vae_mel: [1, 1, 64, 552]

    # Step 2: Diffusion refines it (optional)
    if use_diffusion:
        vae_mel = refine_spectrogram(
            vae_mel, label_idx=label_idx,
            strength=strength, num_steps=num_diffusion_steps,
            device=device, use_ddim=True,
        )

    # Step 3: HiFi-GAN converts to audio
    from src.hifigan.inference import mel_to_waveform
    audio = mel_to_waveform(
        vae_mel,
        device=device,
        use_griffin_lim=use_griffin_lim,
        griffin_lim_iters=griffin_lim_iters,
    )

    return audio


# ═══════════════════════════════════════════════════════════════
#  Quick test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🔧 Diffusion Inference — test\n")

    # Test refine_spectrogram with a dummy VAE output and untrained model
    dummy_mel = torch.randn(1, 1, 64, 552) * 0.5

    refined = refine_spectrogram(
        dummy_mel, label_idx=0,
        strength=0.6, num_steps=10,
        use_ddim=True,
    )

    print(f"   Input shape:  {dummy_mel.shape}")
    print(f"   Output shape: {refined.shape}")
    print(f"   Match: {dummy_mel.shape == refined.shape} ✅" if dummy_mel.shape == refined.shape else "   ❌")
    print(f"   Input range:  [{dummy_mel.min():.4f}, {dummy_mel.max():.4f}]")
    print(f"   Output range: [{refined.min():.4f}, {refined.max():.4f}]")
