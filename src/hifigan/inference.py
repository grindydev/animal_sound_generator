"""
inference.py — HiFi-GAN inference: convert mel spectrogram → audio waveform.

Supports:
  - Chunked processing for arbitrary-length mel spectrograms
  - Griffin-Lim phase refinement (optional, for mel-only trained models)
  - Cached model loading

Usage:
    from src.hifigan.inference import mel_to_waveform

    waveform = mel_to_waveform(mel_spec, device='cuda')
    torchaudio.save('output.wav', waveform, 22050)
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.hifigan.config import config as cfg
from src.hifigan.generator import HiFiGANGenerator


# ══════════════════════════════════════════════════════════════
#  Model loading (cached in memory after first call)
# ══════════════════════════════════════════════════════════════

_generator: HiFiGANGenerator = None
_loaded_device: torch.device = None


def get_generator(device: torch.device = None) -> HiFiGANGenerator:
    """Load HiFi-GAN generator (cached per device)."""
    global _generator, _loaded_device

    if device is None:
        device = torch.device("cpu")

    if _generator is not None and _loaded_device == device:
        return _generator

    _generator = HiFiGANGenerator(cfg).to(device)

    model_dir = cfg.model_dir
    checkpoint_dir = os.path.join(cfg.checkpoint_dir, "train")
    meltrain_dir = os.path.join(cfg.checkpoint_dir, "meltrain")

    paths_to_try = [
        os.path.join(model_dir, "hifigan_generator_train_best.pth"),
        os.path.join(model_dir, "hifigan_generator_train.pth"),
        os.path.join(model_dir, "hifigan_generator_meltrain_best.pth"),
        os.path.join(model_dir, "hifigan_generator_meltrain.pth"),
        os.path.join(model_dir, "hifigan_generator.pth"),
    ]

    # Check for latest checkpoints
    for ckpt_dir in [checkpoint_dir, meltrain_dir]:
        if os.path.isdir(ckpt_dir):
            from src.hifigan.utils import scan_checkpoints
            latest = scan_checkpoints(ckpt_dir, prefix="generator_")
            if latest > 0:
                paths_to_try.append(
                    os.path.join(ckpt_dir, f"generator_{latest:06d}.pth")
                )

    loaded = False
    for path in paths_to_try:
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device, weights_only=True)
            if "generator" in ckpt:
                _generator.load_state_dict(ckpt["generator"])
            else:
                _generator.load_state_dict(ckpt)
            loaded = True
            print(f"✅ HiFi-GAN loaded from: {path}")
            break

    if not loaded:
        print("⚠️  No HiFi-GAN checkpoint found. Output will be noise.")

    _generator.eval()
    _loaded_device = device
    return _generator


# ══════════════════════════════════════════════════════════════
#  Chunked inference
# ══════════════════════════════════════════════════════════════

class HiFiGANInference:
    """
    Chunked HiFi-GAN inference with overlap-add for arbitrary-length mel.

    The generator was trained on fixed-length segments (e.g. 41 frames).
    For longer mels, we chunk into overlapping segments, generate each,
    and crossfade them together.
    """

    def __init__(self, generator: HiFiGANGenerator, device: torch.device):
        self.generator = generator
        self.device = device
        # compute_mel on [1, segment_size] → segment_size//hop_length + 1 frames
        self.segment_frames = cfg.segment_size // cfg.hop_length + 1   # 41 for 8192
        self.hop_frames = self.segment_frames // 2                     # 50% overlap
        self.segment_samples = self.segment_frames * cfg.hop_length    # 8200 for 8192
        # Crossfade window (raised cosine)
        self._fade_window = None

    def _get_fade_window(self):
        if self._fade_window is None:
            t = torch.linspace(0, 1, self.segment_samples)
            self._fade_window = (1 - torch.cos(t * torch.pi)) / 2
        return self._fade_window.to(self.device)

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Convert mel to audio. Handles any length via chunking.

        Args:
            mel: [1, n_mels, total_frames]

        Returns:
            waveform: [1, total_samples]
        """
        n_frames = mel.shape[-1]
        total_samples = n_frames * cfg.hop_length

        # Short enough for single pass
        if n_frames <= self.segment_frames + 4:
            with torch.no_grad():
                return self.generator(
                    mel.to(self.device),
                    target_length=total_samples,
                ).cpu()

        # Chunked inference with overlap-add
        output = torch.zeros(1, total_samples)
        weight = torch.zeros(1, total_samples)
        fade = self._get_fade_window()

        for start in range(0, n_frames - self.segment_frames + 1, self.hop_frames):
            end = start + self.segment_frames
            chunk_mel = mel[:, :, start:end]
            chunk_samples = self.segment_frames * cfg.hop_length
            sample_start = start * cfg.hop_length
            sample_end = sample_start + chunk_samples

            with torch.no_grad():
                chunk_audio = self.generator(
                    chunk_mel.to(self.device),
                    target_length=chunk_samples,
                ).cpu()

            # Apply crossfade
            chunk_audio = chunk_audio * fade

            # Add to output
            actual_end = min(sample_end, total_samples)
            actual_len = actual_end - sample_start
            output[0, sample_start:actual_end] += chunk_audio[0, 0, :actual_len]
            weight[0, sample_start:actual_end] += fade[:actual_len]

        # Normalize by overlap weight
        mask = weight > 1e-6
        output[mask] /= weight[mask]

        return output


# ══════════════════════════════════════════════════════════════
#  Griffin-Lim phase refinement
# ══════════════════════════════════════════════════════════════

def griffin_lim_refine(
    waveform: torch.Tensor,
    mel_target: torch.Tensor,
    n_iter: int = 5,
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 200,
    n_mels: int = 64,
) -> torch.Tensor:
    """
    Light Griffin-Lim to refine phase of HiFi-GAN output.

    Takes the HiFi-GAN waveform as initial estimate and the VAE's mel
    as the target magnitude, then runs a few Griffin-Lim iterations
    to align phase with the correct spectrum.

    Args:
        waveform:   [1, num_samples] — HiFi-GAN output
        mel_target: [1, n_mels, T] — VAE mel (normalized dB)
        n_iter:     Griffin-Lim iterations (5 is fast, 20 is thorough)

    Returns:
        refined waveform [1, num_samples]
    """
    import torchaudio.transforms as T

    device = waveform.device

    # Unnormalize mel target
    mel_db = mel_target.to(device) * cfg.norm_std + cfg.norm_mean
    mel_db = torch.clamp(mel_db, min=-80.0, max=0.0)

    # dB → power
    mel_power = F.db_to_amplitude(mel_db, ref=1.0, power=2.0)

    # Mel → linear via pseudo-inverse of mel filterbank
    n_stft = n_fft // 2 + 1
    import torchaudio.functional as AF
    fb = AF.melscale_fbanks(n_stft, 0.0, sample_rate / 2.0, n_mels, sample_rate)
    fb_t = fb.T.to(device)  # [n_mels, n_stft]
    valid = fb_t.sum(dim=1) > 0
    fb_valid = fb_t[valid]
    mel_valid = mel_power[0][valid]
    result = torch.linalg.lstsq(fb_valid, mel_valid, driver='gelsd')
    linear_power = torch.clamp(result.solution, min=0.0)
    linear_mag = torch.sqrt(linear_power)  # [n_stft, T]

    # Griffin-Lim with HiFi-GAN audio as initial estimate
    griffin_lim = T.GriffinLim(
        n_fft=n_fft, hop_length=hop_length, power=1,
        n_iter=n_iter, momentum=0.99, length=waveform.shape[-1],
    ).to(device)

    refined = griffin_lim(linear_mag.T.unsqueeze(0))  # [1, num_samples]

    # Blend with original to preserve naturalness
    refined = 0.7 * refined + 0.3 * waveform

    return refined.cpu()


# ══════════════════════════════════════════════════════════════
#  Main API: mel → waveform (with optional Griffin-Lim polish)
# ══════════════════════════════════════════════════════════════

def mel_to_waveform(
    mel: torch.Tensor,
    device: torch.device = None,
    use_griffin_lim: bool = True,
    griffin_lim_iters: int = 5,
) -> torch.Tensor:
    """
    Convert a normalized mel spectrogram to audio waveform.

    Pipeline:
      1. HiFi-GAN (chunked) → raw audio (correct spectrum)
      2. Griffin-Lim (optional) → phase refinement

    Args:
        mel:               [1, 64, T] or [1, 1, 64, T]
                           Normalized dB mel (same format as VAE output).
        device:            Device for inference (CPU is fine for 5s audio).
        use_griffin_lim:   Apply light Griffin-Lim for phase cleanup.
        griffin_lim_iters: Number of GL iterations (5 recommended).

    Returns:
        waveform: [1, num_samples] mono audio in [-1, 1]
    """
    if device is None:
        device = torch.device("cpu")

    generator = get_generator(device)
    infer_engine = HiFiGANInference(generator, device)

    # Handle input shapes
    if mel.dim() == 4:
        mel = mel[:, 0, :, :]  # [B, 1, 64, T] → [B, 64, T]
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)  # [64, T] → [1, 64, T]

    # Step 1: HiFi-GAN
    waveform = infer_engine(mel)  # [1, total_samples]

    # Step 2: Griffin-Lim phase refinement
    if use_griffin_lim:
        waveform = griffin_lim_refine(
            waveform, mel,
            n_iter=griffin_lim_iters,
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
        )

    return waveform


# ══════════════════════════════════════════════════════════════
#  Quick test
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("HiFi-GAN inference test:")
    dummy_mel = torch.randn(1, 64, 80)  # ~0.74s
    wav = mel_to_waveform(dummy_mel, use_griffin_lim=False)
    print(f"  Input mel:  {dummy_mel.shape}")
    print(f"  Output wav: {wav.shape}")
    print(f"  Duration:   {wav.shape[-1] / cfg.sample_rate:.2f}s")

    # Full 5-second test
    full_mel = torch.randn(1, 64, 552)
    wav_full = mel_to_waveform(full_mel, use_griffin_lim=False)
    print(f"\n  5s test:")
    print(f"  Input mel:  {full_mel.shape}")
    print(f"  Output wav: {wav_full.shape}")
    print(f"  Duration:   {wav_full.shape[-1] / cfg.sample_rate:.2f}s")
