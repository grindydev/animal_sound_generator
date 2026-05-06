"""
inference.py — HiFi-GAN inference: convert mel spectrogram → audio waveform.

Single-call API for use in server.py and diagnose scripts.

Usage:
    from src.hifigan.inference import mel_to_waveform

    waveform = mel_to_waveform(mel_spec, device='mps')
    torchaudio.save('output.wav', waveform, 22050)

Where mel_spec is a normalized dB mel spectrogram (same format as VAE output):
    [1, 64, T] or [1, 1, 64, T]
"""
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.hifigan.config import config as cfg
from src.hifigan.generator import HiFiGANGenerator


# ══════════════════════════════════════════════════════════════
#  Model loading (cached in memory after first call)
# ══════════════════════════════════════════════════════════════

_generator: HiFiGANGenerator = None
_loaded_device: torch.device = None


def get_generator(device: torch.device = None) -> HiFiGANGenerator:
    """
    Load HiFi-GAN generator (cached per device).
    """
    global _generator, _loaded_device

    if device is None:
        device = torch.device("cpu")  # CPU is fast enough for inference

    if _generator is not None and _loaded_device == device:
        return _generator

    _generator = HiFiGANGenerator(cfg).to(device)

    # Try final model first, then latest checkpoint
    model_dir = cfg.model_dir
    checkpoint_dir = os.path.join(cfg.checkpoint_dir, "train")  # train mode checkpoints

    paths_to_try = [
        os.path.join(model_dir, "hifigan_generator_train.pth"),
        os.path.join(model_dir, "hifigan_generator.pth"),  # legacy
    ]

    # Check for latest checkpoint
    if os.path.isdir(checkpoint_dir):
        from src.hifigan.utils import scan_checkpoints

        latest = scan_checkpoints(checkpoint_dir, prefix="generator_")
        if latest > 0:
            paths_to_try.append(
                os.path.join(checkpoint_dir, f"generator_{latest:06d}.pth")
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
            print(f"HiFi-GAN loaded from: {path}")
            break

    if not loaded:
        print(
            "⚠️  No HiFi-GAN checkpoint found. Using untrained generator "
            "(output will be random noise)."
        )

    _generator.eval()
    _loaded_device = device
    return _generator


# ══════════════════════════════════════════════════════════════
#  Main API
# ══════════════════════════════════════════════════════════════

def mel_to_waveform(
    mel: torch.Tensor,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Convert a normalized mel spectrogram to audio waveform using HiFi-GAN.

    Args:
        mel: [1, 64, T] or [1, 1, 64, T] or [B, 64, T]
             Normalized dB mel spectrogram (matching VAE output format).

    Returns:
        waveform: [1, num_samples] mono audio, ready for torchaudio.save()
    """
    if device is None:
        device = torch.device("cpu")  # CPU inference — fast enough for 5s audio

    generator = get_generator(device)

    # Handle input shapes
    was_batch = mel.dim() == 4  # [B, 1, 64, T]
    if was_batch:
        mel = mel[:, 0, :, :]  # → [B, 64, T]

    if mel.dim() == 2:
        mel = mel.unsqueeze(0)  # → [1, 64, T]

    mel = mel.to(_loaded_device)

    with torch.no_grad():
        waveform = generator(mel, target_length=mel.shape[-1] * cfg.hop_length)  # [B, 1, T*hop_length]

    # Squeeze to [num_samples] mono
    waveform = waveform.squeeze(0).squeeze(0)

    # Ensure 2D for torchaudio: [1, num_samples]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    return waveform.cpu()


# ══════════════════════════════════════════════════════════════
#  Quick test
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("HiFi-GAN inference test:")
    dummy_mel = torch.randn(1, 64, 160)  # ~1.49s of audio
    wav = mel_to_waveform(dummy_mel)
    print(f"  Input mel:  {dummy_mel.shape}")
    print(f"  Output wav: {wav.shape}")
    print(f"  Duration:   {wav.shape[-1] / cfg.sample_rate:.2f}s")

    # Test with full 5-second mel (matching VAE output)
    full_mel = torch.randn(1, 64, 552)  # 5s @ 22050 with hop=200
    wav_full = mel_to_waveform(full_mel)
    print(f"\n  5s test:")
    print(f"  Input mel:  {full_mel.shape}")
    print(f"  Output wav: {wav_full.shape}")
    print(f"  Duration:   {wav_full.shape[-1] / cfg.sample_rate:.2f}s")
