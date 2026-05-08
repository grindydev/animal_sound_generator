#!/usr/bin/env python3
"""
listen_hifigan.py — Compare all vocoder approaches on real audio.

Takes real audio, extracts mel, then re-synthesizes with:
  1. Griffin-Lim (baseline — mathematical phase estimation)
  2. HiFi-GAN meltrain-only (no discriminator, learned phase)
  3. HiFi-GAN full GAN (with discriminator, realistic phase)

Output: models/compare/*.wav — listen to hear the difference!

Usage:
    python src/hifigan/listen_hifigan.py
"""
import os
import sys
import torch
import torchaudio
import torchaudio.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.hifigan.config import config as cfg
from src.hifigan.generator import HiFiGANGenerator
from src.hifigan.inference import HiFiGANInference

# ── Config ─────────────────────────────────────────────────────
TEST_FILES = [
    "data/animal_audio/Dog/100124.wav",
    "data/animal_audio/Cat/100114.wav",
    "data/animal_audio/Insect/100886.wav",
]
OUT_DIR = "models/compare"
DURATION_SEC = 2.0

# ── Helpers ────────────────────────────────────────────────────
def load_real_audio(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != cfg.sample_rate:
        wav = torchaudio.transforms.Resample(sr, cfg.sample_rate)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    target = int(cfg.sample_rate * DURATION_SEC)
    if wav.shape[-1] > target:
        wav = wav[:, :target]
    return wav


def audio_to_mel(audio: torch.Tensor) -> torch.Tensor:
    mel_tfm = T.MelSpectrogram(
        sample_rate=cfg.sample_rate, n_fft=cfg.n_fft,
        hop_length=cfg.hop_length, win_length=cfg.win_length,
        n_mels=cfg.n_mels, f_min=cfg.f_min, f_max=cfg.f_max, power=2,
    )
    db_tfm = T.AmplitudeToDB(stype='power', top_db=None)
    spec = mel_tfm(audio.squeeze(1))
    mel_db = db_tfm(spec)
    return (mel_db - cfg.norm_mean) / cfg.norm_std  # normalized


def save_audio(audio: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # torchaudio.save expects [B, T] — squeeze to 2D if needed
    if audio.dim() == 3:
        audio = audio.squeeze(0)  # [1, 1, T] → [1, T] or [B, 1, T] → [B, T]
    if audio.dim() == 3:
        audio = audio.squeeze(1)  # fallback [B, 1, T] → [B, T]
    torchaudio.save(path, audio, cfg.sample_rate)


def load_generator(model_path: str, device: torch.device) -> HiFiGANGenerator:
    gen = HiFiGANGenerator(cfg).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    state = ckpt["generator"] if "generator" in ckpt else ckpt
    gen.load_state_dict(state)
    gen.eval()
    return gen


# ── Griffin-Lim from normalized mel ────────────────────────────
def mel_to_griffinlim(mel_norm: torch.Tensor, length: int, device: torch.device) -> torch.Tensor:
    """
    Convert normalized mel → audio via Griffin-Lim.
    mel_norm: [1, 64, T] (normalized dB)
    """
    # Unnormalize
    mel_db = mel_norm * cfg.norm_std + cfg.norm_mean
    mel_db = torch.clamp(mel_db, min=-80, max=0)

    # dB → magnitude (AmplitudeToDB does 20*log10(x), so invert)
    mel_mag = torch.pow(10.0, mel_db / 20.0)  # [1, 64, T]

    # Mel filterbank: melscale_fbanks returns [n_stft, n_mels]
    # So mel_mag = fb.T @ linear_mag → linear_mag = fb @ mel_mag
    n_stft = cfg.n_fft // 2 + 1
    fb = torchaudio.functional.melscale_fbanks(
        n_stft, cfg.f_min, cfg.f_max, cfg.n_mels, cfg.sample_rate
    ).to(device)  # [n_stft, n_mels]

    # [n_stft, n_mels] @ [n_mels, T] → [n_stft, T]
    linear_mag = torch.matmul(fb, mel_mag)  # [1, n_stft, T]

    # Griffin-Lim
    gl = T.GriffinLim(
        n_fft=cfg.n_fft, hop_length=cfg.hop_length, power=1,
        n_iter=32, momentum=0.99, length=length,
    ).to(device)

    # GriffinLim expects [B, F, T] where F = n_fft//2+1
    return gl(linear_mag.squeeze(0)).unsqueeze(0)  # [1, length]


# ── Main ──────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Force CPU for inference — MPS doesn't support large Conv1d in HiFi-GAN
    device = torch.device("cpu")
    print(f"🎧 HiFi-GAN comparison — device={device}")
    print(f"   Output → {OUT_DIR}/\n")

    # Load models
    meltrain_path = "models/hifigan_generator_meltrain_best.pth"
    gantrain_path = "models/hifigan_generator_train.pth"

    gen_mel = load_generator(meltrain_path, device) if os.path.exists(meltrain_path) else None
    gen_gan = load_generator(gantrain_path, device) if os.path.exists(gantrain_path) else None
    infer_mel = HiFiGANInference(gen_mel, device) if gen_mel else None
    infer_gan = HiFiGANInference(gen_gan, device) if gen_gan else None

    print(f"   Mel-only model: {'✅' if gen_mel else '❌ not found'}")
    print(f"   GAN model:      {'✅' if gen_gan else '❌ not found'}")

    for path in TEST_FILES:
        if not os.path.exists(path):
            print(f"⚠️  Not found: {path}")
            continue

        label = os.path.basename(os.path.dirname(path))
        base = os.path.splitext(os.path.basename(path))[0]
        print(f"\n{'='*50}")
        print(f"🐾 {label}/{base}.wav")
        print(f"{'='*50}")

        # Real audio
        real = load_real_audio(path)
        mel = audio_to_mel(real).unsqueeze(0).to(device)  # [1, 64, T]
        real_len = real.shape[-1]
        n_frames = mel.shape[-1]

        # 1. Original
        save_audio(real, f"{OUT_DIR}/{label}_{base}_01_original.wav")
        print(f"  01 ✅ original.wav")

        # 2. Griffin-Lim baseline
        gl = mel_to_griffinlim(mel.squeeze(0), real_len, device).cpu()
        save_audio(gl, f"{OUT_DIR}/{label}_{base}_02_griffinlim.wav")
        print(f"  02 ✅ griffinlim.wav")

        # 3. HiFi-GAN meltrain
        if infer_mel:
            with torch.no_grad():
                audio_mel = gen_mel(mel, target_length=real_len).cpu()
            save_audio(audio_mel, f"{OUT_DIR}/{label}_{base}_03_meltrain.wav")
            print(f"  03 ✅ meltrain.wav")

        # 4. HiFi-GAN full GAN
        if gen_gan:
            with torch.no_grad():
                audio_gan = gen_gan(mel, target_length=real_len).cpu()
            save_audio(audio_gan, f"{OUT_DIR}/{label}_{base}_04_gantrain.wav")
            print(f"  04 ✅ gantrain.wav")

    print(f"\n{'='*50}")
    print(f"📁 All files in: {OUT_DIR}/")
    print(f"\n🎧 Listen in order:")
    print(f"   01 = Original (ground truth)")
    print(f"   02 = Griffin-Lim (baseline)")
    print(f"   03 = HiFi-GAN meltrain only")
    print(f"   04 = HiFi-GAN full GAN")


if __name__ == "__main__":
    main()
