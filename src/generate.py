"""
generate.py — Animal Sound Generation

Two modes:
  --retrieval    Real mel → perturb → Griffin-Lim (quick, works now)
  --v15-latent   Noise → latent diffusion → decoder → Griffin-Lim (needs training)

Usage:
    # Quick retrieval (no training needed):
    python src/generate.py --label Dog --retrieval
    
    # After Colab training:
    python src/latent_diff/generate.py --label Dog
"""
import os
import sys
import argparse
import random
import torch
import torchaudio
import soundfile as sf
from datetime import datetime
from torchaudio.transforms import InverseMelScale, GriffinLim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

CLASSES = ['Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen']
_mel_index_cache = {}


def load_mel_index(cls_name):
    if cls_name in _mel_index_cache:
        return _mel_index_cache[cls_name]
    path = f"data/mel_index/{cls_name}.pt"
    if not os.path.exists(path):
        return None
    mels = torch.load(path, weights_only=True)
    _mel_index_cache[cls_name] = mels
    return mels


def generate_retrieval(label, variation=0.3):
    """Real mel → interpolate → Griffin-Lim (guaranteed animal sounds)."""
    mels = load_mel_index(label)
    if mels is None or len(mels) == 0:
        return None

    if variation > 0.1 and len(mels) >= 2:
        idx1, idx2 = random.sample(range(len(mels)), 2)
        mel = random.uniform(0.3, 0.7) * mels[idx1] + (1 - random.uniform(0.3, 0.7)) * mels[idx2]
    else:
        mel = mels[random.randint(0, len(mels) - 1)].clone()
    mel = (mel + torch.randn_like(mel) * variation * 0.1).clamp(-3, 3)

    inv_mel = InverseMelScale(n_stft=513, n_mels=64, sample_rate=22050, f_min=0, f_max=11025)
    gl = GriffinLim(n_fft=1024, hop_length=200, win_length=1024, n_iter=64, power=1)
    mel_db = mel * 19.8031 - 18.4903
    mel_power = 10 ** (mel_db.clamp(-80, 0) / 10.0)
    audio = gl(torch.sqrt(inv_mel(mel_power).clamp(min=0)).unsqueeze(0))
    audio = audio / audio.abs().max() * 0.95
    return audio.squeeze(0)


def main():
    parser = argparse.ArgumentParser(description="Animal Sound Generator")
    parser.add_argument("--label", type=str, default=None, help=f"Class: {', '.join(CLASSES)}")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--retrieval", action="store_true", help="Retrieval mode (works now)")
    parser.add_argument("--v15-latent", action="store_true", help="Print path to v15 generator")
    parser.add_argument("--variation", type=float, default=0.3)
    parser.add_argument("--output-dir", type=str, default="outputs/generated")
    args = parser.parse_args()

    if args.v15_latent:
        print("Use: python src/latent_diff/generate.py --label Dog")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    labels = [args.label] if args.label else CLASSES

    for label in labels:
        for i in range(args.count):
            print(f"🎵 {label} ({i+1}/{args.count})...", end=" ", flush=True)
            wav = generate_retrieval(label, args.variation)
            if wav is None:
                print("❌"); continue
            fname = f"{label}_retrieval_v{args.variation}_{ts}_{i+1}.wav"
            sf.write(os.path.join(args.output_dir, fname), wav.numpy(), 22050)
            print(f"✅ {wav.shape[0]/22050:.1f}s")

    print(f"🎉 {args.output_dir}/")


if __name__ == "__main__":
    main()
