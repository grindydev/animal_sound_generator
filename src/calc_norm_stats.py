"""
calc_norm_stats.py — Recalculate SimpleNormalize mean/std after config change.

Run after changing n_fft or any MelSpectrogram parameter.
Takes ~2 minutes. Outputs the new values to paste into SimpleNormalize.

Usage:
    python src/calc_norm_stats.py
"""

import os, sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import AnimalSoundDataset, get_transformations

# Need identity-normalize for raw stats
import torch.nn as nn
class PassThrough(nn.Module):
    def forward(self, x): return x


def main():
    ds = AnimalSoundDataset(str(Path(__file__).resolve().parent.parent / 'data' / 'animal_audio'))
    
    # Build transform WITHOUT SimpleNormalize
    train_tfm = nn.Sequential(
        *list(get_transformations()[0])[:2],  # MelSpectrogram + AmplitudeToDB
        PassThrough(),                         # skip SimpleNormalize
    )
    
    all_specs = []
    n = len(ds)
    for i in range(n):
        wav, _ = ds[i]
        wav = wav.unsqueeze(0)  # [1, 1, samples]
        with torch.no_grad():
            spec = train_tfm(wav)  # [1, 1, 64, T]
        all_specs.append(spec)
        if i % 1000 == 0:
            print(f"  {i}/{n}...", flush=True)
    
    all_specs = torch.cat(all_specs, dim=0)  # [N, 1, 64, T]
    mean = all_specs.mean().item()
    std = all_specs.std().item()
    
    print(f"\nNew normalization stats (n_fft=1024):")
    print(f"  Mean: {mean:.4f}")
    print(f"  Std:  {std:.4f}")
    print(f"\nUpdate SimpleNormalize in data_loader.py:")
    print(f'  class SimpleNormalize(nn.Module):')
    print(f'      def __init__(self, mean = {mean:.4f}, std = {std:.4f}):')
    print(f"\nOld values (n_fft=400):  mean=-30.8645, std=21.1952")


if __name__ == "__main__":
    main()
