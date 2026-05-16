#!/usr/bin/env python3
"""
build_dataset.py — Combine ESC-50 + energy-filtered old data into one clean dataset.

Strategy:
  1. Copy all ESC-50 files (already clean 5s clips)
  2. Scan old animal_audio files, extract 5s high-energy segments
  3. Output: data/combined/<ClassName>/ with ~1000+ clean segments

Usage:
  python src/scripts/build_dataset.py
"""
import os
import sys
import shutil
import argparse
import warnings
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# ═══════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════

CLASS_NAMES = ['Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen', 'Noise']

# Minimum RMS for a 5s chunk to be considered "has animal sound"
MIN_RMS = 0.02
SEGMENT_SECONDS = 5.0
STRIDE_SECONDS = 2.0  # overlap between extracted segments
SAMPLE_RATE = 22050


def load_audio(path: str) -> torch.Tensor:
    """Load audio with torchaudio, handle errors."""
    import torchaudio
    try:
        wav, sr = torchaudio.load(path)
        return wav, sr
    except Exception as e:
        return None, None


def extract_segments(wav: torch.Tensor, sr: int, min_rms: float = MIN_RMS,
                     segment_s: float = SEGMENT_SECONDS,
                     stride_s: float = STRIDE_SECONDS) -> list:
    """
    Extract all 5s chunks from a long audio file that have sufficient energy.
    Returns list of [1, samples] tensors.
    """
    if sr != SAMPLE_RATE:
        import torchaudio
        wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
        sr = SAMPLE_RATE

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    segment_samples = int(segment_s * sr)
    stride_samples = int(stride_s * sr)
    total = wav.shape[-1]

    segments = []
    for start in range(0, total - segment_samples + 1, stride_samples):
        chunk = wav[:, start:start + segment_samples]
        rms = chunk.pow(2).mean().sqrt().item()
        if rms >= min_rms:
            segments.append(chunk.clone())

    return segments


def build_combined_dataset(
    esc50_dir: str = "data/esc50",
    urbansound_dir: str = None,
    output_dir: str = "data/combined",
    max_noise_files: int = 300,
):
    """Merge ESC-50 + UrbanSound8K into output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    counts = {cls: 0 for cls in CLASS_NAMES}
    from_sources = {cls: {"esc50": 0, "urbansound8k": 0} for cls in CLASS_NAMES}

    # ═══════════════════════════════════════════════════════
    #  Phase 1: Copy ESC-50 files
    # ═══════════════════════════════════════════════════════
    if os.path.exists(esc50_dir):
        for cls_name in CLASS_NAMES:
            src_dir = os.path.join(esc50_dir, cls_name)
            if not os.path.isdir(src_dir):
                continue
            dst_dir = os.path.join(output_dir, cls_name)
            os.makedirs(dst_dir, exist_ok=True)
            for fname in os.listdir(src_dir):
                if fname.endswith('.wav'):
                    shutil.copy2(os.path.join(src_dir, fname),
                                os.path.join(dst_dir, f"esc50_{fname}"))
                    counts[cls_name] += 1
                    from_sources[cls_name]["esc50"] += 1
        print(f"✅ ESC-50 copied: {sum(from_sources[c]['esc50'] for c in CLASS_NAMES)} files")

    # ═══════════════════════════════════════════════════════
    #  Phase 2: UrbanSound8K (dog barks, ~1,000 files)
    # ═══════════════════════════════════════════════════════
    if urbansound_dir and os.path.exists(urbansound_dir):
        audio_dir = os.path.join(urbansound_dir, "audio")
        meta_path = os.path.join(urbansound_dir, "metadata", "UrbanSound8K.csv")
        if os.path.exists(audio_dir) and os.path.exists(meta_path):
            import csv
            file_to_class = {}
            with open(meta_path, "r") as f:
                for row in csv.DictReader(f):
                    file_to_class[row.get("slice_file_name", "")] = int(row.get("classID", -1))
            
            US8K_CLASSES = {0:"air_conditioner",1:"car_horn",2:"children_playing",3:"dog_bark",
                           4:"drilling",5:"engine_idling",6:"gun_shot",7:"jackhammer",8:"siren",9:"street_music"}
            MAPPING = {"dog_bark": "Dog"}
            
            for fold in range(1, 11):
                fold_dir = os.path.join(audio_dir, f"fold{fold}")
                if not os.path.isdir(fold_dir): continue
                for fname in sorted(os.listdir(fold_dir)):
                    if not fname.endswith(".wav"): continue
                    us8k_name = US8K_CLASSES.get(file_to_class.get(fname, -1), "")
                    our_class = MAPPING.get(us8k_name)
                    if our_class is None: continue
                    
                    dst_dir = os.path.join(output_dir, our_class)
                    os.makedirs(dst_dir, exist_ok=True)
                    shutil.copy2(os.path.join(fold_dir, fname),
                                os.path.join(dst_dir, f"us8k_{fname}"))
                    counts[our_class] += 1
                    from_sources[our_class]["urbansound8k"] += 1
            
            us8k_total = sum(from_sources[c]["urbansound8k"] for c in CLASS_NAMES)
            print(f"✅ UrbanSound8K imported: {us8k_total} files")

    # ═══════════════════════════════════════════════════════
    #  Report
    # ═══════════════════════════════════════════════════════
    print(f"\n📊 Combined dataset → {output_dir}/")
    print(f"{'Class':10s} {'ESC-50':>7s} {'US8K':>7s} {'Total':>7s}")
    print("-" * 35)
    for cls_name in CLASS_NAMES:
        esc = from_sources[cls_name]["esc50"]
        us8k = from_sources[cls_name]["urbansound8k"]
        tot = counts[cls_name]
        print(f"{cls_name:10s} {esc:7d} {us8k:7d} {tot:7d}")
    total = sum(counts.values())
    esc_tot = sum(from_sources[c]["esc50"] for c in CLASS_NAMES)
    us8k_tot = sum(from_sources[c]["urbansound8k"] for c in CLASS_NAMES)
    print(f"{'TOTAL':10s} {esc_tot:7d} {us8k_tot:7d} {total:7d}")


def main():
    parser = argparse.ArgumentParser(description="Build combined dataset")
    parser.add_argument("--esc50-dir", default="data/esc50")
    parser.add_argument("--urbansound-dir", default=None,
                       help="Path to extracted UrbanSound8K directory")
    parser.add_argument("--output-dir", default="data/combined")

    args = parser.parse_args()
    build_combined_dataset(
        esc50_dir=args.esc50_dir,
        urbansound_dir=args.urbansound_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
