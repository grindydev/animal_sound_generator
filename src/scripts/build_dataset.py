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
    old_data_dir: str = "data/animal_audio",
    output_dir: str = "data/combined",
    min_rms: float = MIN_RMS,
    max_noise_files: int = 300,
):
    """Merge ESC-50 and filtered old data into output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    counts = {cls: 0 for cls in CLASS_NAMES}
    from_sources = {cls: {"esc50": 0, "old_crop": 0} for cls in CLASS_NAMES}

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
    #  Phase 2: Energy-filter old data
    # ═══════════════════════════════════════════════════════
    if os.path.exists(old_data_dir) and min_rms > 0:
        print(f"\n🔍 Scanning old data with min_rms={min_rms}...")
        total_scanned = 0
        total_extracted = 0

        for cls_name in CLASS_NAMES:
            src_dir = os.path.join(old_data_dir, cls_name)
            if not os.path.isdir(src_dir):
                continue
            dst_dir = os.path.join(output_dir, cls_name)
            os.makedirs(dst_dir, exist_ok=True)

            # Skip Noise from old data (ESC-50 has enough)
            if cls_name == "Noise":
                continue

            wav_files = [f for f in os.listdir(src_dir) if f.endswith('.wav')]
            for fname in sorted(wav_files):
                path = os.path.join(src_dir, fname)
                wav, sr = load_audio(path)
                if wav is None:
                    continue
                total_scanned += 1

                segments = extract_segments(wav, sr, min_rms=min_rms)
                for si, seg in enumerate(segments):
                    # Cap per-class from old data to avoid imbalance
                    max_from_old = {
                        'Dog': 200, 'Cat': 120, 'Rooster': 80, 'Frog': 60,
                        'Crow': 60, 'Insect': 100, 'Hen': 60, 'Noise': 0,
                    }
                    cap = max_from_old.get(cls_name, 100)
                    if from_sources[cls_name]["old_crop"] >= cap:
                        break

                    import soundfile as sf
                    out_name = f"crop_{fname.replace('.wav','')}_{si:02d}.wav"
                    out_path = os.path.join(dst_dir, out_name)
                    sf.write(out_path, seg.squeeze().numpy(), SAMPLE_RATE)
                    counts[cls_name] += 1
                    from_sources[cls_name]["old_crop"] += 1
                    total_extracted += 1

            if total_scanned % 100 == 0:
                print(f"   Scanned {total_scanned} files, extracted {total_extracted} segments...")

        print(f"\n✅ Old data: scanned {total_scanned} files, extracted {total_extracted} clean segments")

    # ═══════════════════════════════════════════════════════
    #  Report
    # ═══════════════════════════════════════════════════════
    print(f"\n📊 Combined dataset → {output_dir}/")
    print(f"{'Class':10s} {'ESC-50':>7s} {'Old':>7s} {'Total':>7s}")
    print("-" * 35)
    for cls_name in CLASS_NAMES:
        esc = from_sources[cls_name]["esc50"]
        old = from_sources[cls_name]["old_crop"]
        tot = counts[cls_name]
        print(f"{cls_name:10s} {esc:7d} {old:7d} {tot:7d}")
    total = sum(counts.values())
    print(f"{'TOTAL':10s} {sum(from_sources[c]['esc50'] for c in CLASS_NAMES):7d} "
          f"{sum(from_sources[c]['old_crop'] for c in CLASS_NAMES):7d} {total:7d}")


def main():
    parser = argparse.ArgumentParser(description="Build combined dataset")
    parser.add_argument("--esc50-dir", default="data/esc50")
    parser.add_argument("--old-data-dir", default="data/animal_audio")
    parser.add_argument("--output-dir", default="data/combined")
    parser.add_argument("--min-rms", type=float, default=MIN_RMS,
                       help="Minimum RMS energy for a segment (0.02 = audible)")
    parser.add_argument("--max-noise", type=int, default=300)
    args = parser.parse_args()

    build_combined_dataset(
        esc50_dir=args.esc50_dir,
        old_data_dir=args.old_data_dir,
        output_dir=args.output_dir,
        min_rms=args.min_rms,
        max_noise_files=args.max_noise,
    )


if __name__ == "__main__":
    main()
