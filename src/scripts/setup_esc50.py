#!/usr/bin/env python3
"""
setup_esc50.py — Organize ESC-50 audio into our class structure.

Maps ESC-50 categories to our 8 animal classes:
  Dog, Cat, Rooster, Frog, Crow, Insect, Hen, Noise

Usage:
  # In Colab:
  !wget -q https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip
  !unzip -q master.zip
  !python src/scripts/setup_esc50.py --source ESC-50-master/audio --target data/esc50

  # Local:
  python src/scripts/setup_esc50.py
"""
import os
import sys
import shutil
import argparse
import csv
from pathlib import Path

# ESC-50 category → our class
CATEGORY_MAP = {
    "dog": "Dog",
    "cat": "Cat",
    "rooster": "Rooster",
    "frog": "Frog",
    "crow": "Crow",
    "hen": "Hen",
    "chicken": "Hen",        # alias
    "insects": "Insect",
    "crickets": "Insect",    # alias
    "chirping_birds": "Crow",  # similar
    # Noise/background:
    "rain": "Noise",
    "wind": "Noise",
    "thunderstorm": "Noise",
    "sea_waves": "Noise",
    "water_drops": "Noise",
    "pouring_water": "Noise",
    "crackling_fire": "Noise",
}


def setup_esc50(source_audio: str, target_dir: str, meta_dir: str = None):
    """Copy ESC-50 audio files into target_dir/<ClassName>/ structure."""
    # Find metadata CSV
    if meta_dir is None:
        parent = os.path.dirname(source_audio)
        meta_dir = os.path.join(parent, "meta")
    
    csv_path = os.path.join(meta_dir, "esc50.csv")
    if not os.path.exists(csv_path):
        print(f"❌ Metadata not found at {csv_path}")
        print(f"   Make sure you downloaded the full ESC-50 repo, not just the audio zip.")
        return
    
    # Read metadata
    filename_to_category = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename_to_category[row["filename"]] = row["category"]
    
    # Organize
    os.makedirs(target_dir, exist_ok=True)
    counts = {}
    
    for fname in sorted(os.listdir(source_audio)):
        if not fname.endswith(".wav"):
            continue
        
        category = filename_to_category.get(fname, "")
        cls = CATEGORY_MAP.get(category)
        if cls is None:
            continue  # skip irrelevant categories
        
        cls_dir = os.path.join(target_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        src = os.path.join(source_audio, fname)
        dst = os.path.join(cls_dir, fname)
        shutil.copy2(src, dst)
        counts[cls] = counts.get(cls, 0) + 1
    
    print(f"✅ ESC-50 organized → {target_dir}/")
    for cls in sorted(counts.keys()):
        print(f"   {cls:8s}: {counts[cls]:3d} files")
    print(f"   {'TOTAL':8s}: {sum(counts.values()):3d} files")

    missing = set(CATEGORY_MAP.values()) - set(counts.keys())
    if missing:
        print(f"\n⚠️  Missing classes: {missing}")
        print(f"   These will be generated as silence. Training will still work.")


def main():
    parser = argparse.ArgumentParser(description="Organize ESC-50 for animal sound generator")
    parser.add_argument("--source", default="ESC-50-master/audio",
                       help="Path to ESC-50 audio directory")
    parser.add_argument("--target", default="data/esc50",
                       help="Target directory for organized classes")
    parser.add_argument("--meta", default=None,
                       help="Path to meta/ directory (auto-detected if not specified)")
    args = parser.parse_args()
    
    setup_esc50(args.source, args.target, args.meta)


if __name__ == "__main__":
    main()
