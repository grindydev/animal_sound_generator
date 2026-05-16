#!/usr/bin/env python3
"""
import_urbansound.py — Extract classes from UrbanSound8K into our dataset.

UrbanSound8K class mapping:
  3 = dog_bark → Dog
  Others: car_horn, children_playing, drilling, engine_idling, 
          gun_shot, jackhammer, siren, street_music

Usage:
  # In Colab:
  !wget -q https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz
  !tar -xzf UrbanSound8K.tar.gz
  !python src/scripts/import_urbansound.py --source UrbanSound8K --target data/combined
"""
import os
import sys
import shutil
import argparse
import csv
from pathlib import Path

# UrbanSound8K labels
US8K_CLASSES = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music",
}

# Map UrbanSound8K class labels to our classes
CLASS_MAP = {
    "dog_bark": "Dog",
}


def import_urbansound(source_dir: str, target_dir: str):
    """Extract relevant classes from UrbanSound8K into target_dir."""
    audio_dir = os.path.join(source_dir, "audio")
    meta_path = os.path.join(source_dir, "metadata", "UrbanSound8K.csv")

    if not os.path.exists(audio_dir):
        print(f"❌ Audio directory not found: {audio_dir}")
        return
    if not os.path.exists(meta_path):
        print(f"❌ Metadata not found: {meta_path}")
        return

    # Read metadata
    file_to_class = {}
    with open(meta_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("slice_file_name", "")
            class_id = int(row.get("classID", -1))
            file_to_class[fname] = class_id

    # Organize
    counts = {}
    for fold in range(1, 11):
        fold_dir = os.path.join(audio_dir, f"fold{fold}")
        if not os.path.isdir(fold_dir):
            continue
        for fname in sorted(os.listdir(fold_dir)):
            if not fname.endswith(".wav"):
                continue
            class_id = file_to_class.get(fname, -1)
            us8k_name = US8K_CLASSES.get(class_id, "")
            our_class = CLASS_MAP.get(us8k_name)
            if our_class is None:
                continue

            cls_dir = os.path.join(target_dir, our_class)
            os.makedirs(cls_dir, exist_ok=True)

            src = os.path.join(fold_dir, fname)
            dst = os.path.join(cls_dir, f"us8k_{fname}")
            shutil.copy2(src, dst)
            counts[our_class] = counts.get(our_class, 0) + 1

    print(f"✅ UrbanSound8K imported → {target_dir}/")
    for cls_name, n in sorted(counts.items()):
        print(f"   {cls_name:8s}: {n:4d} files")
    print(f"   {'TOTAL':8s}: {sum(counts.values()):4d} files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="UrbanSound8K",
                       help="Path to extracted UrbanSound8K directory")
    parser.add_argument("--target", default="data/combined",
                       help="Target directory")
    args = parser.parse_args()
    import_urbansound(args.source, args.target)


if __name__ == "__main__":
    main()
