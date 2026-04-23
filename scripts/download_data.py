#!/usr/bin/env python3
"""
download_data.py — Download animal sound clips from FSD50K
===========================================================

Pulls only the files you need via Git LFS and organizes them into
data/animal_audio/{Class}/ folders with a metadata.csv.

Edit the CONFIG section below to change which classes and how many
files per class to download.

Usage:
    # Download everything available (default):
    python scripts/download_data.py

    # Already cloned, just re-run to change counts:
    python scripts/download_data.py

Requirements:
    - git and git-lfs installed (brew install git-lfs)
"""

import csv
import os
import shutil
import subprocess
import sys

# ╔══════════════════════════════════════════════════════════════╗
# ║  CONFIG — edit this to change what gets downloaded          ║
# ╚══════════════════════════════════════════════════════════════╝

# FSD50K HuggingFace repo
FSD50K_REPO = "https://huggingface.co/datasets/Fhrozen/FSD50k"

# Where data lives (relative to project root)
DATA_DIR = "data"
FSD50K_DIR = os.path.join(DATA_DIR, "fsd50k_metadata")
OUTPUT_DIR = os.path.join(DATA_DIR, "animal_audio")

# Classes to download
# Each entry: { display_name: [list of FSD50K AudioSet mids] }
# Find more mids in: data/fsd50k_metadata/labels/vocabulary.csv
CLASS_MIDS = {
    "Dog":      ["/m/0bt9lr", "/m/05tny_"],          # Dog, Bark
    "Cat":      ["/m/01yrx", "/m/07qrkrw"],           # Cat, Meow
    "Rooster":  ["/m/09b5t"],                          # Chicken_and_rooster
    "Frog":     ["/m/09ld4"],                           # Frog
    "Crow":     ["/m/04s8yn"],                          # Crow
    "Insect":   ["/m/03vt0", "/m/09xqv"],              # Insect, Cricket
    "Hen":      ["/m/025rv6n"],                         # Fowl
    "Noise":    ["/m/0btp2", "/m/06mb1", "/m/0ngt1", "/m/03m9d0z"],  # Traffic, Rain, Thunder, Wind
}

# Maximum files available per class in FSD50K (as of 2024):
#
#   Class      Total    Max train  Max val  Max test   Source labels
#   ─────────  ───────  ─────────  ───────  ────────   ──────────────────────────────
#   Dog          750       525       112      113       Dog (/m/0bt9lr), Bark (/m/05tny_)
#   Noise      1,222       855       183      184       Traffic, Rain, Thunder, Wind
#   Insect       371       259        55       57       Insect (/m/03vt0), Cricket (/m/09xqv)
#   Cat          303       212        45       46       Cat (/m/01yrx), Meow (/m/07qrkrw)
#   Rooster      136        95        20       21       Chicken_and_rooster (/m/09b5t)
#   Hen           86        60        12       14       Fowl (/m/025rv6n)
#   Crow          72        50        10       12       Crow (/m/04s8yn)
#   Frog          61        42         9       10       Frog (/m/09ld4)
#   ─────────  ───────  ─────────  ───────  ────────
#   TOTAL      3,001     2,098      446      457
#
# These are the hard limits. If FILES_PER_CLASS exceeds them, you just get the max.

# Files per class — set to None to download ALL available.
# If you set a number, that class will be capped.
# Example:  {"Dog": None, "Frog": 50}  → all dogs, max 50 frogs
FILES_PER_CLASS = None

# How to split each class's files into train/val/test (percentages)
SPLIT_RATIOS = {
    "train": 0.7,
    "val":   0.15,
    "test":  0.15,
}

# ╔══════════════════════════════════════════════════════════════╗
# ║  END CONFIG — don't edit below unless you know what you do  ║
# ╚══════════════════════════════════════════════════════════════╝


def run(cmd: str, check: bool = True) -> str:
    """Run shell command and return output."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        sys.exit(1)
    return result.stdout.strip()


def find_project_root() -> str:
    """Find project root by looking for roadmap.md."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def clone_fsd50k(fsd50k_dir: str):
    """Clone FSD50K repo if not already present."""
    if os.path.exists(fsd50k_dir):
        print(f"✓ FSD50K repo already exists at {fsd50k_dir}")
        return

    print(f"\n[1/4] Cloning FSD50K repo (metadata only, no audio yet)...")
    run(f"git clone {FSD50K_REPO} {fsd50k_dir}")
    print("  Clone done.")


def init_lfs(fsd50k_dir: str):
    """Ensure Git LFS is installed and initialized."""
    print(f"\n[2/4] Setting up Git LFS...")
    run("git lfs version", check=True)
    run(f"cd {fsd50k_dir} && git lfs install")
    print("  LFS ready.")


def resolve_limit(label: str) -> int | None:
    """Get file limit for a label from FILES_PER_CLASS config."""
    if FILES_PER_CLASS is None:
        return None
    if isinstance(FILES_PER_CLASS, int):
        return FILES_PER_CLASS
    if isinstance(FILES_PER_CLASS, dict):
        return FILES_PER_CLASS.get(label, None)
    return None


def build_subset(fsd50k_dir: str) -> list[tuple[str, str, str]]:
    """Build list of (fname, label, split) from FSD50K dev.csv."""
    print(f"\n[3/4] Selecting files from FSD50K labels...")

    # Build mid → label lookup
    mid_to_label: dict[str, str] = {}
    for label, mids in CLASS_MIDS.items():
        for mid in mids:
            mid_to_label[mid] = label

    dev_csv = os.path.join(fsd50k_dir, "labels", "dev.csv")
    if not os.path.exists(dev_csv):
        print(f"  ERROR: {dev_csv} not found. Did the clone succeed?")
        sys.exit(1)

    # Collect all matching files, grouped by label
    by_label: dict[str, list[str]] = {}  # label → [fname, ...]
    seen_fnames: set[str] = set()

    with open(dev_csv) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            fname = row[0]
            mids = row[2].split(",")
            if fname in seen_fnames:
                continue
            for mid in mids:
                if mid in mid_to_label:
                    label = mid_to_label[mid]
                    by_label.setdefault(label, []).append(fname)
                    seen_fnames.add(fname)
                    break

    # Split ratios
    ratio_train = SPLIT_RATIOS["train"]
    ratio_val = SPLIT_RATIOS["val"]
    ratio_test = SPLIT_RATIOS["test"]
    ratio_sum = ratio_train + ratio_val + ratio_test

    # Select & split per class
    result: list[tuple[str, str, str]] = []
    print()
    for label in sorted(by_label.keys()):
        fnames = by_label[label]
        limit = resolve_limit(label)

        # Cap if limit set
        if limit is not None and limit < len(fnames):
            fnames = fnames[:limit]
            capped = f" (capped from {len(by_label[label])})"
        else:
            capped = ""

        # Split into train/val/test
        n = len(fnames)
        n_train = int(n * ratio_train / ratio_sum)
        n_val = int(n * ratio_val / ratio_sum)
        n_test = n - n_train - n_val  # remainder goes to test

        for fname in fnames[:n_train]:
            result.append((fname, label, "train"))
        for fname in fnames[n_train:n_train + n_val]:
            result.append((fname, label, "val"))
        for fname in fnames[n_train + n_val:]:
            result.append((fname, label, "test"))

        print(f"  {label:10s}  {n:4d} files  ({n_train} train + {n_val} val + {n_test} test){capped}")

    print(f"  {'TOTAL':10s}  {len(result):4d} files")
    return result


def pull_and_organize(fsd50k_dir: str, output_dir: str, subset: list[tuple[str, str, str]]):
    """Git LFS pull selected files, then copy to organized folders."""
    print(f"\n[4/4] Downloading & organizing audio files...")

    audio_src = os.path.join(fsd50k_dir, "clips", "dev")

    # Build list of files that need pulling
    to_pull: list[str] = []
    for fname, _, _ in subset:
        path = os.path.join(audio_src, f"{fname}.wav")
        if not os.path.exists(path) or os.path.getsize(path) < 200:
            to_pull.append(f"clips/dev/{fname}.wav")

    # Pull via LFS in batches if many files (shell argument length limit)
    if to_pull:
        print(f"  Pulling {len(to_pull)} files via Git LFS...")
        batch_size = 200
        for i in range(0, len(to_pull), batch_size):
            batch = to_pull[i:i + batch_size]
            include_str = ",".join(batch)
            run(f"cd {fsd50k_dir} && git lfs pull --include=\"{include_str}\"")
        print("  LFS pull done.")
    else:
        print("  All files already downloaded.")

    # Copy to organized output
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    skipped = 0
    for fname, label, split in subset:
        src = os.path.join(audio_src, f"{fname}.wav")
        dest_dir = os.path.join(output_dir, label)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, f"{fname}.wav")

        if os.path.exists(src) and os.path.getsize(src) > 200:
            shutil.copy2(src, dest)
        else:
            skipped += 1

    # Write metadata.csv
    meta_path = os.path.join(output_dir, "metadata.csv")
    with open(meta_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["fname", "label", "split"])
        for fname, label, split in subset:
            writer.writerow([fname, label, split])

    if skipped:
        print(f"  ⚠ Skipped {skipped} files (still LFS pointers — pull may have failed)")

    # Summary
    print(f"\n  Organized into: {os.path.abspath(output_dir)}/")
    for label in sorted(CLASS_MIDS.keys()):
        folder = os.path.join(output_dir, label)
        n = len([f for f in os.listdir(folder) if f.endswith(".wav")]) if os.path.exists(folder) else 0
        print(f"    {label:10s}  {n:4d} files")

    print(f"    metadata.csv written")
    print(f"\n✅ Done! Open {os.path.abspath(output_dir)} to listen.")


def main():
    root = find_project_root()
    os.chdir(root)

    print("=" * 60)
    print("Animal Sound Generator — Data Download")
    print("=" * 60)
    print(f"Classes:     {list(CLASS_MIDS.keys())}")
    if FILES_PER_CLASS is None:
        print(f"Per class:   ALL available")
    elif isinstance(FILES_PER_CLASS, int):
        print(f"Per class:   max {FILES_PER_CLASS}")
    else:
        print(f"Per class:   {FILES_PER_CLASS}")
    print(f"Split:       {SPLIT_RATIOS['train']:.0%} train / {SPLIT_RATIOS['val']:.0%} val / {SPLIT_RATIOS['test']:.0%} test")
    print(f"Output:      {OUTPUT_DIR}")

    fsd50k_dir = os.path.abspath(FSD50K_DIR)
    output_dir = os.path.abspath(OUTPUT_DIR)

    clone_fsd50k(fsd50k_dir)
    init_lfs(fsd50k_dir)
    subset = build_subset(fsd50k_dir)
    pull_and_organize(fsd50k_dir, output_dir, subset)


if __name__ == "__main__":
    main()
