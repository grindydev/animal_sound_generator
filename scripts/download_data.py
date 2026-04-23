#!/usr/bin/env python3
"""
download_data.py — Download animal sound clips from FSD50K
===========================================================

Downloads ONLY the files for the 8 configured animal classes directly
via HTTP from HuggingFace. No git clone, no Git LFS — just the files
you actually need.

Usage:
    python scripts/download_data.py

Requirements:
    - Python 3.10+
    - requests (pip install requests)
"""

import csv
import io
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    print("ERROR: 'requests' is required. Install with: pip install requests")
    sys.exit(1)

# Force IPv4 — avoids slow IPv6 timeouts when connecting to HF CDN
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)


socket.getaddrinfo = _ipv4_only

# ╔══════════════════════════════════════════════════════════════╗
# ║  CONFIG — edit this to change what gets downloaded          ║
# ╚══════════════════════════════════════════════════════════════╝

# FSD50K HuggingFace repo (used for HTTP download URLs)
HF_REPO = "Fhrozen/FSD50k"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"

# Where to put output (relative to project root)
DATA_DIR = "data"
LABELS_DIR = os.path.join(DATA_DIR, "fsd50k_labels")
OUTPUT_DIR = os.path.join(DATA_DIR, "animal_audio")

# Classes to download
# Each entry: { display_name: [list of FSD50K AudioSet mids] }
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

# Files per class — set to None for ALL available, or an int to cap.
FILES_PER_CLASS = None

# Parallel download threads (increase for faster networks)
DOWNLOAD_WORKERS = 8

# How to split each class's files into train/val/test (percentages)
SPLIT_RATIOS = {
    "train": 0.7,
    "val":   0.15,
    "test":  0.15,
}

# ╔══════════════════════════════════════════════════════════════╗
# ║  END CONFIG — don't edit below unless you know what you do  ║
# ╚══════════════════════════════════════════════════════════════╝


def find_project_root() -> str:
    """Find project root by walking up from this script."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def download_text(url: str, desc: str = "") -> str:
    """Download a text file from URL. Returns content as string."""
    if desc:
        print(f"  Downloading {desc}...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


# Shared session for connection pooling
_session = requests.Session()
_session.headers.update({"User-Agent": "animal-sound-downloader/1.0"})


def download_file(url: str, dest: str) -> tuple[str, bool]:
    """Download a binary file. Returns (fname, success)."""
    fname = os.path.basename(dest)
    try:
        resp = _session.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        return (fname, True)
    except Exception as e:
        print(f"    FAILED {fname}: {e}")
        return (fname, False)


def fetch_labels_csv(labels_dir: str) -> str:
    """Ensure dev.csv is available locally. Download if missing."""
    dev_csv = os.path.join(labels_dir, "dev.csv")
    if os.path.exists(dev_csv):
        print(f"  Using cached {dev_csv}")
        with open(dev_csv) as f:
            return f.read()

    print(f"  Fetching dev.csv from HuggingFace (~2 MB)...")
    content = download_text(f"{HF_BASE_URL}/labels/dev.csv", "dev.csv")
    os.makedirs(labels_dir, exist_ok=True)
    with open(dev_csv, "w") as f:
        f.write(content)
    print(f"  Saved to {dev_csv}")
    return content


def resolve_limit(label: str) -> int | None:
    """Get file limit for a label from FILES_PER_CLASS config."""
    if FILES_PER_CLASS is None:
        return None
    if isinstance(FILES_PER_CLASS, int):
        return FILES_PER_CLASS
    if isinstance(FILES_PER_CLASS, dict):
        return FILES_PER_CLASS.get(label, None)
    return None


def build_subset(csv_text: str) -> list[tuple[str, str, str]]:
    """Parse dev.csv and select files matching our classes.

    Returns list of (fname, label, split).
    """
    print("\n  Selecting files for target classes...")

    # Build mid → label lookup
    mid_to_label: dict[str, str] = {}
    for label, mids in CLASS_MIDS.items():
        for mid in mids:
            mid_to_label[mid] = label

    # Collect matching files grouped by label
    by_label: dict[str, list[str]] = {}
    seen: set[str] = set()

    reader = csv.reader(io.StringIO(csv_text))
    next(reader)  # skip header
    for row in reader:
        fname = row[0]
        mids = row[2].split(",")
        if fname in seen:
            continue
        for mid in mids:
            if mid in mid_to_label:
                label = mid_to_label[mid]
                by_label.setdefault(label, []).append(fname)
                seen.add(fname)
                break

    # Split ratios
    rt = SPLIT_RATIOS["train"]
    rv = SPLIT_RATIOS["val"]
    rsum = rt + rv + SPLIT_RATIOS["test"]

    result: list[tuple[str, str, str]] = []
    print()
    for label in sorted(by_label.keys()):
        fnames = by_label[label]
        limit = resolve_limit(label)

        if limit is not None and limit < len(fnames):
            fnames = fnames[:limit]
            capped = f" (capped from {len(by_label[label])})"
        else:
            capped = ""

        n = len(fnames)
        n_train = int(n * rt / rsum)
        n_val = int(n * rv / rsum)
        n_test = n - n_train - n_val

        for fname in fnames[:n_train]:
            result.append((fname, label, "train"))
        for fname in fnames[n_train:n_train + n_val]:
            result.append((fname, label, "val"))
        for fname in fnames[n_train + n_val:]:
            result.append((fname, label, "test"))

        print(f"    {label:10s}  {n:4d} files  ({n_train} train + {n_val} val + {n_test} test){capped}")

    print(f"    {'TOTAL':10s}  {len(result):4d} files")
    return result


def download_audio(output_dir: str, subset: list[tuple[str, str, str]]):
    """Download WAV files in parallel directly from HuggingFace."""
    print(f"\n  Downloading audio files via HTTP ({DOWNLOAD_WORKERS} workers)...\n")

    # Clear previous output
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)

    total = len(subset)
    done = 0
    failed = 0
    t0 = time.time()

    def _job(item):
        fname, label, split = item
        dest_dir = os.path.join(output_dir, label)
        dest = os.path.join(dest_dir, f"{fname}.wav")
        url = f"{HF_BASE_URL}/clips/dev/{fname}.wav"
        return download_file(url, dest)

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(_job, item): item for item in subset}
        for future in as_completed(futures):
            done += 1
            fname, ok = future.result()
            if not ok:
                failed += 1
            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"    [{done * 100 // total:3d}%] {done}/{total} done, "
                      f"{rate:.1f} files/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0
    print(f"\n    Done: {done - failed}/{total} files in {elapsed:.0f}s ({rate:.1f} files/s)")

    if failed:
        print(f"    ⚠  {failed} files failed to download")

    # Write metadata.csv
    meta_path = os.path.join(output_dir, "metadata.csv")
    with open(meta_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["fname", "label", "split"])
        for fname, label, split in subset:
            writer.writerow([fname, label, split])
    print(f"    metadata.csv written")


def print_summary(output_dir: str):
    """Print final summary."""
    print(f"\n  Organized into: {os.path.abspath(output_dir)}/")
    for label in sorted(CLASS_MIDS.keys()):
        folder = os.path.join(output_dir, label)
        if os.path.exists(folder):
            n = len([f for f in os.listdir(folder) if f.endswith(".wav")])
        else:
            n = 0
        print(f"    {label:10s}  {n:4d} .wav files")
    print()


def main():
    root = find_project_root()
    os.chdir(root)

    print("=" * 60)
    print("Animal Sound Generator — Data Download")
    print("=" * 60)
    print(f"  Classes:     {', '.join(CLASS_MIDS.keys())}")
    print(f"  Workers:     {DOWNLOAD_WORKERS} parallel downloads")
    if FILES_PER_CLASS is None:
        print(f"  Per class:   ALL available")
    else:
        print(f"  Per class:   {FILES_PER_CLASS}")
    print(f"  Split:       {SPLIT_RATIOS['train']:.0%} train / {SPLIT_RATIOS['val']:.0%} val / {SPLIT_RATIOS['test']:.0%} test")
    print(f"  Output:      {OUTPUT_DIR}")
    print()

    # Step 1: Get labels CSV (tiny download, ~2 MB)
    print("[1/3] Fetching labels...")
    labels_dir = os.path.abspath(LABELS_DIR)
    csv_text = fetch_labels_csv(labels_dir)

    # Step 2: Build subset of files we need
    print("\n[2/3] Selecting target files...")
    subset = build_subset(csv_text)
    if not subset:
        print("  ERROR: No matching files found. Check your CLASS_MIDS config.")
        sys.exit(1)

    # Step 3: Download only those files
    print("\n[3/3] Downloading audio (only the files above)...")
    output_dir = os.path.abspath(OUTPUT_DIR)
    download_audio(output_dir, subset)

    print_summary(output_dir)
    print("✅ Done!")


if __name__ == "__main__":
    main()
