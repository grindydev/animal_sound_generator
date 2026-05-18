"""
deduplicate.py — Keep only diverse segments, cap at 500/class

Source files (fs_12345) got split into many similar-sounding segments.
This script groups by source, keeps the most diverse 3-5 per source,
and caps each class at 500 files.

Usage:
  python src/scripts/deduplicate.py
  python src/scripts/deduplicate.py --max-per-source 4 --max-per-class 500 --dry-run
"""
import os, sys, argparse, random
import torch
import torchaudio
import torch.nn.functional as F

DATA_DIR = "data/animal1000"
MAX_PER_SOURCE = 4      # max segments from same original recording
MAX_PER_CLASS = 500     # cap per class
SAMPLE_RATE = 22050


def compute_features(wav):
    """Compute audio features for diversity comparison."""
    wav = wav.float()

    # RMS
    rms = (wav ** 2).mean().sqrt()

    # Spectral centroid
    spec = torch.fft.rfft(wav, dim=-1)
    mag = spec.abs()
    freqs = torch.linspace(0, SAMPLE_RATE/2, mag.shape[-1])
    centroid = (mag * freqs).sum() / mag.sum()

    # Zero-crossing rate
    zcr = ((wav[..., :-1] * wav[..., 1:]) < 0).float().mean()

    # Spectral flatness
    log_mag = torch.log(mag + 1e-8)
    flatness = torch.exp(log_mag.mean()) / (mag.mean() + 1e-8)

    # Onset strength (change in energy)
    frame_size = SAMPLE_RATE // 100
    n_frames = wav.shape[-1] // frame_size
    if n_frames > 1:
        frames = wav[:, :n_frames * frame_size].view(-1, frame_size)
        frame_rms = (frames ** 2).mean(dim=-1).sqrt()
        onset = (frame_rms[1:] - frame_rms[:-1]).abs().mean()
    else:
        onset = torch.tensor(0.0)

    return torch.tensor([rms, centroid, zcr, flatness, onset])


def select_diverse(files_with_features, n_keep):
    """Select N most diverse segments via max-min feature distance."""
    if len(files_with_features) <= n_keep:
        return [f for f, _ in files_with_features]

    selected = [files_with_features[0]]
    remaining = files_with_features[1:]

    while len(selected) < n_keep and remaining:
        # Find file furthest from all selected
        max_min_dist = -1
        best_idx = 0
        for i, (_, feats) in enumerate(remaining):
            min_dist = min(F.pairwise_distance(feats.unsqueeze(0), s.unsqueeze(0)).item()
                          for _, s in selected)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return [f for f, _ in selected]


def process(dry_run=False):
    random.seed(42)
    total_kept = 0
    total_removed = 0

    for cls_name in sorted(os.listdir(DATA_DIR)):
        cls_dir = os.path.join(DATA_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        wavs = [f for f in os.listdir(cls_dir) if f.endswith(".wav")]
        if not wavs:
            continue

        # Group by source: fs_12345_1.wav, fs_12345_2.wav → source = fs_12345
        sources = {}
        non_fs = []
        for fname in wavs:
            if fname.startswith("fs_"):
                # Source ID is everything before the last underscore
                parts = fname.rsplit("_", 1)
                if len(parts) == 2 and parts[1].split(".")[0].isdigit():
                    src_id = parts[0]  # e.g., fs_641420
                else:
                    src_id = fname.replace(".wav", "")
                sources.setdefault(src_id, []).append(fname)
            else:
                non_fs.append(fname)

        kept = []
        removed = 0

        print(f"\n📦 {cls_name}: {len(wavs)} files, {len(sources)} sources, {len(non_fs)} non-fs")

        # Keep all non-fs files (Kaggle, ESC-50, UrbanSound8K)
        kept.extend(non_fs)

        # For each fs source, keep most diverse segments
        for src_id, files in sorted(sources.items()):
            if len(files) == 1:
                kept.append(files[0])
                continue

            # Compute features for diversity selection
            files_with_feats = []
            for fname in files:
                path = os.path.join(cls_dir, fname)
                try:
                    wav, sr = torchaudio.load(path)
                    if sr != SAMPLE_RATE:
                        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                        wav = resampler(wav)
                    if wav.shape[0] > 1:
                        wav = wav.mean(0, keepdim=True)
                    feats = compute_features(wav)
                    files_with_feats.append((fname, feats))
                except Exception:
                    files_with_feats.append((fname, torch.zeros(5)))

            n_keep = min(MAX_PER_SOURCE, len(files))
            diverse = select_diverse(files_with_feats, n_keep)
            kept.extend(diverse)
            removed += len(files) - len(diverse)

        # Cap at MAX_PER_CLASS
        if len(kept) > MAX_PER_CLASS:
            # Keep non-fs files first, then randomly sample fs files
            kept_fs = [f for f in kept if f.startswith("fs_")]
            kept_nonfs = [f for f in kept if not f.startswith("fs_")]
            overflow = len(kept) - MAX_PER_CLASS
            # Remove from fs files first
            if len(kept_fs) >= overflow:
                random.shuffle(kept_fs)
                kept_fs = kept_fs[:-overflow]
            else:
                kept_fs = []
                kept_nonfs = kept_nonfs[:MAX_PER_CLASS]
            kept = kept_nonfs + kept_fs
            removed += overflow

        total_kept += len(kept)
        total_removed += removed

        if dry_run:
            print(f"  Keep: {len(kept)}  |  Remove: {removed}")
        else:
            # Remove files not in kept
            for fname in wavs:
                if fname not in kept:
                    os.remove(os.path.join(cls_dir, fname))
            print(f"  ✅ {len(kept)} kept, {removed} removed")

    print(f"\n{'✅' if not dry_run else '🔍'} {'Done' if not dry_run else 'Dry run'}: "
          f"{total_kept} kept, {total_removed} removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate similar audio segments")
    parser.add_argument("--max-per-source", type=int, default=4,
                        help="Max segments from same source recording")
    parser.add_argument("--max-per-class", type=int, default=500,
                        help="Max files per class")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()

    MAX_PER_SOURCE = args.max_per_source
    MAX_PER_CLASS = args.max_per_class

    process(dry_run=args.dry_run)
