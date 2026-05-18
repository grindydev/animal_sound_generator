"""
clean_freesound.py — Split long Freesound files into 5s chunks around loud moments

Freesound files (fs_*.wav) are often 30-150s with mixed noise/silence.
This script finds the loud animal sound moments and extracts clean 5s segments.

Usage:
  python src/scripts/clean_freesound.py
  python src/scripts/clean_freesound.py --min-energy 0.01 --segment 5 --dry-run
"""
import os, sys, argparse
import torch
import torchaudio
import torch.nn.functional as F

DATA_DIR = "data/animal1000"
SEGMENT_SEC = 5
SAMPLE_RATE = 22050
MIN_RMS = 0.01       # minimum RMS for a segment to be kept
PEAK_SPACING_SEC = 3  # minimum seconds between peaks


def find_peaks(wav, sr, min_spacing_sec=3):
    """Find loud moments in audio using RMS energy."""
    # Compute RMS in windows
    win = int(sr * 0.1)  # 100ms windows
    hop = win // 2
    if wav.shape[-1] < win:
        return []

    # Pad and unfold
    n_frames = (wav.shape[-1] - win) // hop + 1
    if n_frames < 1:
        return []

    frames = wav.unfold(-1, win, hop)  # [1, win, n_frames]
    rms = (frames ** 2).mean(dim=1).sqrt().squeeze(0)  # [n_frames]

    # Threshold: peaks must be above mean + 0.5*std
    threshold = rms.mean() + 0.5 * rms.std()
    if threshold <= 0:
        return []

    # Find peaks above threshold, separated by min_spacing_sec
    min_spacing = int(min_spacing_sec * sr / hop)
    above = (rms > threshold).float()

    peaks = []
    last_peak = -min_spacing
    for i in range(1, len(above) - 1):
        if above[i] > above[i-1] and above[i] >= above[i+1]:
            sample_idx = i * hop + win // 2
            if sample_idx - last_peak >= min_spacing * hop:
                peaks.append(sample_idx)
                last_peak = sample_idx

    return peaks


def extract_segments(wav, peaks, segment_samples, sr):
    """Extract segments centered on peaks."""
    segments = []
    half = segment_samples // 2

    for peak in peaks:
        start = max(0, peak - half)
        end = min(wav.shape[-1], start + segment_samples)
        if end - start < segment_samples // 2:
            continue

        chunk = wav[:, start:end]
        # Pad if too short
        if chunk.shape[-1] < segment_samples:
            chunk = F.pad(chunk, (0, segment_samples - chunk.shape[-1]))

        seg_rms = (chunk ** 2).mean().sqrt().item()
        if seg_rms >= MIN_RMS:
            segments.append(chunk)

    return segments


def process_files(dry_run=False):
    """Process all fs_*.wav files in the dataset."""
    total_original = 0
    total_segments = 0

    for cls_name in sorted(os.listdir(DATA_DIR)):
        cls_dir = os.path.join(DATA_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        fs_files = [f for f in os.listdir(cls_dir) if f.startswith("fs_") and f.endswith(".wav")]
        if not fs_files:
            continue

        print(f"\n📦 {cls_name}: {len(fs_files)} fs_ files")

        for fname in fs_files:
            path = os.path.join(cls_dir, fname)
            try:
                wav, sr = torchaudio.load(path)
            except Exception as e:
                print(f"  ⚠️  {fname}: {e}")
                continue

            # Resample to 22050 if needed
            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                wav = resampler(wav)
                sr = SAMPLE_RATE

            # Convert to mono
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)

            segment_samples = SEGMENT_SEC * sr
            dur = wav.shape[-1] / sr

            # If file is already short enough, keep it
            if dur <= SEGMENT_SEC * 1.5:
                total_original += 1
                continue

            # Find peaks and extract segments
            peaks = find_peaks(wav, sr, PEAK_SPACING_SEC)
            if not peaks:
                # No clear peaks - split evenly
                n_segs = max(1, wav.shape[-1] // segment_samples)
                segments = []
                for i in range(n_segs):
                    start = i * segment_samples
                    end = min(wav.shape[-1], start + segment_samples)
                    chunk = wav[:, start:end]
                    if (chunk ** 2).mean().sqrt().item() >= MIN_RMS:
                        if chunk.shape[-1] < segment_samples:
                            chunk = F.pad(chunk, (0, segment_samples - chunk.shape[-1]))
                        segments.append(chunk)
            else:
                segments = extract_segments(wav, peaks, segment_samples, sr)

            if not segments:
                # Fallback: take first segment_samples
                chunk = wav[:, :segment_samples]
                if chunk.shape[-1] >= segment_samples // 2:
                    if chunk.shape[-1] < segment_samples:
                        chunk = F.pad(chunk, (0, segment_samples - chunk.shape[-1]))
                    segments = [chunk]

            if dry_run:
                print(f"  {fname[:40]}: {dur:.0f}s → {len(segments)} segments")
                total_original += 1
                total_segments += len(segments)
                continue

            # Save segments and remove original
            base = fname.replace(".wav", "")
            saved = 0
            for i, seg in enumerate(segments):
                out_path = os.path.join(cls_dir, f"{base}_{i+1}.wav")
                torchaudio.save(out_path, seg, SAMPLE_RATE)
                saved += 1

            os.remove(path)
            total_original += 1
            total_segments += saved

            if total_original % 20 == 0:
                print(f"  ... processed {total_original} files, {total_segments} segments")

    print(f"\n✅ Done: {total_original} files → {total_segments} segments")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split long Freesound files into clean segments")
    parser.add_argument("--segment", type=int, default=5, help="Segment length in seconds")
    parser.add_argument("--min-energy", type=float, default=0.01, help="Minimum RMS to keep")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no changes")
    args = parser.parse_args()

    SEGMENT_SEC = args.segment
    MIN_RMS = args.min_energy

    process_files(dry_run=args.dry_run)
