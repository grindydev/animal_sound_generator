"""
smart_crop.py — Energy-based Smart Cropping for Audio
======================================================

Problem: Many FSD50K clips are 10-30s but the animal sound is only 1-3s.
         Random cropping often lands on silence between sounds.
         This wastes training data and confuses the model.

Solution: Energy-based Voice Activity Detection (VAD)
          1. Compute RMS energy per short frame
          2. Find frames above a threshold (relative to peak)
          3. Group into "activity regions"
          4. Crop windows centered on the loudest regions

Usage in data_loader.py:
    from smart_crop import smart_crop

    # In __getitem__:
    crops = smart_crop(waveform, crop_samples=5*22050, num_crops=1)
    return crops[0], label

Functions:
    • compute_rms_energy() — RMS per frame (low-level)
    • find_activity_regions() — detect sound bursts
    • smart_crop() — the main function you'll call

Run tests:
    python src/smart_crop.py
"""

import torch

# Must match data_loader.py TARGET_SR
TARGET_SR = 22050


# ── Energy Computation ────────────────────────────────────────

def compute_rms_energy(
    waveform: torch.Tensor,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """
    Compute RMS energy per frame for a waveform.

    Args:
        waveform: [1, samples] or [samples] tensor
        frame_length: window size (2048 ≈ 93ms @ 22050Hz)
        hop_length: step between frames (512 ≈ 23ms @ 22050Hz)

    Returns:
        energy: [num_frames] tensor of RMS values
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Pad so last frame is complete
    padding = frame_length - waveform.shape[-1] % hop_length
    if padding < frame_length:
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    # Unfold into overlapping frames → [num_frames, frame_length]
    frames = waveform.unfold(dimension=-1, size=frame_length, step=hop_length)
    frames = frames.squeeze(0)

    return (frames ** 2).mean(dim=-1).sqrt()


# ── Activity Detection ────────────────────────────────────────

def find_activity_regions(
    waveform: torch.Tensor,
    threshold_db: float = -30.0,
    frame_length: int = 2048,
    hop_length: int = 512,
    min_region_samples: int = 0,
    merge_gap_samples: int = 0,
) -> list:
    """
    Find regions of sound activity using energy-based VAD.

    Args:
        waveform: [1, samples] or [samples]
        threshold_db: dB below peak to count as "active"
                       -20 = strict (loud only)
                       -30 = default
                       -40 = permissive (quiet sounds too)
        frame_length: analysis window in samples
        hop_length: step between windows
        min_region_samples: discard regions shorter than this
        merge_gap_samples: merge regions closer than this

    Returns:
        List of (start_sample, end_sample) tuples
    """
    energy = compute_rms_energy(waveform, frame_length, hop_length)

    # All silence → no regions
    if energy.max().item() < 1e-8:
        return []

    # dB relative to peak
    peak = energy.max().clamp(min=1e-10)
    energy_db = 20 * torch.log10(energy.clamp(min=1e-10) / peak)

    active = energy_db > threshold_db

    # Group consecutive active frames into regions
    regions = []
    in_region = False
    start = 0

    for i in range(len(active)):
        if active[i] and not in_region:
            in_region = True
            start = i * hop_length
        elif not active[i] and in_region:
            end = i * hop_length
            if end - start >= min_region_samples:
                regions.append((start, end))
            in_region = False

    if in_region:
        end = waveform.shape[-1] if waveform.dim() == 1 else waveform.shape[-1]
        if end - start >= min_region_samples:
            regions.append((start, end))

    # Merge nearby regions
    if merge_gap_samples > 0 and len(regions) > 1:
        merged = [regions[0]]
        for s, e in regions[1:]:
            ps, pe = merged[-1]
            if s - pe <= merge_gap_samples:
                merged[-1] = (ps, e)
            else:
                merged.append((s, e))
        regions = merged

    return regions


# ── Smart Crop ────────────────────────────────────────────────

def smart_crop(
    waveform: torch.Tensor,
    crop_samples: int,
    threshold_db: float = -30.0,
    num_crops: int = 1,
    merge_gap_samples: int = 4410,
) -> list:
    """
    Crop waveform to segments centered on the loudest activity regions.

    Args:
        waveform: [1, samples] raw audio
        crop_samples: target length (e.g. 5 * 22050 = 110250)
        threshold_db: dB below peak to count as activity
        num_crops: how many crops to return (1 = best, 2+ = augmentation)
        merge_gap_samples: merge regions closer than this (default 0.2s)

    Returns:
        List of [1, crop_samples] tensors
    """
    n_samples = waveform.shape[-1]

    # Short clip → just pad
    if n_samples <= crop_samples:
        pad = torch.zeros(1, crop_samples - n_samples)
        return [torch.cat([waveform, pad], dim=-1)]

    # Find activity regions
    regions = find_activity_regions(
        waveform,
        threshold_db=threshold_db,
        merge_gap_samples=merge_gap_samples,
    )

    # No activity → center crop fallback
    if not regions:
        start = (n_samples - crop_samples) // 2
        return [waveform[:, start:start + crop_samples]]

    # Rank regions by peak energy (loudest first)
    energy = compute_rms_energy(waveform)
    hop = 512

    def region_peak(region):
        s, e = region
        fs = s // hop
        fe = min(e // hop, len(energy) - 1)
        if fs >= fe:
            fe = fs + 1
        return energy[fs:fe].max().item()

    ranked = sorted(regions, key=region_peak, reverse=True)

    # Extract crops centered on each region
    crops = []
    used_centers = set()

    for s, e in ranked:
        if len(crops) >= num_crops:
            break

        center = (s + e) // 2

        # Avoid duplicate positions (round to 0.5s buckets)
        bucket = center // (TARGET_SR // 2)
        if bucket in used_centers:
            continue
        used_centers.add(bucket)

        crop_start = max(0, min(center - crop_samples // 2, n_samples - crop_samples))
        crops.append(waveform[:, crop_start:crop_start + crop_samples])

    # Not enough from regions → fill with random crops
    while len(crops) < num_crops:
        start = torch.randint(0, n_samples - crop_samples + 1, (1,)).item()
        crops.append(waveform[:, start:start + crop_samples])

    return crops


# ══════════════════════════════════════════════════════════════
#  TESTS — run with: python src/smart_crop.py
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import os
    import numpy as np

    PASSED = 0
    FAILED = 0

    def assert_test(condition, name, detail=""):
        global PASSED, FAILED
        if condition:
            PASSED += 1
            print(f"  ✅ {name}")
        else:
            FAILED += 1
            print(f"  ❌ {name} — {detail}")

    # ── Helpers ──
    def make_burst(sr=22050, duration=10.0, burst_times=[2.0, 5.0, 8.0],
                   burst_dur=0.3, noise_amp=0.01, burst_amp=0.8):
        n = int(sr * duration)
        wf = np.random.randn(n) * noise_amp
        for t in burst_times:
            s = int(t * sr)
            e = min(s + int(burst_dur * sr), n)
            wf[s:e] += np.random.randn(e - s) * burst_amp
        return torch.tensor(wf, dtype=torch.float32).unsqueeze(0)

    def make_sine(freq=440, sr=22050, duration=3.0, amp=0.5):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        return torch.tensor(np.sin(2 * np.pi * freq * t) * amp, dtype=torch.float32).unsqueeze(0)

    sr = TARGET_SR
    crop_5s = 5 * sr

    # ── compute_rms_energy ──
    print("\n=== compute_rms_energy ===")

    e = compute_rms_energy(make_sine(duration=2.0))
    assert_test(e.dim() == 1, "Returns 1D tensor")
    assert_test(len(e) > 0, "Has frames")

    assert_test(
        compute_rms_energy(torch.zeros(1, sr)).max().item() < 1e-6,
        "Silence → energy ≈ 0",
    )
    assert_test(
        compute_rms_energy(torch.randn(1, sr) * 0.5).mean().item() > 0.1,
        "Noise → energy > 0.1",
    )
    assert_test(
        compute_rms_energy(torch.randn(sr)).dim() == 1,
        "Handles 1D input",
    )

    # ── find_activity_regions ──
    print("\n=== find_activity_regions ===")

    wf = make_burst(duration=10.0, burst_times=[2.0, 5.0, 8.0])
    regions = find_activity_regions(wf, threshold_db=-30.0)
    assert_test(2 <= len(regions) <= 4, f"Detects 2-4 regions (got {len(regions)})")
    for i, (s, e) in enumerate(regions):
        print(f"    Region {i}: {s/sr:.2f}s – {e/sr:.2f}s")

    assert_test(
        len(find_activity_regions(torch.zeros(1, sr * 5))) == 0,
        "Silence → no regions",
    )

    r_no = find_activity_regions(make_burst(burst_times=[2.0, 2.3]), merge_gap_samples=0)
    r_yes = find_activity_regions(make_burst(burst_times=[2.0, 2.3]), merge_gap_samples=int(0.5 * sr))
    assert_test(len(r_yes) <= len(r_no), f"Merge works ({len(r_yes)} ≤ {len(r_no)})")

    # Stricter threshold → fewer regions (or same), never more
    wf2 = make_burst(duration=5.0, burst_times=[1.0, 3.5], burst_dur=0.2, noise_amp=0.005)
    r_strict = find_activity_regions(wf2, threshold_db=-5.0, merge_gap_samples=0)
    r_loose = find_activity_regions(wf2, threshold_db=-40.0, merge_gap_samples=0)
    assert_test(len(r_loose) >= len(r_strict), f"More permissive → more regions ({len(r_loose)} >= {len(r_strict)})")

    # ── smart_crop ──
    print("\n=== smart_crop ===")

    # Short clip (2s) → pad to 5s
    short = make_sine(duration=2.0)
    crops = smart_crop(short, crop_5s)
    assert_test(len(crops) == 1, "Short clip: returns 1 crop")
    assert_test(crops[0].shape == (1, crop_5s), f"Short clip: padded to {crop_5s}")
    assert_test(
        crops[0][:, :2*sr].abs().mean() > crops[0][:, 3*sr:].abs().mean() * 10,
        "Short clip: signal in first 2s, zeros after",
    )

    # Long clip with burst at 7s → crop should include it
    wf_long = make_burst(duration=10.0, burst_times=[7.0], burst_dur=0.5, burst_amp=0.9)
    crops = smart_crop(wf_long, crop_5s, threshold_db=-30.0)
    assert_test(len(crops) == 1, "Long clip: returns 1 crop")
    assert_test(crops[0].shape == (1, crop_5s), "Long clip: correct shape")
    crop_e = compute_rms_energy(crops[0]).max()
    full_e = compute_rms_energy(wf_long).max()
    assert_test(
        crop_e > full_e * 0.5,
        f"Long clip: crop contains burst ({crop_e:.3f} vs {full_e:.3f})",
    )

    # Multiple crops
    wf_multi = make_burst(duration=15.0, burst_times=[2.0, 7.0, 12.0])
    crops = smart_crop(wf_multi, 3 * sr, num_crops=3)
    assert_test(len(crops) == 3, f"Multi crop: returns 3 (got {len(crops)})")
    for i, c in enumerate(crops):
        assert_test(c.shape == (1, 3 * sr), f"Multi crop {i}: correct shape")

    # Silence → center crop fallback
    crops = smart_crop(torch.zeros(1, 10 * sr), crop_5s)
    assert_test(len(crops) == 1, "Silence: returns 1 crop (fallback)")
    assert_test(crops[0].abs().max() < 1e-6, "Silence: crop is silent")

    # Exact length
    wf_exact = make_sine(duration=5.0)
    crops = smart_crop(wf_exact, crop_5s)
    assert_test(crops[0].shape[-1] == crop_5s, "Exact length: no change")

    # ── Real audio ──
    print("\n=== Real audio ===")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'animal_audio')
    if os.path.isdir(data_dir):
        import soundfile as sf
        import random as _rnd
        _rnd.seed(42)
        for cls in ['Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen', 'Noise']:
            cls_dir = os.path.join(data_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            wavs = [f for f in os.listdir(cls_dir) if f.endswith('.wav')]
            # Pick a long file
            test_file = None
            for f in sorted(wavs):
                info = sf.info(os.path.join(cls_dir, f))
                if info.duration > 7.0:
                    test_file = f
                    break
            if not test_file:
                print(f"  ⚠️ {cls}: no file > 7s, skip")
                continue
            data, file_sr = sf.read(os.path.join(cls_dir, test_file), dtype='float32')
            wf_real = torch.from_numpy(data).unsqueeze(0)
            crops = smart_crop(wf_real, crop_5s, threshold_db=-30.0)
            assert_test(len(crops) >= 1, f"{cls}: got {len(crops)} crop(s)")
            assert_test(crops[0].shape == (1, crop_5s), f"{cls}: correct shape")
            # Smart crop vs random crop energy
            crop_e = crops[0].abs().mean().item()
            rand_es = []
            n = wf_real.shape[-1]
            for _ in range(5):
                s = _rnd.randint(0, max(0, n - crop_5s))
                rand_es.append(wf_real[:, s:s+crop_5s].abs().mean().item())
            ratio = crop_e / max(sum(rand_es)/len(rand_es), 1e-10)
            assert_test(ratio >= 0.6, f"{cls}: energy ratio {ratio:.2f}x (≥ 0.6x)")
    else:
        print("  ⚠️ data/animal_audio not found, skipping real audio tests")

    # ── Summary ──
    print(f"\n{'='*50}")
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print(f"{'='*50}")
    if FAILED > 0:
        sys.exit(1)
