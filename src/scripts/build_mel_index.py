"""
build_mel_index.py — Precompute mel spectrograms for all training samples.

Creates a cached index of mel spectrograms for retrieval-based generation.
This is used by v14 retrieval mode: pick a real mel → add noise → refine → audio.

Usage:
    python src/scripts/build_mel_index.py
"""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from src.diffusion.config import config as cfg

# ── Audio loading ─────────────────────────────────────────────
def _load_audio(path):
    try:
        import torchaudio
        return torchaudio.load(path)
    except Exception:
        import soundfile as sf
        data, sr = sf.read(path, dtype='float32')
        wav = torch.from_numpy(data)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.transpose(0, 1)
        return wav, sr

# ── Mel computation (same as training) ────────────────────────
_mel_tfm_cache = {}
_db_tfm_cache = {}

def compute_mel(audio):
    d = audio.device
    if d not in _mel_tfm_cache:
        _mel_tfm_cache[d] = MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft,
            hop_length=cfg.hop_length, win_length=cfg.n_fft,
            n_mels=cfg.n_mels, f_min=cfg.f_min, f_max=cfg.f_max, power=2,
        ).to(d)
        _db_tfm_cache[d] = AmplitudeToDB(stype='power', top_db=80).to(d)
    
    spec = _mel_tfm_cache[d](audio.squeeze(0))
    mel = (_db_tfm_cache[d](spec) - (-18.4903)) / 19.8031
    return mel  # [n_mels, time_frames]

# ── Smart crop ────────────────────────────────────────────────
def smart_crop(wav, crop_samples, threshold_db=-30.0, num_crops=1, merge_gap_samples=4410):
    """Extract energy-rich regions from audio."""
    energy = (wav ** 2).mean(dim=0)
    energy_db = 10 * torch.log10(energy + 1e-10)
    
    threshold = energy_db.max() + threshold_db
    mask = energy_db > threshold
    
    # Find contiguous regions
    regions = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start > crop_samples // 2:
                regions.append((start, i))
            start = None
    if start is not None and len(mask) - start > crop_samples // 2:
        regions.append((start, len(mask)))
    
    # Merge nearby regions
    if not regions:
        regions = [(0, len(wav[0]))]
    
    merged = [regions[0]]
    for start, end in regions[1:]:
        if start - merged[-1][1] < merge_gap_samples:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    
    # Pick best regions
    best = sorted(merged, key=lambda r: energy_db[r[0]:r[1]].mean(), reverse=True)[:num_crops]
    
    crops = []
    for start, end in best:
        center = (start + end) // 2
        crop_start = max(0, center - crop_samples // 2)
        crop_end = min(len(wav[0]), crop_start + crop_samples)
        crops.append(wav[:, crop_start:crop_end])
    
    return crops

# ── Main ──────────────────────────────────────────────────────
def main():
    data_dir = "data/esc50"
    output_dir = "data/mel_index"
    os.makedirs(output_dir, exist_ok=True)
    
    CLASSES = ['Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen']
    
    total = 0
    for cls in CLASSES:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"  ⚠️  {cls}: directory not found")
            continue
        
        wavs = [f for f in os.listdir(cls_dir) if f.endswith('.wav')]
        crop_samples = cfg.segment_frames * cfg.hop_length
        
        mels = []
        for wav_file in wavs:
            path = os.path.join(cls_dir, wav_file)
            wav, sr = _load_audio(path)
            if wav is None:
                continue
            
            if sr != cfg.sample_rate:
                import torchaudio
                wav = torchaudio.transforms.Resample(sr, cfg.sample_rate)(wav)
            
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            if wav.shape[-1] <= crop_samples:
                pad = torch.zeros(1, crop_samples - wav.shape[-1])
                wav = torch.cat([wav, pad], dim=-1)
            else:
                crops = smart_crop(wav, crop_samples=crop_samples, threshold_db=-30.0)
                wav = crops[0]
            
            mel = compute_mel(wav)  # [n_mels, T]
            
            # Trim/pad
            T = mel.shape[-1]
            if T > cfg.segment_frames:
                mel = mel[..., :cfg.segment_frames]
            elif T < cfg.segment_frames:
                mel = torch.nn.functional.pad(mel, (0, cfg.segment_frames - T))
            
            mels.append(mel)
        
        if mels:
            mels_tensor = torch.stack(mels)  # [N, n_mels, T]
            output_path = os.path.join(output_dir, f"{cls}.pt")
            torch.save(mels_tensor, output_path)
            print(f"  ✅ {cls}: {len(mels)} mels → {output_path} ({mels_tensor.shape})")
            total += len(mels)
        else:
            print(f"  ❌ {cls}: no valid files")
    
    print(f"\n🎉 Total: {total} mels indexed in {output_dir}/")

if __name__ == "__main__":
    main()
