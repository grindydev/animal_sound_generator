"""
data_loader.py — Phase 1: Audio Data Loading & Spectrograms
=============================================================

WHAT YOU'LL BUILD:
  • Load .wav files from FSD50K dataset using torchaudio
  • Variable-length audio: pad in collate_fn (no cropping!)
  • get_dataloaders() with train/val/test split
  • get_transformations() → MelSpectrogram + AmplitudeToDB (called in train.py)

KEY CONCEPTS:
  • Mel-spectrogram = 2D image where Y=frequency, X=time, brightness=energy
  • transforms.MelSpectrogram converts waveform → spectrogram
  • AmplitudeToDB() converts to log scale (like dB in audio)
  • Variable-length: collate_fn pads waveforms so different lengths can batch

DATASET:
  • FSD50K: data/animal_audio/ (3001 clips)
  • Classes: Dog, Cat, Rooster, Frog, Crow, Insect, Hen, Noise

ARCHITECTURE DECISION — Option B: Pad waveforms, batch transform on GPU
  • DataLoader outputs raw padded waveforms [batch, 1, max_samples]
  • MelSpectrogram + AmplitudeToDB run on GPU in the training loop
  • Why? GPU batch transform is ~10x faster than CPU per-sample transform

  Flow:
    __getitem__ → raw waveform [1, variable_samples]
        ↓
    collate_fn → pad to [batch, 1, max_samples_in_batch]
        ↓
    train.py → train_transform(waveforms.to(device))
             → [batch, 1, n_mels, max_time_frames]
        ↓
    model → CNN → AdaptiveAvgPool → classifier
"""


import os
from typing import List, Tuple

import numpy as np
from torch import nn
from torch.utils.data import Dataset, random_split, DataLoader
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F_audio
import soundfile as sf

# Import smart_crop from scripts/
from smart_crop import smart_crop


# ── Config ────────────────────────────────────────────────────
# Path to data — relative to project root (parent of src/)
path_dataset = Path(__file__).resolve().parent.parent / 'data' / 'animal_audio'

# Target sample rate and max clip length
# 5 seconds @ 22050 Hz = 110250 samples → spectrogram ~550 time frames
# This keeps GPU memory manageable (spectrogram ~17 MB for batch=16)
TARGET_SR = 22050
MAX_SECONDS = 5
MAX_SAMPLES = TARGET_SR * MAX_SECONDS  # 110250
#     Mean: -30.8645
#     Std:  21.1952

class SimpleNormalize(nn.Module):
    def __init__(self, mean = -30.8645, std = 21.1952):
        super().__init__()
        self.mean = mean
        self.std = std
    
    def forward(self, x):
        return (x - self.mean) / self.std

# ── Collate Function ──────────────────────────────────────────
def collate_fn(batch):
    """
    Pad variable-length waveforms so they can be stacked into a batch tensor.
    
    WHY: DataLoader needs to return a single tensor per batch.
         But audio clips have different lengths → can't stack directly.
         Solution: find the longest clip in the batch, zero-pad all others.
    
    Input:  list of (waveform [1, variable_samples], label)
    Output: (padded_waveforms [batch, 1, max_samples], labels [batch])
    """
    waveforms, labels = zip(*batch)
    # Find the longest waveform in this batch
    max_len = max(w.shape[-1] for w in waveforms)
    # Create a zero-padded tensor for the whole batch
    padded = torch.zeros(len(waveforms), 1, max_len)
    for i, w in enumerate(waveforms):
        padded[i, :, :w.shape[-1]] = w
    return padded, torch.tensor(labels)


# ── Dataset ───────────────────────────────────────────────────
class AnimalSoundDataset(Dataset):
    """
    Loads raw .wav files from data/animal_audio/{Class}/ folders.
    
    Each __getitem__ call returns:
        waveform — [1, num_samples] raw audio tensor
        label    — int class index (0–7)
    
    Note: No transform applied here. Transforms (MelSpectrogram etc.)
    happen later in the training loop on GPU for speed.
    """

    CLASSES = ['Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen', 'Noise']

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.num_classes = len(self.CLASSES)
        # Map class name → index: {'Dog': 0, 'Cat': 1, ...}
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}
        # Scan all .wav files and store (filepath, label) pairs
        # Each sample: (filepath, label, crop_idx, num_crops)
        # Long files with multiple activity regions → multiple entries
        self.samples = self._make_dataset()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Load a single .wav file and return (waveform, label).
        
        torchaudio.load() returns:
            waveform — [channels, samples] tensor
            sample_rate — int (e.g. 44100)
        
        We only need the waveform. Sample rate is the same for all FSD50K
        files (44100 Hz) so we don't store it.
        """
        filepath, label, crop_idx, num_crops = self.samples[index]
        data, sr = sf.read(filepath, dtype='float32')
        waveform = torch.from_numpy(data)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, samples)
        else:
            waveform = waveform.T             # (channels, samples)

        # Resample to target SR if needed
        if sr != TARGET_SR:
            waveform = F_audio.resample(waveform, sr, TARGET_SR)

        # Smart crop: get all activity-region crops, pick the one for this index
        crops = smart_crop(waveform, crop_samples=MAX_SAMPLES, threshold_db=-30.0,
                           num_crops=num_crops)
        waveform = crops[min(crop_idx, len(crops) - 1)]

        return waveform, label

    def _make_dataset(self):
        """
        Scan the folder structure and collect all .wav files.
        
        Long files are expanded into multiple samples — one per activity region.
        A 20s clip with 3 detectable regions becomes 3 training samples.
        A 3s clip stays as 1 sample (padded later).
        
        Each entry: (filepath, label, crop_idx, num_crops)
        """
        samples = []
        for class_name in self.CLASSES:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for entry in os.scandir(class_dir):
                if entry.is_file() and entry.name.lower().endswith(('.wav', '.mp3')):
                    label = self.class_to_idx[class_name]
                    # Estimate crops from duration (sf.info is fast — header only)
                    info = sf.info(entry.path)
                    duration = info.duration
                    num_crops = max(1, int(duration) // MAX_SECONDS)
                    for i in range(num_crops):
                        samples.append((entry.path, label, i, num_crops))
        return samples


# ── Dataloaders ───────────────────────────────────────────────
def get_dataloaders(
    batch_size: int = 32,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    root_dir: str = path_dataset,
    num_workers: int = 1,
):
    """
    Create train / val / test DataLoaders with random split.
    
    The split is seeded (manual_seed=42) so it's reproducible across runs.
    
    train_fraction + val_fraction + (remainder) = 1.0
    test_fraction = 1.0 - train_fraction - val_fraction
    
    Returns: (train_loader, val_loader, test_loader, num_classes)
    """
    dataset = AnimalSoundDataset(root_dir)

    data_size = len(dataset)
    train_size = int(data_size * train_fraction)
    val_size = int(data_size * val_fraction)
    test_size = data_size - train_size - val_size  # remainder avoids rounding errors

    # Seed the split so every run gives the same train/val/test sets
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    # All 3 loaders use collate_fn to pad variable-length waveforms
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, dataset.num_classes


# ── Transformations ───────────────────────────────────────────
def get_transformations():
    """
    Returns (train_transform, eval_transform) as nn.Sequential modules.
    
    Call this in train.py and move to GPU:
        train_tfm, eval_tfm = get_transformations()
        train_tfm = train_tfm.to(device)
        eval_tfm  = eval_tfm.to(device)
    
    Then in the training loop:
        for waveforms, labels in train_loader:
            specs = train_tfm(waveforms.to(device))  # [B, 1, max_samples] → [B, 1, n_mels, max_time]
            output = model(specs)
    
    Pipeline:
        waveform [B, 1, samples]
            → MelSpectrogram  → [B, 1, n_mels, time_frames]
            → AmplitudeToDB   → [B, 1, n_mels, time_frames]  (log scale)
    
    Why AmplitudeToDB?
        Raw spectrogram values span 0 to ~100000. Human hearing is logarithmic
        (doubling loudness ≈ +10 dB). Log scale makes quiet sounds visible
        and keeps the value range manageable for the neural network.
    
    Train vs Eval transforms:
        Same for now. Later you'll add augmentations to train only:
            • FrequencyMasking — mask random frequency bands
            • TimeMasking — mask random time windows
            • (SpecAugment = FrequencyMasking + TimeMasking)
    """
    train_transform = nn.Sequential(
        T.MelSpectrogram(sample_rate=TARGET_SR, n_mels=64),   # waveform → mel spectrogram
        T.AmplitudeToDB(stype='power', top_db=80),            # linear → log scale (dB)
        SimpleNormalize(),
        # TODO: add SpecAugment here later
    )

    eval_transform = nn.Sequential(
        T.MelSpectrogram(sample_rate=TARGET_SR, n_mels=64),
        T.AmplitudeToDB(stype='power', top_db=80),
        SimpleNormalize(),
    )

    return train_transform, eval_transform


def get_mean_std(dataset: Dataset):
    train_tfm, _ = get_transformations()                                                                                                                                                                                            
    all_specs = []                                                                                                                                                                                                                  
    for waveforms, _ in dataset:                                                                                                                                                                                               
        specs = train_tfm(waveforms)                                                                                                                                                                                                
        all_specs.append(specs)                                                                                                                                                                                                     
    all_specs = torch.cat(all_specs, dim=0)                                                                                                                                                                                         
    print(f"Mean: {all_specs.mean():.4f}")                                                                                                                                                                                          
    print(f"Std:  {all_specs.std():.4f}")  

# ── Smoke Test ────────────────────────────────────────────────
# if __name__ == "__main__":

#     print("=" * 60)
#     print("data_loader.py smoke test")
#     print("=" * 60)

#     # 1) Test dataset
# dataset = AnimalSoundDataset(path_dataset)
#     print(f"\nDataset: {len(dataset)} samples, {dataset.num_classes} classes")
#     print(f"Classes: {dataset.CLASSES}")
#     print(f"Class map: {dataset.class_to_idx}")
#     get_mean_std(dataset)
#     Mean: -30.8645
#     Std:  21.1952

#     # Load one sample
#     waveform, label = dataset[0]
#     print(f"\nSample 0: waveform shape={waveform.shape}, label={label} ({dataset.CLASSES[label]})")
#     print(f"  Duration: {waveform.shape[-1] / 44100:.2f} seconds")

#     # 2) Test dataloaders
#     train_loader, val_loader, test_loader, num_classes = get_dataloaders(batch_size=8)

#     batch_waveforms, batch_labels = next(iter(train_loader))
#     print(f"\nTrain batch: waveforms={batch_waveforms.shape}, labels={batch_labels.shape}")
#     print(f"  Labels: {[dataset.CLASSES[l] for l in batch_labels]}")

#     # 3) Test transformations (on CPU for smoke test)
#     train_tfm, eval_tfm = get_transformations()
#     specs = train_tfm(batch_waveforms)
#     print(f"\nAfter MelSpectrogram + AmplitudeToDB: {specs.shape}")
#     print(f"  [batch, channels, n_mels, time_frames]")
#     print(f"  Value range: min={specs.min():.1f}, max={specs.max():.1f}, mean={specs.mean():.1f}")

#     print(f"\n✅ Data loader ready!")
#     print(f"  Train: {len(train_loader.dataset)} samples ({len(train_loader)} batches)")
#     print(f"  Val:   {len(val_loader.dataset)} samples ({len(val_loader)} batches)")
#     print(f"  Test:  {len(test_loader.dataset)} samples ({len(test_loader)} batches)")
