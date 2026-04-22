"""
data_loader.py — Phase 1: Audio Data Loading & Spectrograms
=============================================================

WHAT YOU'LL BUILD:
  • Load .wav files from ESC-50 dataset using torchaudio
  • Compute mel-spectrograms (audio as 2D images)
  • Crop clips to 2 seconds (loudest part) for faster training
  • Audio augmentation: time shift, noise, SpecAugment
  • AudioDataset class: __getitem__ returns (spectrogram, label)
  • get_dataloaders() with train/val/test split

KEY CONCEPTS:
  • Mel-spectrogram = 2D image where Y=frequency, X=time, brightness=energy
  • torchaudio.transforms.MelSpectrogram converts waveform → spectrogram
  • AmplitudeToDB() converts to log scale (like dB in audio)
  • Griffin-Lim algorithm converts spectrogram back to audio (for listening)

COURSE REFERENCE:
  • L1-M3 data_management/main.py — Custom Dataset, transforms, augmentation
  • L1-M3 data_pipeline — train/val/test split with different transforms

DATASET:
  • ESC-50: data/esc50_audio/ (~600 animal clips)
  • Classes: dog, rooster, pig, cow, frog, cat, hen, insects, sheep, crow
  • Each clip: 5 seconds → crop to 2 seconds

AUDIO AUGMENTATION (new, not in course):
  • Time shift: shift audio ±0.5 seconds
  • Add noise: background noise at low SNR
  • Time stretch: speed up/slow down (0.8x-1.2x)
  • SpecAugment: mask random frequency bands + time windows
"""
