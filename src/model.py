"""
model.py — Phase 2: Audio Classifier (2D CNN on Spectrograms)
==============================================================

WHAT YOU'LL BUILD:
  • SimpleAudioCNN — same architecture pattern as NSFW SimpleCNN
  • Input: mel-spectrogram [1, 128, 87] (like a grayscale image)
  • Conv blocks: Conv2d → BatchNorm → ReLU → MaxPool2d
  • AdaptiveAvgPool2d → Linear classifier
  • Output: 10 animal classes

KEY CONCEPTS:
  • Spectrograms are images — your NSFW CNN skills transfer directly!
  • 1 channel (grayscale) instead of 3 (RGB) for images
  • Same CNN block pattern: conv → bn → relu → pool

COURSE REFERENCE:
  • L1-M4 cnn/main.py — CNN blocks, AdaptiveAvgPool2d
  • L1-M4 nature_classification — full image classification pipeline

MODEL ARCHITECTURE:
  Input: [batch, 1, 128, 87]  (128 mel bins × 87 time frames)
    → ConvBlock(1, 32)   → [batch, 32, 64, 43]
    → ConvBlock(32, 64)  → [batch, 64, 32, 21]
    → ConvBlock(64, 128) → [batch, 128, 16, 10]
    → AdaptiveAvgPool2d  → [batch, 128, 1, 1]
    → Flatten            → [batch, 128]
    → Linear(128, 10)    → [batch, 10]  (10 animal classes)
"""
