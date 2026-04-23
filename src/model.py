"""
model.py — Phase 2: Audio Classifier (2D CNN on Spectrograms)
==============================================================

WHAT YOU'LL BUILD:
  • SimpleAudioCNN — same architecture pattern as NSFW SimpleCNN
  • Input: mel-spectrogram [batch, 1, 128, variable_time] (like a grayscale image)
  • Conv blocks: Conv2d → BatchNorm → ReLU → MaxPool2d
  • AdaptiveAvgPool2d → Linear classifier (handles variable-length!)
  • Output: 8 animal classes

KEY CONCEPTS:
  • Spectrograms are images — your NSFW CNN skills transfer directly!
  • 1 channel (grayscale) instead of 3 (RGB) for images
  • AdaptiveAvgPool2d((1,1)) squashes variable time → fixed size
    This is what makes variable-length audio work — the model accepts any duration.

COURSE REFERENCE:
  • L1-M4 cnn/main.py — CNN blocks, AdaptiveAvgPool2d
  • L1-M4 nature_classification — full image classification pipeline

MODEL ARCHITECTURE:
  Input: [batch, 1, 128, variable_time]  (128 mel bins × variable time frames)
    → ConvBlock(1, 32)   → [batch, 32, 64, time/2]
    → ConvBlock(32, 64)  → [batch, 64, 32, time/4]
    → ConvBlock(64, 128) → [batch, 128, 16, time/8]
    → AdaptiveAvgPool2d  → [batch, 128, 1, 1]    ← squashes to fixed size
    → Flatten            → [batch, 128]
    → Linear(128, 128)   → ReLU → Dropout
    → Linear(128, 8)     → [batch, 8]  (8 animal classes)
"""

import torch
import torch.nn as nn


class SimpleAudioCNNBlock(nn.Module):
    """
    Same pattern as NSFW CNNBlock:
      Conv2d → BatchNorm2d → ReLU → MaxPool2d
    
    padding=1 keeps spatial dimensions same before pooling
    (kernel_size=3, padding=1 → output same size → then MaxPool halves it)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(SimpleAudioCNNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class SimpleAudioCNN(nn.Module):
    """
    Audio classifier — 3 conv blocks + AdaptiveAvgPool + classifier head.
    
    AdaptiveAvgPool2d((1,1)) is the key to variable-length audio:
    Whatever the time dimension is after convolutions (could be 5 or 500),
    it averages across all time frames → always produces [batch, channels, 1, 1].
    """
    def __init__(self, num_classes=8, dropout=0.3):
        super(SimpleAudioCNN, self).__init__()

        self.conv_block1 = SimpleAudioCNNBlock(1, 32)    # 1 channel (mel spec) → 32
        self.conv_block2 = SimpleAudioCNNBlock(32, 64)    # 32 → 64
        self.conv_block3 = SimpleAudioCNNBlock(64, 128)   # 64 → 128

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),    # [batch, 128, H, W] → [batch, 128, 1, 1]
            nn.Flatten(start_dim=1),          # [batch, 128, 1, 1] → [batch, 128]
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x
