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
  Input: [batch, 1, 64, variable_time]  (64 mel bins × variable time frames)
    → ConvBlock(1, 32)   → [batch, 32, 32, time/2]
    → ConvBlock(32, 64)  → [batch, 64, 16, time/4]
    → ConvBlock(64, 128) → [batch, 128, 8, time/8]
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
        self.conv_block4 = SimpleAudioCNNBlock(128, 256) 

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),    # [batch, 128, H, W] → [batch, 128, 1, 1]
            nn.Flatten(start_dim=1),          # [batch, 128, 1, 1] → [batch, 128]
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.classifier(x)
        return x


class SimpleEncoderBlock(nn.Module):
    """
    Encoder building block: Conv2d(stride=2) → BatchNorm → ReLU
    
    Uses stride=2 in Conv2d instead of MaxPool2d to downsample.
    
    WHY: MaxPool2d keeps only the MAX value in each 2×2 window →
         throws away 75% of values irreversibly. The decoder can never
         recover what MaxPool discarded, which hurts reconstruction.
    
         Conv2d with stride=2 lets the model LEARN what to keep.
         It's a learned downsampling instead of a hardcoded max rule.
    
    Effect on dimensions (starting from 64×552):
      After 1 block → 32×276    (stride=2 halves both dims)
      After 2 blocks → 16×138
      After 3 blocks → 8×69
      After 4 blocks → 4×35
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(SimpleEncoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.block(x)
        return x
    
class SimpleDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True):
        super(SimpleDecoderBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            nn.BatchNorm2d(num_features=out_channels),
             nn.ReLU() if activation else nn.Identity(),
        )
    

    def forward(self, x):
        x = self.block(x)
        return x


class SimpleAudioAutoencoder(nn.Module):
    """
    Autoencoder: spectrogram → encoder → latent vector → decoder → reconstructed spectrogram.

    ENCODER (stride=2 Conv2d — no MaxPool, so model learns what to keep):
      Input:                              [B, 1, 64, 552]    35,328 px
      EncoderBlock(1→32)     stride=2    [B, 32, 32, 276]   8,832 px
      EncoderBlock(32→64)    stride=2    [B, 64, 16, 138]   3,532 px
      EncoderBlock(64→128)   stride=2    [B, 128, 8, 69]    1,766 px
      EncoderBlock(128→256)  stride=2    [B, 256, 4, 35]    896 px
      Flatten:                            [B, 256×4×35] = [B, 35,840]

    BOTTLENECK (where information is compressed):
      fc_encode: Linear(35,840 → 1024)   [B, 1024]   ← 35× compression (was 138× with dim=256)
      fc_decode: Linear(1024 → 35,840)   [B, 35,840]
      Reshape:                            [B, 256, 4, 35]   tiny blurry thumbnail

    DECODER (ConvTranspose2d doubles dimensions each step):
      DecoderBlock(256→128):              [B, 128, 8, 70]
      DecoderBlock(128→64):               [B, 64, 16, 140]
      DecoderBlock(64→32):                [B, 32, 32, 280]
      DecoderBlock(32→1):                 [B, 1, 64, 560]
      Interpolate:                        [B, 1, 64, 552]   ← stretch 560→552 to match input
    """
    def __init__(self, latent_dim=1024):
        super(SimpleAudioAutoencoder, self).__init__()
        self.encode = nn.Sequential(
            SimpleEncoderBlock(1, 32),
            SimpleEncoderBlock(32, 64),
            SimpleEncoderBlock(64, 128),
            SimpleEncoderBlock(128, 256),
        )

        self.flat_dim = 256 * 4 * 35  # 35,840 (stride=2: 552→276→138→69→35)
        self.fc_encode = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)

        self.decode = nn.Sequential(
            SimpleDecoderBlock(256, 128),
            SimpleDecoderBlock(128, 64),
            SimpleDecoderBlock(64, 32),
            SimpleDecoderBlock(32, 1, activation=False),
        )
        
    
    def forward(self, x):
        target_size = x.shape[2:]

        # Encode
        z = self.encode(x)
        z = z.flatten(start_dim=1)
        z = self.fc_encode(z)

        # Decode
        z = self.fc_decode(z)
        z = z.view(-1, 256, 4, 35)
        z = self.decode(z)

        # Fix size mismatch (544 -> 552)
        z = nn.functional.interpolate(z, size=target_size, mode='bilinear')
        return z









