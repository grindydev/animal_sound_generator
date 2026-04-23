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


# ==================== 
"""
    SimpleEncoderBlock — one building block for the Audio Autoencoder (spectrogram version)
    
    ────────────────────────────────────────────────────────────────
    KEY CONCEPT FOR AUDIO (exactly like the image case you already know)
    ────────────────────────────────────────────────────────────────
    A mel-spectrogram is treated as a grayscale "image":
    
        Input shape: [batch, 1, 64, 552]
    
        • Height = 64  → Number of Mel-frequency bins
          (This is the vertical axis = frequency information)
          The Mel scale splits the frequency range in a way that mimics
          how human ears hear (more detail at low frequencies).
    
        • Width  = 552 → Number of time frames
          (This is the horizontal axis = time information)
          Comes from 5-second audio @ 22,050 Hz sample rate + typical
          hop_length ≈ 200 samples per frame.
          Result: each column represents roughly ~9 ms of audio.
    
    Analogy to regular images:
        Photo      → [B, 3, 128, 128]  (128 vertical pixels × 128 horizontal pixels)
        Spectrogram→ [B, 1,  64, 552]  ( 64 frequency "pixels" × 552 time "pixels")
    
    The CNN doesn't know it's audio — it just sees a 64×552 grayscale picture!
    ────────────────────────────────────────────────────────────────
    
    What this block actually does:
      Conv2d (3×3, padding=1) → BatchNorm2d → ReLU → MaxPool2d(2,2)
    
    Effect on dimensions (starting from 64×552):
      After 1 block → height/2, width/2  → 32×276
      After 2 blocks → 16×138
      After 3 blocks → 8×69
      After 4 blocks → 4×34   (this is what the autoencoder encoder uses)
    """
# ==================== 

class SimpleEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(SimpleEncoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
    
    def forward(self, x):
        x = self.block(x)
        return x
    
class SimpleDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleDecoderBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
    

    def forward(self, x):
        x = self.block(x)
        return x


class SimpleAudioAutoencoder(nn.Module):
    """
    All input spectrograms are [B, 1, 64, 552] (64 mel bins, 552 time frames for 5s @ 22050Hz).

    ENCODER:
      Input:                          [B, 1, 64, 552]
      EncoderBlock(1→32)   MaxPool   [B, 32, 32, 276]
      EncoderBlock(32→64)  MaxPool   [B, 64, 16, 138]
      EncoderBlock(64→128) MaxPool   [B, 128, 8, 69]
      EncoderBlock(128→256)MaxPool   [B, 256, 4, 34]
      Flatten:                        [B, 256*4*34] = [B, 34,816]

    LATENT:
      Linear(34,816 → latent_dim):   [B, 256]        ← compressed!

    DECODER:
      Linear(latent_dim → 34,816):   [B, 34,816]
      Reshape:                        [B, 256, 4, 34]
      DecoderBlock(256→128):          [B, 128, 8, 68]    ← 34*2=68, not 69!
      DecoderBlock(128→64):           [B, 64, 16, 136]   ← 68*2=136, not 138!
      DecoderBlock(64→32):            [B, 32, 32, 272]   ← 136*2=272, not 276!
      DecoderBlock(32→1):             [B, 1, 64, 544]    ← 272*2=544, not 552!

      ⚠️ 544 ≠ 552 — off by 8 time frames. Fix needed!
      # After DecoderBlock(32→1): [B, 1, 64, 544]
      output = nn.functional.interpolate(output, size=(64, 552), mode='bilinear')
    """
    def __init__(self, latent_dim=256):
        super(SimpleAudioAutoencoder, self).__init__()
        self.encode = nn.Sequential(
            SimpleEncoderBlock(1, 32),
            SimpleEncoderBlock(32, 64),
            SimpleEncoderBlock(64, 128),
            SimpleEncoderBlock(128, 256),
        )

        self.flat_dim = 256 * 4 * 34
        self.fc_encode = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)

        self.decode = nn.Sequential(
            SimpleDecoderBlock(256, 128),
            SimpleDecoderBlock(128, 64),
            SimpleDecoderBlock(64, 32),
            SimpleDecoderBlock(32, 1),
        )
        
    
    def forward(self, x):
        target_size = x.shape[2:]

        # Encode
        z = self.encode(x)
        z = z.flatten(start_dim=1)
        z = self.fc_encode(z)

        # Decode
        z = self.fc_decode(z)
        z = z.view(-1, 256, 4, 34)
        z = self.decode(z)

        # Fix size mismatch (544 -> 552)
        z = nn.functional.interpolate(z, size=target_size, mode='bilinear')
        return z









