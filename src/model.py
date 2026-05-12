"""
model.py — Audio Classifier (SimpleAudioCNN).

Used by:
  - evaluate.py (classifier evaluation)
  - train.py (classifier training)
  - finetune_vae.py / src/vae/finetune.py (VAE class supervision loss)

Note: Encoder/decoder blocks have moved to src/vae/blocks.py.
      Autoencoder/VAE models have moved to src/vae/.
"""

import torch
import torch.nn as nn


class SimpleAudioCNNBlock(nn.Module):
    """Conv2d → BatchNorm2d → ReLU → MaxPool2d"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class SimpleAudioCNN(nn.Module):
    """Audio classifier: 4 conv blocks + AdaptiveAvgPool + classifier head."""
    def __init__(self, num_classes=8, dropout=0.3):
        super().__init__()
        self.conv_block1 = SimpleAudioCNNBlock(1, 32)
        self.conv_block2 = SimpleAudioCNNBlock(32, 64)
        self.conv_block3 = SimpleAudioCNNBlock(64, 128)
        self.conv_block4 = SimpleAudioCNNBlock(128, 256)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
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
