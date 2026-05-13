"""
model.py — Audio Classifier (ImprovedAudioCNN).

Used by:
  - evaluate.py (classifier evaluation)
  - train.py (classifier training)
  - finetune_vae.py / src/vae/finetune.py (VAE class supervision loss)

Note: Encoder/decoder blocks have moved to src/vae/blocks.py.
      Autoencoder/VAE models have moved to src/vae/.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
#  Residual Conv Block (kept for backward compatibility)
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
#  SimpleAudioCNN (original — kept for loading old checkpoints)
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
#  ImprovedAudioCNN — Residual + Attention + Less Aggressive Pooling
# ═══════════════════════════════════════════════════════════════

class ResClassifierBlock(nn.Module):
    """Residual block: Conv → GN → SiLU → Conv → GN, with 1×1 skip.
    Only pools on the FIRST block of each stage (stride=2 on skip too)."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.gn1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(min(32, out_ch), out_ch)

        self.use_skip = (stride != 1 or in_ch != out_ch)
        if self.use_skip:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride)

    def forward(self, x):
        residual = self.skip(x) if self.use_skip else x
        h = F.silu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return F.silu(h + residual)


class ImprovedAudioCNN(nn.Module):
    """
    Improved classifier with residual blocks and asymmetric pooling.

    Architecture:
      Stem: Conv 7×7 → 32ch (preserves spatial dims)
      Stage 1: 2× ResBlock(32→64), pool freq only (2×1)  → [B, 64, 32, 552]
      Stage 2: 2× ResBlock(64→128), pool 2×1             → [B, 128, 16, 552]
      Stage 3: 2× ResBlock(128→256), pool 2×2            → [B, 256, 8, 276]
      Stage 4: 2× ResBlock(256→256), pool 2×2            → [B, 256, 4, 138]
      Head: AdaptiveAvgPool → 3-layer MLP → 8 classes

    Key improvements over SimpleAudioCNN:
      - Residual connections → better gradient flow
      - GroupNorm → stable with small batches
      - Asymmetric pooling (freq first) → preserves time resolution longer
      - Deeper (10 conv layers vs 4)
      - ~4.5M params (vs 1.5M) — still fast
    """
    def __init__(self, num_classes: int = 8, dropout: float = 0.3):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, padding=3),
            nn.GroupNorm(min(32, 32), 32),
            nn.SiLU(),
        )

        # Stage 1: 32→64, pool freq only
        self.stage1_down = ResClassifierBlock(32, 64, stride=(2, 1))
        self.stage1_res = ResClassifierBlock(64, 64)

        # Stage 2: 64→128, pool freq only
        self.stage2_down = ResClassifierBlock(64, 128, stride=(2, 1))
        self.stage2_res = ResClassifierBlock(128, 128)

        # Stage 3: 128→256, pool both
        self.stage3_down = ResClassifierBlock(128, 256, stride=2)
        self.stage3_res = ResClassifierBlock(256, 256)

        # Stage 4: 256→256, pool both
        self.stage4_down = ResClassifierBlock(256, 256, stride=2)
        self.stage4_res = ResClassifierBlock(256, 256)

        # Classifier head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Stem
        x = self.stem(x)                          # [B, 32, 64, 552]

        # Stage 1
        x = self.stage1_down(x)                   # [B, 64, 32, 552]
        x = self.stage1_res(x)                    # [B, 64, 32, 552]

        # Stage 2
        x = self.stage2_down(x)                   # [B, 128, 16, 552]
        x = self.stage2_res(x)                    # [B, 128, 16, 552]

        # Stage 3
        x = self.stage3_down(x)                   # [B, 256, 8, 276]
        x = self.stage3_res(x)                    # [B, 256, 8, 276]

        # Stage 4
        x = self.stage4_down(x)                   # [B, 256, 4, 138]
        x = self.stage4_res(x)                    # [B, 256, 4, 138]

        # Head
        x = self.head(x)                          # [B, num_classes]

        return x
