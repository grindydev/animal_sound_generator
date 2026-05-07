"""
generator.py — HiFi-GAN Generator: mel spectrogram → audio waveform.

Architecture (Multi-Receptive Field Fusion):
    mel [B, 64, T]
         ↓
    Conv1D pre-net  (in=64, hidden, kernel=7)
         ↓
    ┌───── MRF Block × 4 ─────┐
    │  Upsample [5,5,4,2]      │  ← 5×5×4×2 = 200 = hop_length
    │  ┌── ResBlocks(3,7,11) ─┐│
    │  │ kernel=3, dil=[1,3,5] ││  ← 3 kernels → summed
    │  │ kernel=7, dil=[1,3,5] ││     captures fine + coarse patterns
    │  │ kernel=11,dil=[1,3,5] ││     simultaneously
    │  └───────────────────────┘│
    └──────────────────────────┘
         ↓
    Conv1D post-net (kernel=7, tanh)
         ↓
    waveform [B, 1, T × 200]

Reference: Kong, Kim, Bae (2020) "HiFi-GAN: Generative Adversarial Networks
for Efficient and High Fidelity Speech Synthesis"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_padding, init_weights
from .config import config


# ══════════════════════════════════════════════════════════════
#  ResBlock — dilated 1D convolution inside MRF
# ══════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """
    One residual block with dilated convolutions.
    
    Inside each MRF kernel size group, we stack N dilated conv layers.
    Each ResBlock: x → Conv(d,1) → LeakyReLU → Conv(d,2) → + x
    """

    def __init__(self, channels: int, kernel_size: int, dilations: tuple):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            padding = get_padding(kernel_size, d)
            self.convs.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(
                        channels, channels, kernel_size,
                        dilation=d, padding=padding,
                    ),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(
                        channels, channels, kernel_size,
                        dilation=1, padding=get_padding(kernel_size, 1),
                    ),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            residual = x
            x = conv(x)
            x = x + residual
        return x


# ══════════════════════════════════════════════════════════════
#  MRFBlock — one upsample stage with parallel ResBlocks
# ══════════════════════════════════════════════════════════════

class MRFBlock(nn.Module):
    """
    Multi-Receptive Field Fusion block.
    
    Upsamples input, halves channels, then applies the SAME signal through
    3 parallel ResBlock paths with different kernel sizes (3, 7, 11).
    Output = sum of all 3 paths.
    
    Why 3 kernel sizes?
        kernel=3  → catches fast transients (attack of a bark)
        kernel=7  → catches medium patterns (vocal timbre)
        kernel=11 → catches slow modulation (pitch contour)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple = (3, 7, 11),
        dilations: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        upsample_rate: int = 2,
        upsample_kernel_size: int = 4,
    ):
        super().__init__()

        # ConvTranspose1d: upsample AND halve channels
        self.upsample = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_rate,
            padding=(upsample_kernel_size - upsample_rate) // 2,
        )

        # Parallel ResBlock paths (3 kernel sizes)
        self.resblocks = nn.ModuleList()
        for ks, ds in zip(kernel_sizes, dilations):
            self.resblocks.append(ResBlock(out_channels, ks, ds))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        outputs = []
        for resblock in self.resblocks:
            outputs.append(resblock(x))
        return sum(outputs)  # merge all receptive field paths


# ══════════════════════════════════════════════════════════════
#  HiFiGAN Generator — full model
# ══════════════════════════════════════════════════════════════

class HiFiGANGenerator(nn.Module):
    """
    HiFi-GAN Generator: mel spectrogram → raw audio waveform.

    Args:
        h: config object with all hyperparameters
    """

    def __init__(self, h=None):
        super().__init__()
        if h is None:
            h = config

        self.h = h

        # Pre-net: expand mel channels
        self.pre_conv = nn.Conv1d(
            h.n_mels, h.upsample_initial_channel,
            kernel_size=7, padding=3,
        )

        # MRF blocks — each upsamples by a factor
        in_channels = h.upsample_initial_channel
        self.mrf_blocks = nn.ModuleList()

        for rate, ksize in zip(h.upsample_rates, h.upsample_kernel_sizes):
            out_channels = in_channels // 2
            self.mrf_blocks.append(
                MRFBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_sizes=h.resblock_kernel_sizes,
                    dilations=h.resblock_dilation_sizes,
                    upsample_rate=rate,
                    upsample_kernel_size=ksize,
                )
            )
            in_channels = out_channels

        # Post-net: compress to 1 channel → audio
        self.post_conv = nn.Conv1d(in_channels, 1, kernel_size=7, padding=3)

        self.apply(init_weights)
        # Override post_conv to zero — generator starts silent, learns amplitude gradually
        nn.init.constant_(self.post_conv.weight, 0.0)
        nn.init.constant_(self.post_conv.bias, 0.0)

    def forward(self, mel: torch.Tensor, target_length: int = None) -> torch.Tensor:
        """
        Args:
            mel: [B, n_mels, T] mel spectrogram (normalized dB)
            target_length: if given, trim/pad output to this many samples

        Returns:
            waveform: [B, 1, T × hop_length] audio waveform
        """
        # Ensure correct shape
        if mel.dim() == 4:
            mel = mel.squeeze(1)  # [B, 1, 64, T] → [B, 64, T]

        x = self.pre_conv(mel)
        for mrf in self.mrf_blocks:
            x = mrf(x)
        x = self.post_conv(x)
        # No tanh — time-domain loss keeps output bounded

        if target_length is not None and x.shape[-1] != target_length:
            if x.shape[-1] > target_length:
                x = x[..., :target_length]
            else:
                x = F.pad(x, (0, target_length - x.shape[-1]))

        return x

    def remove_weight_norm(self):
        """Remove weight norm for inference (if applied during training)."""
        pass  # We don't use weight norm in this implementation


# ══════════════════════════════════════════════════════════════
#  Quick test
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    h = config
    model = HiFiGANGenerator(h)
    dummy = torch.randn(2, h.n_mels, 160)  # ~1.49s of mel frames
    out = model(dummy)
    print(f"Generator test:")
    print(f"  Input:  {dummy.shape}  (mel)")
    print(f"  Output: {out.shape}   (waveform)")
    print(f"  Expected samples: {160 * h.hop_length}")
    # Params
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total:,}")
