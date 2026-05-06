"""
discriminator.py — HiFi-GAN discriminators.

Two families:

MPD (Multi-Period Discriminator)
    Checks audio at different period intervals. Reshapes audio
    [B, 1, L] → [B×period, 1, L/period] so 2D convolutions
    see the waveform folded across its period.

    Why: catches repeating artifacts at specific frequencies.
         period=2  → high-frequency (11kHz)
         period=5  → mid-frequency (4.4kHz)
         period=11 → low-frequency (2kHz)

MSD (Multi-Scale Discriminator)
    Checks audio at different time resolutions.
    Average-pools the waveform 2×, 4× then applies 1D convs.

    Why: catches overall envelope shape at coarse scales,
         fine glitches at full scale.

Reference: Kumar et al. (2019) "MelGAN: Generative Adversarial Networks
for Conditional Waveform Synthesis" (MelGAN discriminator)
+ Kong et al. (2020) "HiFi-GAN" (MPD)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_padding, init_weights
from .config import config


# ══════════════════════════════════════════════════════════════
#  MPD — Period Discriminator
# ══════════════════════════════════════════════════════════════

class PeriodDiscriminator(nn.Module):
    """
    One period discriminator.
    
    Reshapes [B, 1, L] → [B×period, 1, L/period] then applies
    2D convolutions to judge realism at this specific period.
    
    The 2D conv kernel has height 1 (processes one "folded row" at a time)
    and width W (looks at W consecutive samples across the period fold).
    """

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period

        # Layers: slim channels to save GPU memory (~12M params total)
        channels = [1, 16, 64, 128]
        self.convs = nn.ModuleList()

        for i in range(len(channels) - 1):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels[i], channels[i + 1],
                        kernel_size=(5, kernel_size),
                        stride=(3, stride),
                        padding=(2, get_padding(kernel_size)),
                    ),
                    nn.LeakyReLU(0.1),
                )
            )

        # Final: 128 → 1 score per position
        self.post_conv = nn.Conv2d(128, 1, kernel_size=(3, 1), padding=(1, 0))

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [B, 1, L] waveform

        Returns:
            scores: [B, 1, H, W] raw discriminator scores
            features: list of intermediate features for FM loss
        """
        B, C, L = x.shape

        # Pad to multiple of period
        if L % self.period != 0:
            n_pad = self.period - (L % self.period)
            x = F.pad(x, (0, n_pad), mode="reflect")
            L = x.shape[-1]

        # Reshape: [B, 1, period, L/period] → [B×period, 1, 1, L/period]
        x = x.view(B, C, self.period, L // self.period)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B * self.period, 1, L // self.period).unsqueeze(2)  # add H=1 dim

        features = []
        for conv in self.convs:
            x = conv(x)  # [B*period, C_out, H, W]
            features.append(x)

        x = self.post_conv(x)
        features.append(x)

        # Back to original batch shape
        x = x.view(B, self.period, -1)
        return x, features


class MultiPeriodDiscriminator(nn.Module):
    """Collection of PeriodDiscriminators at different periods."""

    def __init__(self, periods: tuple = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Returns:
            all_scores: list of [score_tensor] per discriminator
            all_features: list of feature lists per discriminator
        """
        all_scores = []
        all_features = []
        for disc in self.discriminators:
            scores, feats = disc(x)
            all_scores.append(scores)
            all_features.append(feats)
        return all_scores, all_features


# ══════════════════════════════════════════════════════════════
#  MSD — Scale Discriminator
# ══════════════════════════════════════════════════════════════

class ScaleDiscriminator(nn.Module):
    """
    One scale discriminator. Processes 1D (Conv1d) at a given resolution.
    
    Uses spectral normalization for training stability.
    """

    def __init__(self, channel_list: tuple):
        super().__init__()
        self.convs = nn.ModuleList()

        for i in range(len(channel_list) - 1):
            self.convs.append(
                nn.Sequential(
                    nn.utils.spectral_norm(
                        nn.Conv1d(channel_list[i], channel_list[i + 1], kernel_size=15, padding=7)
                    ),
                    nn.LeakyReLU(0.1),
                )
            )

        # Strided convolutions for downsampling
        self.strided_convs = nn.ModuleList()
        current = channel_list[-1]
        for _ in range(3):
            next_ch = min(current * 2, 256)
            self.strided_convs.append(
                nn.Sequential(
                    nn.utils.spectral_norm(
                        nn.Conv1d(current, next_ch, kernel_size=15, stride=2, padding=7)
                    ),
                    nn.LeakyReLU(0.1),
                )
            )
            current = next_ch

        self.post_conv = nn.utils.spectral_norm(
            nn.Conv1d(current, 1, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [B, 1, L] waveform

        Returns:
            scores: [B, 1, T] raw scores
            features: list of intermediate features
        """
        features = []
        for conv in self.convs:
            x = conv(x)
            features.append(x)

        for strided in self.strided_convs:
            x = strided(x)
            features.append(x)

        x = self.post_conv(x)
        features.append(x)
        return x, features


class MultiScaleDiscriminator(nn.Module):
    """
    Collection of ScaleDiscriminators at different resolutions.
    
    Scales are achieved by average-pooling the input:
        scale 0: raw audio  [B, 1, L]
        scale 1: pool × 2   [B, 1, L/2]
        scale 2: pool × 4   [B, 1, L/4]
    """

    def __init__(self, channel_lists: tuple = None):
        super().__init__()
        if channel_lists is None:
            channel_lists = config.msd_norms

        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(cl) for cl in channel_lists
        ])
        self.pools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),  # scale 0 (no pool, but included for indexing)
            nn.AvgPool1d(4, 2, padding=2),  # scale 1: 2× downsample
            nn.AvgPool1d(4, 2, padding=2),  # scale 2: 4× downsample
        ])

    def forward(self, x: torch.Tensor) -> tuple:
        all_scores = []
        all_features = []

        for i, disc in enumerate(self.discriminators):
            if i == 0:
                x_i = x
            else:
                x_i = self.pools[i](x)
            scores, feats = disc(x_i)
            all_scores.append(scores)
            all_features.append(feats)

        return all_scores, all_features


# ══════════════════════════════════════════════════════════════
#  Combined Discriminator
# ══════════════════════════════════════════════════════════════

class Discriminator(nn.Module):
    """MPD + MSD combined."""

    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(self, x: torch.Tensor) -> tuple:
        mpd_scores, mpd_feats = self.mpd(x)
        msd_scores, msd_feats = self.msd(x)
        return mpd_scores + msd_scores, mpd_feats + msd_feats


# ══════════════════════════════════════════════════════════════
#  Quick test
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    D = Discriminator()
    dummy = torch.randn(2, 1, config.segment_size)
    scores, feats = D(dummy)

    print("Discriminator test:")
    print(f"  Input:  {dummy.shape}")
    print(f"  MPD discriminators: {len(D.mpd.discriminators)}")
    print(f"  MSD discriminators: {len(D.msd.discriminators)}")
    print(f"  Total score groups: {len(scores)}")
    print(f"  Feature groups:     {len(feats)}")
    for i, fg in enumerate(feats):
        layer_shapes = [f.shape for f in fg]
        print(f"    Group {i}: {len(fg)} layers, shapes: {layer_shapes[:3]}...")

    total = sum(p.numel() for p in D.parameters())
    print(f"  Params: {total:,}")
