"""
config.py — All hyperparameters for HiFi-GAN training.

Single source of truth. Imported by train.py, generator.py, and inference.py.
"""
from dataclasses import dataclass, field


@dataclass
class HiFiGANConfig:
    # ── Audio ──────────────────────────────────────────────
    sample_rate: int = 22050
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 200
    win_length: int = 1024
    f_min: float = 0.0
    f_max: float = 11025.0  # sample_rate / 2

    # ── Spectrogram normalization (from calc_norm_stats.py) ─
    norm_mean: float = -18.4903
    norm_std: float = 19.8031

    # ── Generator ─────────────────────────────────────────
    hidden_dim: int = 128                     # base channel count
    resblock_kernel_sizes: tuple = (3, 7, 11) # MRF kernel sizes
    resblock_dilation_sizes: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    upsample_rates: tuple = (5, 5, 4, 2)      # 5×5×4×2 = 200 = hop_length
    upsample_kernel_sizes: tuple = (10, 10, 8, 4)
    upsample_initial_channel: int = 256       # channels before first upsample

    # ── Discriminator ─────────────────────────────────────
    mpd_periods: tuple = (2, 3, 5, 7, 11)     # MPD periods
    mpd_conv_kernel: int = 5
    msd_scale_count: int = 3                   # MSD scales
    msd_norms: tuple = (
        (1, 16, 32, 64, 128, 256),              # scale 0: raw
        (1, 16, 32, 64, 128),                     # scale 1: ×2 pooled
        (1, 16, 32, 64),                          # scale 2: ×4 pooled
    )

    # ── Training ──────────────────────────────────────────
    segment_size: int = 16384                  # 0.74s — need enough context for full bark/meow
    batch_size: int = 8
    learning_rate: float = 2e-4
    lr_decay: float = 0.999                    # per-epoch decay
    adam_betas: tuple = (0.8, 0.99)
    num_epochs: int = 50
    num_workers: int = 4

    # ── Loss weights ─────────────────────────────────────
    lambda_mel: float = 45.0   # STRONG — forces generator to match input mel (prevents content drift)
    lambda_fm: float = 2.0     # feature matching — guides waveform realism
    lambda_adv: float = 1.0    # adversarial — small push for naturalness (must NOT overpower mel)

    # ── Smart crop (energy VAD) ──────────────────────────
    smart_crop_threshold_db: float = -30.0
    smart_crop_merge_gap: int = 4410            # 0.2s @ 22050Hz

    # ── Paths ────────────────────────────────────────────
    data_dir: str = "data/animal_audio"
    model_dir: str = "models"
    checkpoint_dir: str = "models/hifigan_checkpoints"

    # ── Misc ─────────────────────────────────────────────
    fp16: bool = False                          # MPS doesn't benefit from AMP
    log_interval: int = 50                      # batches between logs
    save_interval: int = 5                      # epochs between checkpoints
    eval_samples: int = 4                       # how many validation samples to convert


# Default instance for easy import
config = HiFiGANConfig()
