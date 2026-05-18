"""
GAN Configuration — v16
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class GANConfig:
    # ── Model ──
    latent_dim: int = 256
    cond_dim: int = 256          # conditioning vector after MLP
    num_classes: int = 7
    gen_base_ch: int = 128       # starting channels after dense
    gen_max_ch: int = 256
    disc_base_ch: int = 48
    disc_max_ch: int = 384

    # ── Mel dimensions ──
    mel_height: int = 64
    mel_width: int = 552         # input, will be padded for generation
    mel_pad_width: int = 576     # divisible by 2^4 = 16 → clean upsample path
    hop_length: int = 200
    n_mels: int = 64
    sample_rate: int = 22050
    n_fft: int = 1024
    f_min: float = 0.0
    f_max: float = 11025.0

    # ── Data ──
    data_dir: str = "data/animal1000"
    train_split: float = 0.95
    segment_seconds: float = 5.0

    # ── Training ──
    batch_size: int = 32          # L4: 22GB VRAM, 4.2 used → room for 32
    epochs: int = 300
    g_lr: float = 2e-4
    d_lr: float = 2e-4
    beta1: float = 0.0           # Adam β₁ = 0 for GAN stability
    beta2: float = 0.99
    r1_gamma: float = 10.0       # R1 gradient penalty strength
    r1_every: int = 16           # apply R1 every N steps (saves memory)

    # ── Loss weights ──
    g_adv_weight: float = 1.0
    g_class_weight: float = 0.5
    d_class_weight: float = 0.5

    # ── AMP ──
    use_amp: bool = True

    # ── Augmentation (mild, only for small classes) ──
    aug_freq_mask_max: int = 4   # max consecutive freq bins to mask
    aug_time_mask_max: int = 40  # max consecutive time frames to mask
    aug_prob: float = 0.3        # probability of applying augmentation

    # ── Generation ──
    num_samples: int = 5

    # ── Paths ──
    checkpoint_dir: str = "models"
    generator_path: str = "models/gan_generator_best.pth"
    discriminator_path: str = "models/gan_discriminator_best.pth"

    # ── Classes ──
    CLASSES: Tuple[str, ...] = ('Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen')

    @property
    def class_to_idx(self):
        return {c: i for i, c in enumerate(self.CLASSES)}


config = GANConfig()
