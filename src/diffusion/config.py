"""
config.py — Hyperparameters for Diffusion Refinement (Phase 7b).

Single source of truth. Imported by unet.py, diffusion.py, train.py, and inference.py.
"""
from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    # ── Spectrogram ───────────────────────────────────────
    n_mels: int = 64
    sample_rate: int = 22050
    hop_length: int = 200
    n_fft: int = 1024
    f_min: float = 0.0
    f_max: float = 11025.0               # sample_rate / 2
    spec_channels: int = 1               # grayscale mel spectrogram

    # ── Diffusion ─────────────────────────────────────────
    timesteps: int = 1000                 # total noise steps (train)
    beta_start: float = 0.0001
    beta_end: float = 0.02
    cosine_s: float = 0.008              # cosine schedule offset (Improved DDPM)

    # ── U-Net ────────────────────────────────────────────
    base_channels: int = 64               # channels at first level
    channel_multipliers: tuple = (1, 2, 4, 4)  # [64, 128, 256, 256]
    res_blocks_per_level: int = 2
    attention_levels: tuple = (2, 3)      # apply self-attention at levels 2 and 3
    time_emb_dim: int = 256               # sinusoidal time embedding size
    class_emb_dim: int = 64               # animal class embedding size
    num_classes: int = 8                  # Dog, Cat, Rooster, Frog, Crow, Insect, Hen, Noise
    dropout: float = 0.1

    # ── Training ──────────────────────────────────────────
    segment_frames: int = 552             # ~5 seconds of mel frames (at hop_length=200)
    batch_size: int = 8
    learning_rate: float = 2e-4
    lr_decay: float = 0.999               # per-epoch decay
    adam_betas: tuple = (0.9, 0.999)
    num_epochs: int = 50
    num_workers: int = 0

    # ── Inference ────────────────────────────────────────
    inference_steps: int = 50             # DDIM sampling steps
    refinement_strength: float = 0.6      # default img2img strength (0.0-1.0)

    # ── Paths ────────────────────────────────────────────
    data_dir: str = "data/animal_audio"
    mel_dir: str = "data/animal_mel"      # precomputed mel cache (optional)
    model_dir: str = "models"
    checkpoint_dir: str = "models/diffusion_checkpoints"

    # ── Misc ─────────────────────────────────────────────
    fp16: bool = False                     # MPS doesn't benefit from AMP
    log_interval: int = 50                # batches between logs
    save_interval: int = 5                # epochs between checkpoints
    eval_samples: int = 4                 # how many validation samples to generate


# Default instance for easy import
config = DiffusionConfig()
