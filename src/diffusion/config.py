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
    base_channels: int = 96               # channels at first level (balanced for L4 24GB)
    channel_multipliers: tuple = (1, 2, 3, 4)  # [96, 192, 288, 384]
    res_blocks_per_level: int = 2         # 2 blocks per resolution
    attention_levels: tuple = (2, 3)      # attention at deepest 2 levels (288ch, 384ch)
    time_emb_dim: int = 512               # sinusoidal time embedding size
    class_emb_dim: int = 256              # animal class embedding size (was 64)
    num_classes: int = 8
    dropout: float = 0.1

    # ── Training ──────────────────────────────────────────
    segment_frames: int = 552             # ~5 seconds of mel frames (at hop_length=200)
    batch_size: int = 8
    learning_rate: float = 2e-4
    lr_decay: float = 0.999               # per-epoch decay
    adam_betas: tuple = (0.9, 0.999)
    adam_weight_decay: float = 1e-4       # AdamW weight decay
    num_epochs: int = 50
    num_workers: int = 0
    grad_clip_norm: float = 5.0           # relaxed clipping for diffusion
    ema_decay: float = 0.9999             # EMA decay for smoothed inference weights
    loss_type: str = "l2"                 # "l2" = MSE | "l1" = MAE | "huber" = smooth L1

    # ── Inference ────────────────────────────────────────
    inference_steps: int = 50             # DDIM sampling steps
    refinement_strength: float = 0.6      # default img2img strength (0.0-1.0)
    ddim_eta: float = 0.0                # DDIM stochasticity (0.0 = deterministic, 1.0 = DDPM)

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
