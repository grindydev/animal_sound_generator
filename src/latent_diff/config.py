"""
config.py — Latent Diffusion Config (Path B).

Diffuse in compressed 2D latent space [16, 4, 35] instead of raw mel [1, 64, 552].
Encoder (149M) is frozen. Decoder (2M) is trained from scratch without skip connections.
Tiny UNet (3M) diffuses on 2,240 values — 16× fewer than raw mel.
"""
from dataclasses import dataclass


@dataclass
class LatentDiffConfig:
    # ── Encoder (frozen) ──────────────────────────────────
    encoder_ckpt: str = "models/best_autoencoder_train.pth"
    encoder_base_channels: int = 32
    encoder_latent_dim: int = 2048

    # ── Latent space ─────────────────────────────────────
    bottleneck_channels: int = 256        # autoencoder c4 (base_ch=32 → 32*8=256)
    latent_channels: int = 16             # reduced channels for diffusion
    latent_height: int = 4                # spatial H after encoder
    latent_width: int = 35                # spatial W after encoder
    # Total: 16 × 4 × 35 = 2,240 values (16× less than 64×552=35,328)

    # ── Small Decoder (no skip connections) ──────────────
    # Relative channel multipliers from bottleneck. Auto-detected at runtime.
    decoder_multipliers: tuple = (1.0, 0.5, 0.25, 0.125)  # × bottleneck_ch

    # ── Spectrogram ───────────────────────────────────────
    n_mels: int = 64
    sample_rate: int = 22050
    hop_length: int = 200
    spec_channels: int = 1
    segment_frames: int = 552

    # ── Diffusion UNet (tiny) ────────────────────────────
    unet_base_channels: int = 64
    unet_channel_multipliers: tuple = (1, 2, 2)        # 3 levels
    unet_res_blocks: int = 1
    unet_attention_levels: tuple = (2,)                # bottleneck only
    time_emb_dim: int = 256
    class_emb_dim: int = 128
    num_classes: int = 8
    dropout: float = 0.1
    uncond_prob: float = 0.1                           # 10% unconditional for CFG

    # ── Diffusion ────────────────────────────────────────
    timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    use_linear_schedule: bool = True
    loss_type: str = "l2"
    inference_steps: int = 100
    cfg_scale: float = 2.0

    # ── Training ─────────────────────────────────────────
    learning_rate: float = 2e-4
    adam_weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    ema_decay: float = 0.9999

    # ── Paths ────────────────────────────────────────────
    data_dir: str = "data/animal_audio"
    model_dir: str = "models"
    decoder_ckpt: str = "models/latent_decoder_best.pth"
    diffusion_ckpt: str = "models/latent_diffusion_best.pth"


config = LatentDiffConfig()
