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
    use_linear_schedule: bool = True      # linear schedule → more signal at high t (vs cosine)

    # ── U-Net ────────────────────────────────────────────
    base_channels: int = 32               # v12: ~4M params (was 64 → 18M overfit)
    channel_multipliers: tuple = (1, 1, 2, 2)  # [32, 32, 64, 64] (was (1,2,2,4))
    res_blocks_per_level: int = 1         # 1 block per level
    attention_levels: tuple = (3,)        # attention only at deepest level
    time_emb_dim: int = 256               # v12: reduced (was 512)
    class_emb_dim: int = 128              # v12: reduced (was 256)
    num_classes: int = 7                  # v12: removed Noise class (was 8)
    dropout: float = 0.3                  # v12: more dropout (was 0.2)

    # ── Training ──────────────────────────────────────────
    segment_frames: int = 552
    train_fraction: float = 0.95
    batch_size: int = 8
    learning_rate: float = 2e-4
    adam_betas: tuple = (0.9, 0.999)
    adam_weight_decay: float = 1e-4
    num_epochs: int = 150                 # v10: more epochs, smaller dataset
    num_workers: int = 4
    grad_clip_norm: float = 5.0
    ema_decay: float = 0.9999
    loss_type: str = "l1"
    predict_x0: bool = True

    # ── V9: Class Balance ────────────────────────────────
    balance_classes: bool = True           # oversample rare classes each epoch
    noise_max_samples: int = 200           # cap Noise class at 200 (Option B)

    # ── Augmentation ─────────────────────────────────────
    augment: bool = True
    strong_augment: bool = True
    pitch_shift_bins: int = 4
    time_stretch_range: tuple = (0.85, 1.15)   # gentler stretch for 5s clean clips

    # ── Frequency-Weighted Loss ──────────────────────────
    freq_weight_max: float = 3.0          # v8: bin 63 weighted 3× more than bin 0

    # ── GAN (Fix C) ─────────────────────────────────────
    gan_weight: float = 0.1               # λ: GAN loss weight vs L1
    disc_channels: int = 64               # base channels for discriminator
    disc_lr: float = 2e-4                 # discriminator learning rate

    # ── Inference ────────────────────────────────────────
    inference_steps: int = 200            # v12: more steps for stability (was 100)
    refinement_strength: float = 0.4      # v12: gentler refinement (was 0.6)
    ddim_eta: float = 0.0                # DDIM stochasticity (0.0 = deterministic)
    cfg_scale: float = 1.5               # v12: reduced CFG (was 2.0, causing artifacts)
    uncond_prob: float = 0.1             # v12: reduced uncond (was 0.15)

    # ── V12: Spectral Balance + Smoothness + Classifier Guidance ─
    spectral_balance_weight: float = 10.0  # weight for spectral band ratio loss
    temporal_smooth_weight: float = 5.0    # weight for frame-to-frame smoothness
    classifier_guidance_weight: float = 0.3  # weight for classifier perceptual loss
    classifier_ckpt: str = "models/best_audio_cnn_train.pth"  # frozen classifier path

    # ── V12: Post-Generation Stats Matching ─
    target_mel_mean: float = 0.0           # target mean for generated mels
    target_mel_std: float = 0.5            # target std for generated mels
    mel_clip_range: float = 2.0            # clamp generated mels to [-2, 2]

    # ── Paths ────────────────────────────────────────────

    # ── Paths ────────────────────────────────────────────
    data_dir: str = "data/esc50"
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
