"""
train.py — HiFi-GAN training loop.

Trains on (mel, audio) pairs from real data. Uses smart_crop
for energy-based segment selection, then computes mel on-the-fly.

Run:
    python -m src.hifigan.train
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Project path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.smart_crop import smart_crop
from src.hifigan.config import config as cfg
from src.hifigan.generator import HiFiGANGenerator
from src.hifigan.discriminator import Discriminator
from src.hifigan.losses import MelL1Loss, generator_loss, discriminator_loss
from src.hifigan.utils import save_checkpoint, load_checkpoint, scan_checkpoints


# ══════════════════════════════════════════════════════════════
#  Dataset — loads audio, smart_crop → (mel, audio) pairs
# ══════════════════════════════════════════════════════════════

class HiFiGANDataset(Dataset):
    """
    Loads audio files, applies smart_crop for energy-based selection,
    returns (mel_spectrogram, audio_segment) pairs.
    """

    def __init__(self, data_dir: str, segment_size: int, split: str = "train"):
        self.data_dir = data_dir
        self.segment_size = segment_size
        self.files = []

        # Collect all .wav files
        for cls_name in sorted(os.listdir(data_dir)):
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith('.wav'):
                    self.files.append(os.path.join(cls_dir, fname))

        # Simple train/val split (last 10% for validation)
        np.random.seed(42)
        np.random.shuffle(self.files)
        split_idx = int(len(self.files) * 0.9)

        if split == "train":
            self.files = self.files[:split_idx]
        else:
            self.files = self.files[split_idx:]

        print(f"HiFiGAN {split} dataset: {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        try:
            wav, sr = torchaudio.load(path)
        except Exception:
            # Corrupt file fallback
            wav = torch.zeros(1, self.segment_size)
            return wav

        # Resample if needed
        if sr != cfg.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, cfg.sample_rate)
            wav = resampler(wav)

        # Convert to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # smart_crop to get a segment with actual animal sound
        if wav.shape[-1] <= self.segment_size:
            # Short clip: pad with zeros
            pad = torch.zeros(1, self.segment_size - wav.shape[-1])
            wav = torch.cat([wav, pad], dim=-1)
        else:
            crops = smart_crop(
                wav,
                crop_samples=self.segment_size,
                threshold_db=cfg.smart_crop_threshold_db,
                num_crops=1,
                merge_gap_samples=cfg.smart_crop_merge_gap,
            )
            wav = crops[0]

        return wav  # [1, segment_size]


# ══════════════════════════════════════════════════════════════
#  Mel computation (on-the-fly, used in training loop)
# ══════════════════════════════════════════════════════════════

def compute_mel(audio: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Compute normalized mel spectrogram from raw audio.

    Uses the SAME MelSpectrogram + SimpleNormalize as the VAE training.
    This ensures HiFi-GAN inputs match what VAE produces.

    Args:
        audio: [B, 1, segment_size] raw waveform (already on device)
    Returns:
        mel: [B, 64, T_frames] normalized dB mel spectrogram
    """
    from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

    # Create transforms directly on target device
    mel_tfm = MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
        power=2,
    ).to(device)

    db_tfm = AmplitudeToDB(stype='power', top_db=None).to(device)

    # [B, 1, T] → squeeze channel → MelSpectrogram expects [B, T]
    spec = mel_tfm(audio.squeeze(1))  # [B, n_mels, T_frames]
    spec_db = db_tfm(spec)
    spec_norm = (spec_db - cfg.norm_mean) / cfg.norm_std

    return spec_norm


# ══════════════════════════════════════════════════════════════
#  Training Loop
# ══════════════════════════════════════════════════════════════

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: segment={cfg.segment_size}, batch={cfg.batch_size}, lr={cfg.learning_rate}")

    # ── Models ──────────────────────────────────────────
    generator = HiFiGANGenerator(cfg).to(device)
    discriminator = Discriminator().to(device)

    print(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    # ── Optimizers ──────────────────────────────────────
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=cfg.learning_rate,
        betas=cfg.adam_betas,
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=cfg.learning_rate,
        betas=cfg.adam_betas,
    )

    # ── LR Scheduler ────────────────────────────────────
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=cfg.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=cfg.lr_decay)

    # ── Loss ────────────────────────────────────────────
    mel_loss_fn = MelL1Loss(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
    ).to(device)

    # ── Data ────────────────────────────────────────────
    train_ds = HiFiGANDataset(cfg.data_dir, cfg.segment_size, split="train")
    val_ds = HiFiGANDataset(cfg.data_dir, cfg.segment_size, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        drop_last=False,
    )

    # ── Resume from checkpoint ──────────────────────────
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    start_epoch = load_checkpoint(
        generator, discriminator, optimizer_g, optimizer_d,
        cfg.checkpoint_dir, device,
    )
    if start_epoch > 0:
        print(f"Resumed from epoch {start_epoch}")

    # ── Train ───────────────────────────────────────────
    for epoch in range(start_epoch, cfg.num_epochs):
        generator.train()
        discriminator.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_mel_loss = 0.0
        epoch_start = time.time()

        for batch_idx, audio in enumerate(train_loader):
            audio = audio.to(device)  # [B, 1, segment_size]

            # Compute mel from real audio (input to generator)
            real_mel = compute_mel(audio, device)  # [B, 64, T_frames]

            # Trim audio to match exact mel frame count
            target_len = real_mel.shape[-1] * cfg.hop_length
            real_audio_trim = audio[..., :target_len]

            # ── Generator forward ───────────────────────
            fake_audio = generator(real_mel, target_length=target_len)

            # ── Discriminator step ──────────────────────
            optimizer_d.zero_grad()

            real_scores, real_features = discriminator(real_audio_trim)
            fake_scores_d, fake_features_d = discriminator(fake_audio.detach())

            d_loss, d_losses = discriminator_loss(real_scores, fake_scores_d)
            d_loss.backward()
            optimizer_d.step()

            # ── Generator step ──────────────────────────
            optimizer_g.zero_grad()

            fake_scores_g, fake_features_g = discriminator(fake_audio)

            g_loss, g_losses = generator_loss(
                fake_audio, real_audio_trim,
                fake_scores_g, fake_features_g, real_features,
                mel_loss_fn,
                lambda_mel=cfg.lambda_mel,
                lambda_fm=cfg.lambda_fm,
                lambda_adv=cfg.lambda_adv,
            )
            g_loss.backward()
            optimizer_g.step()

            # ── Logging ─────────────────────────────────
            epoch_g_loss += g_losses["g_total"]
            epoch_d_loss += d_losses["d_total"]
            epoch_mel_loss += g_losses["g_mel"]

            if batch_idx % cfg.log_interval == 0:
                print(
                    f"Epoch {epoch+1:3d} | Batch {batch_idx:4d} | "
                    f"G={g_loss.item():.4f} D={d_loss.item():.4f} "
                    f"mel={g_losses['g_mel']:.4f} "
                    f"fm={g_losses['g_fm']:.4f} "
                    f"adv={g_losses['g_adv']:.4f}"
                )

        # ── Epoch summary ───────────────────────────────
        n_batches = len(train_loader)
        scheduler_g.step()
        scheduler_d.step()

        epoch_time = time.time() - epoch_start
        print(
            f"── Epoch {epoch+1}/{cfg.num_epochs} "
            f"({epoch_time:.0f}s) ── "
            f"G={epoch_g_loss/n_batches:.4f} "
            f"D={epoch_d_loss/n_batches:.4f} "
            f"mel={epoch_mel_loss/n_batches:.4f} "
            f"lr={scheduler_g.get_last_lr()[0]:.2e}"
        )

        # ── Validation ──────────────────────────────────
        if epoch % cfg.save_interval == 0 or epoch == cfg.num_epochs - 1:
            generator.eval()
            val_mel_loss = 0.0
            with torch.no_grad():
                for val_audio in val_loader:
                    val_audio = val_audio.to(device)
                    val_mel = compute_mel(val_audio, device)
                    val_target_len = val_mel.shape[-1] * cfg.hop_length
                    val_trimmed = val_audio[..., :val_target_len]
                    val_fake = generator(val_mel, target_length=val_target_len)
                    val_mel_loss += mel_loss_fn(val_fake, val_trimmed).item()

            val_mel_loss /= max(len(val_loader), 1)
            print(f"   Val mel loss: {val_mel_loss:.4f}")

            # ── Save checkpoint ─────────────────────────
            save_checkpoint(
                generator, discriminator,
                optimizer_g, optimizer_d,
                epoch + 1, cfg.checkpoint_dir,
            )
            generator.train()

    # ── Save final ──────────────────────────────────────
    final_path = os.path.join(cfg.model_dir, "hifigan_generator.pth")
    os.makedirs(cfg.model_dir, exist_ok=True)
    torch.save(
        {"generator": generator.state_dict(), "config": cfg.__dict__},
        final_path,
    )
    print(f"Final model saved: {final_path}")
    print("Training complete!")


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train()
