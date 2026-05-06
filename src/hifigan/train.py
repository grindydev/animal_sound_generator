"""
train.py — HiFi-GAN training loop.

Trains on (mel, audio) pairs from real data. Uses smart_crop
for energy-based segment selection, then computes mel on-the-fly.

Two modes (like train_vae.py / finetune_vae.py):
    test   — 5 epochs, batch=4, 1 worker  → quick dev check
    train  — 50 epochs, batch=8, 4 workers → full training

Set mode at the top of CONFIG dict below.

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
#  CONFIG — change "mode" to switch test ↔ train
# ══════════════════════════════════════════════════════════════

CONFIG = {
    # ── Mode & Device ──────────────────────────────────
    "mode": "train",         # "test" = quick dev (5 epochs), "train" = full (50 epochs)
    "device": "auto",        # "auto", "cuda", "mps", or "cpu"

    # ── Shared (not mode-specific) ─────────────────────
    "segment_size": cfg.segment_size,
    "checkpoint_dir": cfg.checkpoint_dir,
    "model_dir": cfg.model_dir,
    "data_dir": cfg.data_dir,
    "log_interval": cfg.log_interval,
    "save_interval": cfg.save_interval,

    # ── Test mode ──────────────────────────────────────
    "test": {
        "num_epochs": 5,
        "batch_size": 4,
        "num_workers": 1,
    },

    # ── Train mode ─────────────────────────────────────
    "train": {
        "num_epochs": 50,
        "batch_size": 8,
        "num_workers": 4,
    },
}

# ══════════════════════════════════════════════════════════════
#  APPLY CONFIG
# ══════════════════════════════════════════════════════════════

MODE = CONFIG["mode"]
SETTINGS = CONFIG[MODE]

NUM_EPOCHS = SETTINGS["num_epochs"]
BATCH_SIZE = SETTINGS["batch_size"]
NUM_WORKERS = SETTINGS["num_workers"]
SEGMENT_SIZE = CONFIG["segment_size"]

# Device
if CONFIG["device"] == "auto":
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
else:
    DEVICE = torch.device(CONFIG["device"])

BEST_MODEL_PATH = os.path.join(cfg.model_dir, f"hifigan_generator_{MODE}.pth")
CHECKPOINT_DIR = os.path.join(cfg.checkpoint_dir, MODE)


def print_banner():
    print("=" * 60)
    print(f"🔧 HiFi-GAN → {MODE.upper()} MODE")
    print(f"   Device:    {DEVICE}")
    print(f"   Epochs:    {NUM_EPOCHS}")
    print(f"   Batch:     {BATCH_SIZE}")
    print(f"   Workers:   {NUM_WORKERS}")
    print(f"   Segment:   {SEGMENT_SIZE} samples (~{SEGMENT_SIZE/cfg.sample_rate:.1f}s)")
    print(f"   Best path: {BEST_MODEL_PATH}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════
#  Dataset — loads audio, smart_crop → audio segments
# ══════════════════════════════════════════════════════════════

class HiFiGANDataset(Dataset):
    """Loads audio files, applies smart_crop for energy-based selection."""

    def __init__(self, data_dir: str, segment_size: int, split: str = "train"):
        self.data_dir = data_dir
        self.segment_size = segment_size
        self.files = []

        for cls_name in sorted(os.listdir(data_dir)):
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith('.wav'):
                    self.files.append(os.path.join(cls_dir, fname))

        np.random.seed(42)
        np.random.shuffle(self.files)
        split_idx = int(len(self.files) * 0.9)

        if split == "train":
            self.files = self.files[:split_idx]
        else:
            self.files = self.files[split_idx:]

        print(f"   HiFiGAN {split}: {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            wav, sr = torchaudio.load(path)
        except Exception:
            return torch.zeros(1, self.segment_size)

        if sr != cfg.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, cfg.sample_rate)
            wav = resampler(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if wav.shape[-1] <= self.segment_size:
            pad = torch.zeros(1, self.segment_size - wav.shape[-1])
            wav = torch.cat([wav, pad], dim=-1)
        else:
            crops = smart_crop(
                wav, crop_samples=self.segment_size,
                threshold_db=cfg.smart_crop_threshold_db,
                num_crops=1,
                merge_gap_samples=cfg.smart_crop_merge_gap,
            )
            wav = crops[0]

        return wav  # [1, segment_size]


# ══════════════════════════════════════════════════════════════
#  Mel computation (on-the-fly)
# ══════════════════════════════════════════════════════════════

def compute_mel(audio: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Compute normalized mel spectrogram matching VAE training."""
    from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

    mel_tfm = MelSpectrogram(
        sample_rate=cfg.sample_rate, n_fft=cfg.n_fft,
        hop_length=cfg.hop_length, win_length=cfg.win_length,
        n_mels=cfg.n_mels, f_min=cfg.f_min, f_max=cfg.f_max, power=2,
    ).to(device)

    db_tfm = AmplitudeToDB(stype='power', top_db=None).to(device)
    spec = mel_tfm(audio.squeeze(1))
    spec_db = db_tfm(spec)
    return (spec_db - cfg.norm_mean) / cfg.norm_std


# ══════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════

def train():
    print_banner()

    # ── Models ──────────────────────────────────────────
    generator = HiFiGANGenerator(cfg).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"\n   Generator:     {g_params:,} params")
    print(f"   Discriminator: {d_params:,} params")
    print(f"   Total:         {g_params + d_params:,} params\n")

    # ── Optimizers ──────────────────────────────────────
    opt_g = torch.optim.Adam(generator.parameters(), lr=cfg.learning_rate, betas=cfg.adam_betas)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.learning_rate, betas=cfg.adam_betas)

    sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=cfg.lr_decay)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=cfg.lr_decay)

    # ── Loss ────────────────────────────────────────────
    mel_loss_fn = MelL1Loss(
        sample_rate=cfg.sample_rate, n_fft=cfg.n_fft,
        hop_length=cfg.hop_length, n_mels=cfg.n_mels,
        f_min=cfg.f_min, f_max=cfg.f_max,
    ).to(DEVICE)

    # ── Data ────────────────────────────────────────────
    train_ds = HiFiGANDataset(CONFIG["data_dir"], SEGMENT_SIZE, split="train")
    val_ds = HiFiGANDataset(CONFIG["data_dir"], SEGMENT_SIZE, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
    )
    print(f"   Batches/epoch: train={len(train_loader)}, val={len(val_loader)}\n")

    # ── Resume ──────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)
    start_epoch = load_checkpoint(
        generator, discriminator, opt_g, opt_d,
        CHECKPOINT_DIR, DEVICE,
    )
    if start_epoch > 0:
        print(f"   Resumed from epoch {start_epoch}\n")

    # ── Train loop ──────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS):
        generator.train()
        discriminator.train()

        ep_g, ep_d, ep_mel = 0.0, 0.0, 0.0
        t0 = time.time()

        for bi, audio in enumerate(train_loader):
            audio = audio.to(DEVICE)

            real_mel = compute_mel(audio, DEVICE)
            target_len = real_mel.shape[-1] * cfg.hop_length
            real_trim = audio[..., :target_len]

            fake = generator(real_mel, target_length=target_len)

            # Discriminator
            opt_d.zero_grad()
            r_score, r_feat = discriminator(real_trim)
            f_score_d, _ = discriminator(fake.detach())
            d_loss, d_dict = discriminator_loss(r_score, f_score_d)
            d_loss.backward()
            opt_d.step()

            # Generator
            opt_g.zero_grad()
            f_score_g, f_feat_g = discriminator(fake)
            g_loss, g_dict = generator_loss(
                fake, real_trim, f_score_g, f_feat_g, r_feat, mel_loss_fn,
                lambda_mel=cfg.lambda_mel, lambda_fm=cfg.lambda_fm, lambda_adv=cfg.lambda_adv,
            )
            g_loss.backward()
            opt_g.step()

            ep_g += g_dict["g_total"]
            ep_d += d_dict["d_total"]
            ep_mel += g_dict["g_mel"]

            if bi % cfg.log_interval == 0:
                print(
                    f"  Epoch {epoch+1:3d} | Batch {bi:4d} | "
                    f"G={g_loss.item():.4f} D={d_loss.item():.4f} "
                    f"mel={g_dict['g_mel']:.4f} fm={g_dict['g_fm']:.4f} adv={g_dict['g_adv']:.4f}"
                )

        # Epoch summary
        nb = len(train_loader)
        sched_g.step()
        sched_d.step()
        dt = time.time() - t0
        print(
            f"── Epoch {epoch+1}/{NUM_EPOCHS} "
            f"({dt:.0f}s) ── "
            f"G={ep_g/nb:.4f} D={ep_d/nb:.4f} "
            f"mel={ep_mel/nb:.4f} lr={sched_g.get_last_lr()[0]:.2e}"
        )

        # Validation + save
        if epoch % CONFIG["save_interval"] == 0 or epoch == NUM_EPOCHS - 1:
            generator.eval()
            val_mel = 0.0
            with torch.no_grad():
                for va in val_loader:
                    va = va.to(DEVICE)
                    vm = compute_mel(va, DEVICE)
                    vt = vm.shape[-1] * cfg.hop_length
                    vf = generator(vm, target_length=vt)
                    val_mel += mel_loss_fn(vf, va[..., :vt]).item()
            val_mel /= max(len(val_loader), 1)
            print(f"   Val mel loss: {val_mel:.4f}")

            save_checkpoint(generator, discriminator, opt_g, opt_d, epoch + 1, CHECKPOINT_DIR)
            generator.train()

    # Final save
    torch.save(
        {"generator": generator.state_dict(), "config": cfg.__dict__},
        BEST_MODEL_PATH,
    )
    print(f"\n✅ Final model saved: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    train()
