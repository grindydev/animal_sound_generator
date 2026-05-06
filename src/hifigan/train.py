"""
train.py — Phase 7a: HiFi-GAN Neural Vocoder Training
======================================================

Trains a HiFi-GAN vocoder: learns to convert mel spectrograms back to
realistic audio waveforms. Uses smart_crop for energy-based segment
selection, then computes mel on-the-fly.

FRAMEWORK (matches train_vae.py / finetune_vae.py):
  - CONFIG dict at top → mode/test/train settings
  - Device auto-detection (CUDA > MPS > CPU)
  - AMP mixed precision on CUDA
  - Separate functions: train_epoch(), validate(), training_loop()
  - Checkpoint resume support

MODES:
  "test"  — 5 epochs, batch=4  → quick dev smoke test
  "train" — 50 epochs, batch=8 → full training

Run:
    python src/hifigan/train.py
"""
import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.smart_crop import smart_crop
from src.hifigan.config import config as cfg
from src.hifigan.generator import HiFiGANGenerator
from src.hifigan.discriminator import Discriminator
from src.hifigan.losses import MelL1Loss, generator_loss, discriminator_loss
from src.hifigan.utils import save_checkpoint, load_checkpoint
import helper_utils


# ═══════════════════════════════════════════════════════════════
#  CONFIG (EDIT ONLY THIS SECTION)
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    "mode": "train",         # "test" = quick dev, "train" = full
    "device": "auto",        # "auto", "cuda", "mps", or "cpu"

    # ── Shared ──────────────────────────────────────────
    "data_dir": cfg.data_dir,
    "segment_size": cfg.segment_size,
    "save_interval": cfg.save_interval,

    "test": {
        "num_epochs": 5,
        "batch_size": 1,
        "num_workers": 1,
    },

    "train": {
        "num_epochs": 50,
        "batch_size": 2,       # GTX 1650 (4GB VRAM) — safe limit
        "num_workers": 4,
    },
}

# ═══════════════════════════════════════════════════════════════
#  APPLY CONFIG
# ═══════════════════════════════════════════════════════════════

MODE = CONFIG["mode"]
SETTINGS = CONFIG[MODE]

NUM_EPOCHS = SETTINGS["num_epochs"]
BATCH_SIZE = SETTINGS["batch_size"]
NUM_WORKERS = SETTINGS["num_workers"]
SEGMENT_SIZE = CONFIG["segment_size"]

BEST_MODEL_PATH = f"models/hifigan_generator_{MODE}.pth"
CHECKPOINT_DIR = os.path.join(cfg.checkpoint_dir, MODE)

# ═══════════════════════════════════════════════════════════════
#  DEVICE SETUP (matches train_vae.py)
# ═══════════════════════════════════════════════════════════════

if CONFIG["device"] == "auto":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        is_cuda = True
        print("🚀 Using CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        is_cuda = False
        print("🍎 Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        is_cuda = False
        print("⚠️  Using CPU")
else:
    device = torch.device(CONFIG["device"])
    is_cuda = (CONFIG["device"] == "cuda")

use_amp = is_cuda
scaler = GradScaler() if use_amp else None

# ═══════════════════════════════════════════════════════════════
#  BANNER
# ═══════════════════════════════════════════════════════════════

print(f"\n🔧 HiFi-GAN → {MODE.upper()} MODE")
if is_cuda:
    print(f"   GPU:    {torch.cuda.get_device_name(0)}")
print(f"   Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | Workers: {NUM_WORKERS}")
print(f"   Segment: {SEGMENT_SIZE} samples (~{SEGMENT_SIZE/cfg.sample_rate:.1f}s)")
print(f"   Mixed precision: {'yes' if use_amp else 'no'}")
print(f"   Best model → {BEST_MODEL_PATH}")


# ═══════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════

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
        self.files = self.files[:split_idx] if split == "train" else self.files[split_idx:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            wav, sr = torchaudio.load(path)
        except Exception:
            return torch.zeros(1, self.segment_size)

        if sr != cfg.sample_rate:
            wav = torchaudio.transforms.Resample(sr, cfg.sample_rate)(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if wav.shape[-1] <= self.segment_size:
            pad = torch.zeros(1, self.segment_size - wav.shape[-1])
            wav = torch.cat([wav, pad], dim=-1)
        else:
            crops = smart_crop(
                wav, crop_samples=self.segment_size,
                threshold_db=cfg.smart_crop_threshold_db,
                num_crops=1, merge_gap_samples=cfg.smart_crop_merge_gap,
            )
            wav = crops[0]

        return wav  # [1, segment_size]


# ═══════════════════════════════════════════════════════════════
#  MEL COMPUTATION (on-the-fly)
# ═══════════════════════════════════════════════════════════════

def compute_mel(audio: torch.Tensor) -> torch.Tensor:
    """Compute normalized mel spectrogram matching VAE training."""
    from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

    d = audio.device
    mel_tfm = MelSpectrogram(
        sample_rate=cfg.sample_rate, n_fft=cfg.n_fft,
        hop_length=cfg.hop_length, win_length=cfg.win_length,
        n_mels=cfg.n_mels, f_min=cfg.f_min, f_max=cfg.f_max, power=2,
    ).to(d)

    db_tfm = AmplitudeToDB(stype='power', top_db=None).to(d)
    spec = mel_tfm(audio.squeeze(1))
    return (db_tfm(spec) - cfg.norm_mean) / cfg.norm_std


# ═══════════════════════════════════════════════════════════════
#  TRAIN / VALIDATE
# ═══════════════════════════════════════════════════════════════

def train_epoch(generator, discriminator, train_loader,
                opt_g, opt_d, mel_loss_fn, pbar=None):
    """Train one epoch. Returns (avg_g, avg_d, avg_mel)."""
    generator.train()
    discriminator.train()

    total_g, total_d, total_mel = 0.0, 0.0, 0.0
    nb = len(train_loader)

    for bi, audio in enumerate(train_loader):
        audio = audio.to(device)

        real_mel = compute_mel(audio)
        n_frames = real_mel.shape[-1]
        target_len = n_frames * cfg.hop_length

        # Pad real audio to match exact mel frame count (avoids off-by-1 mismatch)
        if audio.shape[-1] < target_len:
            real_trim = F.pad(audio, (0, target_len - audio.shape[-1]))
        else:
            real_trim = audio[..., :target_len]

        fake = generator(real_mel, target_length=target_len)

        # ── Discriminator ───────────────────────────────
        opt_d.zero_grad()

        if use_amp:
            with autocast(device_type="cuda"):
                r_score, r_feat = discriminator(real_trim)
                f_score_d, _ = discriminator(fake.detach())
                d_loss, d_dict = discriminator_loss(r_score, f_score_d)
            scaler.scale(d_loss).backward()
            scaler.step(opt_d)
            scaler.update()
        else:
            r_score, r_feat = discriminator(real_trim)
            f_score_d, _ = discriminator(fake.detach())
            d_loss, d_dict = discriminator_loss(r_score, f_score_d)
            d_loss.backward()
            opt_d.step()

        # ── Generator ───────────────────────────────────
        opt_g.zero_grad()

        if use_amp:
            with autocast(device_type="cuda"):
                f_score_g, f_feat_g = discriminator(fake)
                g_loss, g_dict = generator_loss(
                    fake, real_trim, f_score_g, f_feat_g, r_feat, mel_loss_fn,
                    lambda_mel=cfg.lambda_mel,
                    lambda_fm=cfg.lambda_fm,
                    lambda_adv=cfg.lambda_adv,
                )
            scaler.scale(g_loss).backward()
            scaler.step(opt_g)
            scaler.update()
        else:
            f_score_g, f_feat_g = discriminator(fake)
            g_loss, g_dict = generator_loss(
                fake, real_trim, f_score_g, f_feat_g, r_feat, mel_loss_fn,
                lambda_mel=cfg.lambda_mel,
                lambda_fm=cfg.lambda_fm,
                lambda_adv=cfg.lambda_adv,
            )
            g_loss.backward()
            opt_g.step()

        total_g += g_dict["g_total"]
        total_d += d_dict["d_total"]
        total_mel += g_dict["g_mel"]

        if pbar:
            pbar.update_batch(bi + 1, postfix_dict={
                "G": f"{g_loss.item():.2f}",
                "D": f"{d_loss.item():.2f}",
                "mel": f"{g_dict['g_mel']:.1f}",
            })

    return total_g / nb, total_d / nb, total_mel / nb


def validate(generator, val_loader, mel_loss_fn, pbar=None):
    """Validate — returns average mel loss."""
    generator.eval()
    total_mel = 0.0

    with torch.no_grad():
        for bi, va in enumerate(val_loader):
            va = va.to(device)
            vm = compute_mel(va)
            vt = vm.shape[-1] * cfg.hop_length
            vf = generator(vm, target_length=vt)
            total_mel += mel_loss_fn(vf, va[..., :vt]).item()
            if pbar:
                pbar.update_batch(bi + 1)

    return total_mel / max(len(val_loader), 1)


# ═══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def training_loop():
    """Full HiFi-GAN training with test/train mode, AMP, checkpoint resume."""

    # ── Models ──────────────────────────────────────────
    generator = HiFiGANGenerator(cfg).to(device)
    discriminator = Discriminator().to(device)

    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"\n   Generator:     {g_params:,} params")
    print(f"   Discriminator: {d_params:,} params")
    print(f"   Total:         {g_params + d_params:,} params")

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
    ).to(device)

    # ── Data ────────────────────────────────────────────
    train_ds = HiFiGANDataset(CONFIG["data_dir"], SEGMENT_SIZE, split="train")
    val_ds = HiFiGANDataset(CONFIG["data_dir"], SEGMENT_SIZE, split="val")

    print(f"\n✅ Data loaded: {len(train_ds)} train / {len(val_ds)} val")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
    )

    # ── Resume ──────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)
    start_epoch = load_checkpoint(
        generator, discriminator, opt_g, opt_d,
        CHECKPOINT_DIR, device,
    )
    if start_epoch > 0:
        print(f"   Resumed from epoch {start_epoch}")

    # ── Train ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"🚀 HiFi-GAN TRAINING — {MODE.upper()} MODE")
    print(f"   Device: {device} | Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"   Saving last model → {BEST_MODEL_PATH}")
    print(f"{'='*60}\n")

    best_val_mel = float("inf")
    BEST_PATH = os.path.join(cfg.model_dir, f"hifigan_generator_{MODE}_best.pth")

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_pbar = helper_utils.NestedProgressBar(
            total_epochs=NUM_EPOCHS,
            total_batches=len(train_loader),
            mode="train",
        )
        train_pbar.update_epoch(epoch + 1)

        avg_g, avg_d, avg_mel = train_epoch(
            generator, discriminator, train_loader,
            opt_g, opt_d, mel_loss_fn, pbar=train_pbar,
        )
        train_pbar.batch_bar.close()

        sched_g.step()
        sched_d.step()

        lr = sched_g.get_last_lr()[0]

        # Validate + save checkpoint
        if epoch % CONFIG["save_interval"] == 0 or epoch == NUM_EPOCHS - 1:
            val_pbar = helper_utils.NestedProgressBar(
                total_epochs=1,
                total_batches=len(val_loader),
                mode="eval",
            )
            val_mel = validate(generator, val_loader, mel_loss_fn, pbar=val_pbar)
            val_pbar.close()

            save_checkpoint(
                generator, discriminator, opt_g, opt_d,
                epoch + 1, CHECKPOINT_DIR,
            )
        else:
            val_mel = validate(generator, val_loader, mel_loss_fn)

        # Track best model
        if val_mel < best_val_mel:
            best_val_mel = val_mel
            torch.save(
                {"generator": generator.state_dict(), "config": cfg.__dict__},
                BEST_PATH,
            )

        train_pbar.update_epoch(epoch + 1, postfix_dict={
            "G": f"{avg_g:.2f}",
            "D": f"{avg_d:.2f}",
            "mel": f"{avg_mel:.1f}",
            "val": f"{val_mel:.1f}",
            "lr": f"{lr:.1e}",
        })

    # ── Save final ──────────────────────────────────────
    torch.save(
        {"generator": generator.state_dict(), "config": cfg.__dict__},
        BEST_MODEL_PATH,
    )
    print(f"\n💾 Final model saved to: {BEST_MODEL_PATH}")
    print("✅ Training complete!")

    return generator


# ═══════════════════════════════════════════════════════════════
#  RUN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    trained_generator = training_loop()
