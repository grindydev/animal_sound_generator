"""
train.py — Phase 7b: Diffusion U-Net Training
===============================================

Trains a U-Net diffusion model to denoise spectrograms. The U-Net learns
to predict noise at various timesteps, enabling img2img refinement of
VAE-generated spectrograms at inference time.

TRAINING APPROACH (Standard DDPM on real spectrograms):
  1. Load real audio → compute mel spectrogram → x_0 (clean)
  2. Pick random timestep t → add noise → x_t
  3. U-Net(x_t, t, class_label) → predicted noise
  4. Loss = MSE(predicted_noise, actual_noise)

INFERENCE (img2img refinement):
  1. VAE generates blurry spectrogram → x_vae
  2. Add noise at t = strength * T → x_t
  3. Denoise N steps → sharp spectrogram
  4. HiFi-GAN converts → crisp audio

FRAMEWORK (matches src/hifigan/train.py):
  - CONFIG dict at top → mode/test/train settings
  - Device auto-detection (CUDA > MPS > CPU)
  - Separate functions: train_epoch(), validate(), training_loop()
  - Checkpoint resume support
  - Progress tracking with 📉/➡️ indicators

MODES:
  "test"  — 5 epochs, batch=4  → quick dev smoke test
  "train" — 50 epochs, batch=8 → full training

Run:
    python src/diffusion/train.py
"""
import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.diffusion.config import config as cfg
from src.diffusion.unet import SpectrogramUNet
from src.diffusion.diffusion import DiffusionProcess


# ═══════════════════════════════════════════════════════════════
#  CONFIG (EDIT ONLY THIS SECTION)
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    "mode": "train",         # "test" = 5 epoch smoke test | "train" = full training
    "device": "auto",        # "auto", "cuda", "mps", or "cpu"

    # ── Shared ──────────────────────────────────────────
    "data_dir": cfg.data_dir,
    "segment_frames": cfg.segment_frames,
    "save_interval": cfg.save_interval,
    "log_interval": cfg.log_interval,

    "test": {
        "num_epochs": 5,
        "batch_size": 4,
        "num_workers": 1,
    },

    "train": {
        "num_epochs": 50,
        "batch_size": 1,
        "num_workers": 0,
    },
}

# ═══════════════════════════════════════════════════════════════
#  APPLY CONFIG
# ═══════════════════════════════════════════════════════════════

MODE = CONFIG["mode"]
SETTINGS = CONFIG[MODE] if MODE in CONFIG else CONFIG["train"]

NUM_EPOCHS = SETTINGS["num_epochs"]
BATCH_SIZE = SETTINGS["batch_size"]
NUM_WORKERS = SETTINGS["num_workers"]
SEGMENT_FRAMES = CONFIG["segment_frames"]
SAVE_INTERVAL = CONFIG["save_interval"]
LOG_INTERVAL = CONFIG["log_interval"]

BEST_MODEL_PATH = f"models/diffusion_unet_{MODE}.pth"
CHECKPOINT_DIR = os.path.join(cfg.checkpoint_dir, MODE)


# ═══════════════════════════════════════════════════════════════
#  DEVICE SETUP (matches hifigan/train.py)
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
scaler = torch.amp.GradScaler() if use_amp else None


# ═══════════════════════════════════════════════════════════════
#  AUDIO LOADER
# ═══════════════════════════════════════════════════════════════

def _load_audio(path: str):
    """Load audio file. Tries torchaudio first, falls back to soundfile."""
    try:
        import torchaudio
        return torchaudio.load(path)
    except Exception:
        pass

    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype='float32')
        wav = torch.from_numpy(data)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.transpose(0, 1)
        return wav, sr
    except Exception as e:
        import warnings
        warnings.warn(f"⚠️  Audio load FAILED: {path} → {e}")
        return None, None


# ═══════════════════════════════════════════════════════════════
#  CLASS MAPPING
# ═══════════════════════════════════════════════════════════════

CLASSES = ['Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen', 'Noise']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}


# ═══════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════

class DiffusionDataset(Dataset):
    """
    Loads audio files, computes mel spectrograms, pairs with class labels.

    Returns:
        mel: normalized mel spectrogram [1, 64, segment_frames]
        label: class index [1]
    """

    def __init__(self, data_dir: str, segment_frames: int, split: str = "train"):
        self.data_dir = data_dir
        self.segment_frames = segment_frames
        self.samples = []  # list of (path, class_idx)

        for cls_name in sorted(os.listdir(data_dir)):
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            if cls_name not in CLASS_TO_IDX:
                continue
            cls_idx = CLASS_TO_IDX[cls_name]
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith('.wav'):
                    self.samples.append((
                        os.path.abspath(os.path.join(cls_dir, fname)),
                        cls_idx,
                    ))

        # Shuffle and split
        np.random.seed(42)
        np.random.shuffle(self.samples)
        split_idx = int(len(self.samples) * 0.9)
        self.samples = self.samples[:split_idx] if split == "train" else self.samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        wav, sr = _load_audio(path)

        if wav is None:
            return torch.zeros(1, 64, self.segment_frames), torch.tensor(0, dtype=torch.long)

        # Resample if needed
        if sr != cfg.sample_rate:
            import torchaudio
            wav = torchaudio.transforms.Resample(sr, cfg.sample_rate)(wav)

        # Mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Compute mel spectrogram
        mel = compute_mel(wav)  # [1, 64, T]

        # Crop or pad to segment_frames
        T = mel.shape[-1]
        if T < self.segment_frames:
            pad = torch.zeros(1, 64, self.segment_frames - T)
            mel = torch.cat([mel, pad], dim=-1)
        elif T > self.segment_frames:
            # Random crop
            start = torch.randint(0, T - self.segment_frames, (1,)).item()
            mel = mel[:, :, start:start + self.segment_frames]

        return mel, torch.tensor(label, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════
#  MEL COMPUTATION (on-the-fly, matches data_loader.py format)
# ═══════════════════════════════════════════════════════════════

def compute_mel(audio: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized mel spectrogram matching VAE training format.
    Uses the same params as data_loader.py and hifigan/config.py.
    """
    from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

    # Use the norm stats from HiFi-GAN config (matches data_loader.py SimpleNormalize)
    norm_mean = -18.4903
    norm_std = 19.8031

    d = audio.device
    mel_tfm = MelSpectrogram(
        sample_rate=cfg.sample_rate, n_fft=cfg.n_fft,
        hop_length=cfg.hop_length, win_length=cfg.n_fft,
        n_mels=cfg.n_mels, f_min=cfg.f_min, f_max=cfg.f_max, power=2,
    ).to(d)

    db_tfm = AmplitudeToDB(stype='power', top_db=None).to(d)
    spec = mel_tfm(audio.squeeze(1))
    return (db_tfm(spec) - norm_mean) / norm_std


# ═══════════════════════════════════════════════════════════════
#  SAVE / LOAD CHECKPOINT
# ═══════════════════════════════════════════════════════════════

def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    """Save model + optimizer state for resume."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"unet_{epoch:06d}.pth")
    torch.save({
        "unet": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": cfg.__dict__,
    }, path)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load latest checkpoint. Returns start_epoch (0 if none found)."""
    if not os.path.isdir(checkpoint_dir):
        return 0
    files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("unet_") and f.endswith(".pth")])
    if not files:
        return 0
    latest = files[-1]
    path = os.path.join(checkpoint_dir, latest)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["unet"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"   Resumed from {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


# ═══════════════════════════════════════════════════════════════
#  TRAIN / VALIDATE
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, diffusion, train_loader, optimizer):
    """Train one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc="  Train", leave=False)

    for mel, labels in pbar:
        mel = mel.to(device)         # [B, 1, 64, T]
        labels = labels.to(device)   # [B]
        B = mel.shape[0]

        # Sample random timesteps
        t = torch.randint(0, diffusion.timesteps, (B,), device=device)

        # Add noise: x_t = sqrt(α_cumprod) * x_0 + sqrt(1-α_cumprod) * ε
        noise = torch.randn_like(mel)
        x_t = diffusion.q_sample(mel, t, noise)

        # U-Net predicts noise
        pred_noise = model(x_t, t, labels)

        # Loss: MSE between predicted and actual noise
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(train_loader)


def validate(model, diffusion, val_loader):
    """Validate — returns average MSE loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for mel, labels in val_loader:
            mel = mel.to(device)
            labels = labels.to(device)
            B = mel.shape[0]

            t = torch.randint(0, diffusion.timesteps, (B,), device=device)
            noise = torch.randn_like(mel)
            x_t = diffusion.q_sample(mel, t, noise)
            pred_noise = model(x_t, t, labels)
            loss = F.mse_loss(pred_noise, noise)
            total_loss += loss.item()

    return total_loss / max(len(val_loader), 1)


# ═══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def training_loop():
    """Full diffusion training with test/train mode, checkpoint resume."""

    # ═════════════════════════════════════════════════════════
    #  BANNER
    # ═════════════════════════════════════════════════════════
    model = SpectrogramUNet(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n🔧 Diffusion Refinement → {MODE.upper()} MODE")
    if is_cuda:
        print(f"   GPU:    {torch.cuda.get_device_name(0)}")
    print(f"   Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | Workers: {NUM_WORKERS}")
    print(f"   Timesteps: {cfg.timesteps} | Model: {n_params:,} params ({n_params/1e6:.1f}M)")
    print(f"   Segment: {SEGMENT_FRAMES} mel frames")
    print(f"   Mixed precision: {'yes' if use_amp else 'no'}")
    print(f"   Best model → {BEST_MODEL_PATH}")

    # ═════════════════════════════════════════════════════════
    #  DATA
    # ═════════════════════════════════════════════════════════
    train_ds = DiffusionDataset(CONFIG["data_dir"], SEGMENT_FRAMES, split="train")
    val_ds = DiffusionDataset(CONFIG["data_dir"], SEGMENT_FRAMES, split="val")

    print(f"\n✅ Data loaded: {len(train_ds)} train / {len(val_ds)} val")

    pin_mem = NUM_WORKERS > 0 and is_cuda
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin_mem, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin_mem, drop_last=False,
    )

    # ── DATA VALIDATION ─────────────────────────────────
    _check_mel, _check_labels = next(iter(train_loader))
    print(f"   📊 First batch: mel shape={tuple(_check_mel.shape)}, "
          f"min={_check_mel.min():.4f}, max={_check_mel.max():.4f}, "
          f"mean_abs={_check_mel.abs().mean():.4f}")
    if _check_mel.abs().max() < 1e-6:
        print("   🚨 ALL ZEROS — mel data not loading correctly.")
        print("   🚨 Training stopped to save time.")
        return None
    print(f"   Labels: {_check_labels.tolist()} → {[CLASSES[l] for l in _check_labels.tolist()]}")

    # ═════════════════════════════════════════════════════════
    #  MODELS
    # ═════════════════════════════════════════════════════════
    model = model.to(device)
    diffusion = DiffusionProcess(cfg)

    # ═════════════════════════════════════════════════════════
    #  OPTIMIZER
    # ═════════════════════════════════════════════════════════
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, betas=cfg.adam_betas)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_decay)

    # ═════════════════════════════════════════════════════════
    #  RESUME
    # ═════════════════════════════════════════════════════════
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)
    start_epoch = load_checkpoint(model, optimizer, CHECKPOINT_DIR, device)

    # ═════════════════════════════════════════════════════════
    #  TRAIN
    # ═════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"🚀 DIFFUSION TRAINING — {MODE.upper()} MODE")
    print(f"   Device: {device} | Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"   Saving last model → {BEST_MODEL_PATH}")
    print(f"{'='*60}\n")

    best_val = float("inf")
    BEST_PATH = os.path.join(cfg.model_dir, f"diffusion_unet_{MODE}_best.pth")

    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()

        avg_loss = train_epoch(model, diffusion, train_loader, optimizer)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        val_loss = validate(model, diffusion, val_loader)

        dt = time.time() - t0
        trend = "📉" if val_loss < best_val else "➡️"
        print(f"── Epoch {epoch+1:3d}/{NUM_EPOCHS} "
              f"({dt:.0f}s) ── "
              f"loss={avg_loss:.4f} val={val_loss:.4f} {trend} lr={lr:.2e}")

        # Save checkpoint
        if (epoch + 1) % SAVE_INTERVAL == 0 or epoch == NUM_EPOCHS - 1:
            save_checkpoint(model, optimizer, epoch + 1, CHECKPOINT_DIR)

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"unet": model.state_dict(), "config": cfg.__dict__}, BEST_PATH)

    # Save final model
    torch.save({"unet": model.state_dict(), "config": cfg.__dict__}, BEST_MODEL_PATH)
    print(f"\n💾 Final model saved to: {BEST_MODEL_PATH}")
    print(f"   Best val loss: {best_val:.4f}")
    print("✅ Training complete!")

    return model


if __name__ == "__main__":
    trained_model = training_loop()
