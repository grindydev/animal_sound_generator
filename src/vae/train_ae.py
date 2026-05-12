"""
train_autoencoder.py — Autoencoder v2 Training Pipeline.

Trains the ImprovedAutoencoder with skip connections, residual blocks, and attention.
Uses 80/20 train/val split (test data folded into train for +33% data).

Features:
  - Tqdm-style per-batch progress
  - Checkpoint resume (Ctrl+C → save → restart → continue)
"""
import copy
import os
import sys
import warnings
import torch
from torch import nn
from torch import optim
from torch.amp import autocast, GradScaler

# Ensure project root and src/ are importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src'))

from data_loader import get_dataloaders, get_transformations
from vae.autoencoder import ImprovedAutoencoder
import helper_utils

warnings.filterwarnings("ignore")

# ==================== CONFIG ====================
CONFIG = {
    "mode": "train",          # "test" = 5 epoch smoke test | "train" = full training
    "device": "auto",
    "train_fraction": 0.8,    # 80% train (was 60% — test folded in)
    "val_fraction": 0.2,      # 20% val
    "lr": 3e-4,               # lower LR for stability (300M model)
    "weight_decay": 1e-3,
    "latent_dim": 2048,
    "base_channels": 32,       # 32→64→128→256 = 149M params, fits 4GB VRAM
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",

    "test": {
        "num_epochs": 5,
        "batch_size": 8,       # smaller batch for bigger model (GTX 1650 4GB)
        "patience": 3,
        "num_workers": 1,
    },

    "train": {
        "num_epochs": 40,
        "batch_size": 8,
        "patience": 8,
        "num_workers": 4,
    }
}

# ==================== APPLY CONFIG ====================
MODE = CONFIG["mode"]
SETTINGS = CONFIG[MODE]

NUM_EPOCHS = SETTINGS["num_epochs"]
BATCH_SIZE = SETTINGS["batch_size"]
PATIENCE = SETTINGS["patience"]
NUM_WORKERS = SETTINGS["num_workers"]
TRAIN_FRACTION = CONFIG["train_fraction"]
VAL_FRACTION = CONFIG["val_fraction"]
LR = CONFIG["lr"]
WEIGHT_DECAY = CONFIG["weight_decay"]
LATENT_DIM = CONFIG["latent_dim"]

BEST_MODEL_PATH = f"models/best_autoencoder_{MODE}.pth"
CHECKPOINT_DIR = f"models/autoencoder_checkpoints/{MODE}"

print(f"🔧 CONFIG → {MODE.upper()} MODE (Improved V2)")
print(f"   Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR} | Latent: {LATENT_DIM}")
print(f"   Split: {TRAIN_FRACTION*100:.0f}% train / {VAL_FRACTION*100:.0f}% val")
print(f"   Checkpoints: {CHECKPOINT_DIR} | Best: {BEST_MODEL_PATH}")

# ==================== DEVICE SETUP ====================
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

use_amp = False  # disabled — float16 overflows with large random model

# ==================== LOAD DATA ====================
train_loader, val_loader, test_loader, num_classes = get_dataloaders(
    batch_size=BATCH_SIZE,
    train_fraction=TRAIN_FRACTION,
    val_fraction=VAL_FRACTION,
    num_workers=NUM_WORKERS,
)

print(f"✅ Data: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val / {len(test_loader.dataset)} test (unused)")

# ==================== MODEL ====================
BASE_CH = CONFIG["base_channels"]
model = ImprovedAutoencoder(latent_dim=LATENT_DIM, base_channels=BASE_CH)
n_params = sum(p.numel() for p in model.parameters())
print(f"✅ Model: {n_params:,} params ({n_params/1e6:.1f}M)")

loss_function = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# ==================== CHECKPOINTING ====================

def save_checkpoint(model, optimizer, scheduler, epoch):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"ae_{epoch:06d}.pth")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }, path)


def load_checkpoint(model, optimizer, scheduler):
    if not os.path.isdir(CHECKPOINT_DIR):
        return 0
    files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("ae_")])
    if not files:
        return 0
    path = os.path.join(CHECKPOINT_DIR, files[-1])
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    print(f"   📂 Resumed from {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


scaler = GradScaler() if use_amp else None

train_transform, eval_transform = get_transformations()
train_transform = train_transform.to(device)
eval_transform = eval_transform.to(device)


# ==================== TRAINING FUNCTIONS ====================

def train_epoch(model, train_loader, loss_fn, optimizer, device, train_tfm,
                scaler, use_amp, pbar=None):
    model.train()
    running_loss = 0.0
    nan_count = 0

    for batch_idx, (waveforms, _) in enumerate(train_loader):
        waveforms = waveforms.to(device)
        spectrograms = train_tfm(waveforms)

        # Safety: check for NaN in input
        if torch.isnan(spectrograms).any():
            print(f"⚠️  NaN in input spectrograms — skipping batch")
            continue

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast(device_type="cuda"):
                reconstructed = model(spectrograms)
                reconstructed = torch.clamp(reconstructed, -10, 10)  # prevent explosion
                loss = loss_fn(reconstructed, spectrograms)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            reconstructed = model(spectrograms)
            reconstructed = torch.clamp(reconstructed, -10, 10)
            loss = loss_fn(reconstructed, spectrograms)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if torch.isnan(loss):
            nan_count += 1
            if nan_count <= 3:
                print(f"⚠️  NaN loss at batch {batch_idx} — skipping")
            continue

        running_loss += loss.item() * spectrograms.size(0)
        if pbar:
            pbar.update_batch(batch_idx + 1, postfix_dict={"mse": f"{loss.item():.4f}"})

    if nan_count > 0:
        print(f"⚠️  {nan_count} batches had NaN — model may be unstable, check LR")

    return running_loss / max(len(train_loader.dataset), 1)


def validate_epoch(model, val_loader, loss_fn, device, eval_tfm, pbar=None):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (waveforms, _) in enumerate(val_loader):
            waveforms = waveforms.to(device)
            spectrograms = eval_tfm(waveforms)
            reconstructed = model(spectrograms)
            loss = loss_fn(reconstructed, spectrograms)
            running_loss += loss.item() * spectrograms.size(0)
            if pbar:
                pbar.update_batch(batch_idx + 1)

    return running_loss / len(val_loader.dataset)


# ==================== TRAINING LOOP ====================

def training_loop():
    model.to(device)

    # Resume
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    start_epoch = load_checkpoint(model, optimizer, scheduler)

    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"🚀 AUTOENCODER V2 TRAINING — {MODE.upper()} MODE")
    print(f"   Device: {device} | Epochs: {start_epoch+1}-{NUM_EPOCHS}")
    print(f"{'='*70}\n")

    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            train_pbar = helper_utils.NestedProgressBar(
                total_epochs=NUM_EPOCHS, total_batches=len(train_loader), mode="train")
            train_pbar.update_epoch(epoch + 1)

            epoch_loss = train_epoch(model, train_loader, loss_function, optimizer, device,
                                     train_transform, scaler, use_amp, pbar=train_pbar)
            train_pbar.batch_bar.close()

            val_pbar = helper_utils.NestedProgressBar(
                total_epochs=1, total_batches=len(val_loader), mode="eval")
            epoch_val_loss = validate_epoch(model, val_loader, loss_function, device,
                                            eval_transform, pbar=val_pbar)
            val_pbar.close()

            current_lr = scheduler.get_last_lr()[0]
            train_pbar.update_epoch(epoch + 1, postfix_dict={
                "train_mse": f"{epoch_loss:.4f}", "val_mse": f"{epoch_val_loss:.4f}",
                "lr": f"{current_lr:.6f}",
            })
            scheduler.step()

            # Save checkpoint EVERY epoch (safe for Ctrl+C)
            save_checkpoint(model, optimizer, scheduler, epoch + 1)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'latent_dim': LATENT_DIM, 'val_mse': best_val_loss,
                    'epoch': best_epoch, 'mode': MODE,
                }, BEST_MODEL_PATH)
                print(f"  → ✅ Best model (MSE={best_val_loss:.6f}, epoch {best_epoch})")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"\n⏹️ Early stopping: {patience_counter} epochs no improvement")
                    break
    except KeyboardInterrupt:
        print(f"\n⏸️  Interrupted at epoch {epoch+1}. Checkpoint saved — resume anytime.")

    if best_model_state:
        model.load_state_dict(best_model_state)

    print(f"\n💾 Best model → {BEST_MODEL_PATH} (val_mse={best_val_loss:.6f})")
    return model


if __name__ == "__main__":
    training_loop()
