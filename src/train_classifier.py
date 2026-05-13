"""
train_classifier.py — Audio Classifier Training (ImprovedAudioCNN)

Standard pattern matching src/diffusion/train.py and src/hifigan/train.py:
  - CONFIG dict at top → mode/test/train settings
  - Device auto-detection (CUDA > MPS > CPU)
  - Separate functions: train_epoch(), validate_epoch(), training_loop()
  - Checkpoint resume + best model tracking
  - Progress: ── Epoch X/Y (Ts) ── loss=X val_acc=X% 📉/➡️ lr=X

Usage:
    python src/train_classifier.py
"""
import copy
import os
import sys
import time
import warnings
import torch
from torch import nn, optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src'))

from data_loader import get_dataloaders, get_transformations
from model import ImprovedAudioCNN

warnings.filterwarnings("ignore")


# ==================== CONFIG ====================
CONFIG = {
    "mode": "train",              # "test" = 5 epoch smoke test | "train" = full training
    "device": "auto",
    "train_fraction": 0.8,
    "val_fraction": 0.2,
    "lr": 1e-3,
    "weight_decay": 0.05,
    "label_smoothing": 0.1,
    "dropout": 0.3,

    "test": {
        "num_epochs": 5,
        "batch_size": 64,
        "patience": 3,
        "num_workers": 8,
    },

    "train": {
        "num_epochs": 40,
        "batch_size": 64,
        "patience": 8,
        "num_workers": 8,
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
LABEL_SMOOTHING = CONFIG["label_smoothing"]
DROPOUT = CONFIG["dropout"]

BEST_MODEL_PATH = f"models/best_audio_cnn_{MODE}.pth"
CHECKPOINT_DIR = f"models/classifier_checkpoints/{MODE}"

print(f"🔧 CONFIG → {MODE.upper()} MODE (ImprovedAudioCNN)")
print(f"   Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}")
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

use_amp = is_cuda

# ==================== LOAD DATA ====================
train_loader, val_loader, test_loader, num_classes = get_dataloaders(
    batch_size=BATCH_SIZE,
    train_fraction=TRAIN_FRACTION,
    val_fraction=VAL_FRACTION,
    num_workers=NUM_WORKERS,
)
print(f"✅ Data: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val / {len(test_loader.dataset)} test (unused)")

# ==================== MODEL ====================
model = ImprovedAudioCNN(num_classes=num_classes, dropout=DROPOUT)
n_params = sum(p.numel() for p in model.parameters())
print(f"✅ Model: {n_params:,} params ({n_params/1e6:.1f}M)")

loss_function = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
scaler = GradScaler() if use_amp else None

train_transform, eval_transform = get_transformations()
train_transform = train_transform.to(device)
eval_transform = eval_transform.to(device)


# ==================== CHECKPOINTING ====================

def save_checkpoint(model, optimizer, scheduler, epoch):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"classifier_{epoch:06d}.pth")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }, path)


def load_checkpoint(model, optimizer, scheduler):
    if not os.path.isdir(CHECKPOINT_DIR):
        return 0
    files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("classifier_")])
    if not files:
        return 0
    path = os.path.join(CHECKPOINT_DIR, files[-1])
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    print(f"   📂 Resumed from {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


# ==================== TRAINING FUNCTIONS ====================

def train_epoch(model, train_loader, loss_fn, optimizer, device, train_tfm,
                scaler, use_amp):
    """Train one epoch. Returns average loss."""
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc="  Train", leave=False)

    for waveforms, labels in pbar:
        waveforms, labels = waveforms.to(device), labels.to(device)
        spectrograms = train_tfm(waveforms)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast(device_type="cuda"):
                outputs = model(spectrograms)
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(spectrograms)
            loss = loss_fn(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * spectrograms.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / len(train_loader.dataset)


def validate_epoch(model, val_loader, loss_fn, device, eval_tfm):
    """Validate — returns (avg_loss, accuracy%)."""
    model.eval()
    running_loss = 0.0
    correct = total = 0

    with torch.no_grad():
        for waveforms, labels in val_loader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            spectrograms = eval_tfm(waveforms)

            outputs = model(spectrograms)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * spectrograms.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader.dataset)
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


# ==================== TRAINING LOOP ====================

def training_loop():
    model.to(device)

    # Resume
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    start_epoch = load_checkpoint(model, optimizer, scheduler)

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"🚀 CLASSIFIER TRAINING — {MODE.upper()} MODE")
    print(f"   Device: {device} | Epochs: {start_epoch+1}-{NUM_EPOCHS}")
    print(f"{'='*60}\n")

    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            t0 = time.time()

            epoch_loss = train_epoch(model, train_loader, loss_function, optimizer,
                                     device, train_transform, scaler, use_amp)
            val_loss, val_acc = validate_epoch(model, val_loader, loss_function,
                                               device, eval_transform)
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            dt = time.time() - t0
            trend = "📉" if val_acc > best_val_acc else "➡️"

            print(f"── Epoch {epoch+1:3d}/{NUM_EPOCHS} "
                  f"({dt:.0f}s) ── "
                  f"loss={epoch_loss:.4f} val_acc={val_acc:.1f}% {trend} lr={current_lr:.2e}")

            # Save checkpoint EVERY epoch (safe for Ctrl+C)
            save_checkpoint(model, optimizer, scheduler, epoch + 1)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'num_classes': num_classes,
                    'val_accuracy': best_val_acc,
                    'epoch': best_epoch,
                    'mode': MODE,
                }, BEST_MODEL_PATH)
                print(f"      💾 Best model saved (val_acc={best_val_acc:.1f}%)")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"\n⏹️  Early stopping: {patience_counter} epochs no improvement")
                    break

    except KeyboardInterrupt:
        print(f"\n⏸️  Interrupted at epoch {epoch+1}. Checkpoint saved — resume anytime.")

    # Save final
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'val_accuracy': best_val_acc,
        'epoch': best_epoch,
        'mode': MODE,
    }, BEST_MODEL_PATH)
    print(f"\n💾 Best model → {BEST_MODEL_PATH} (val_acc={best_val_acc:.1f}% at epoch {best_epoch})")

    return model


if __name__ == "__main__":
    training_loop()
