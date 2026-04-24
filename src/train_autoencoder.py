"""
train_autoencoder.py — Phase 3: Autoencoder Training Pipeline
==============================================================

WHAT YOU'LL BUILD:
  • CONFIG dict — lr, weight_decay, batch_size, epochs (same pattern as train.py)
  • Autoencoder training loop with MSE reconstruction loss
  • Early stopping on val loss (lower = better, opposite of classifier!)
  • Best model checkpointing (saved to disk when val loss improves)
  • Device-aware: CUDA / MPS / CPU auto-detect

KEY CONCEPTS:
  • Autoencoder loss = MSE(reconstructed, original) — no labels needed!
  • Early stopping tracks LOWEST val loss (classifier tracked HIGHEST accuracy)
  • Same training pipeline as classifier — only the loss changes

COURSE REFERENCE:
  • NSFW main.py — CONFIG dict, training loop pattern
  • L2-M1 scheduler/main.py — Cosine LR, early stopping
"""

import copy
import warnings
import torch
from torch import nn
from torch import optim
from torch.amp import autocast, GradScaler

from data_loader import get_dataloaders, get_transformations
from model import SimpleAudioAutoencoder
import helper_utils

warnings.filterwarnings("ignore")

# ==================== CONFIG (EDIT ONLY THIS SECTION) ====================
CONFIG = {
    "mode": "test",                      # "test" = fast dev, "train" = full training
    "device": "auto",                    # "auto", "cuda", "mps", or "cpu"
    "train_fraction": 0.6,
    "val_fraction": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-3,
    "latent_dim": 1024,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",

    "test": {
        "num_epochs": 5,
        "batch_size": 16,
        "patience": 3,
        "num_workers": 1,
    },

    "train": {
        "num_epochs": 40,
        "batch_size": 16,
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

print(f"🔧 CONFIG → {MODE.upper()} MODE")
print(f"   Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR} | Patience: {PATIENCE} | Workers: {NUM_WORKERS}")
print(f"   Latent dim: {LATENT_DIM}")
print(f"   Best model → {BEST_MODEL_PATH}")

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

# ==================== LOAD DATA ====================
train_loader, val_loader, test_loader, num_classes = get_dataloaders(
    batch_size=BATCH_SIZE,
    train_fraction=TRAIN_FRACTION,
    val_fraction=VAL_FRACTION,
    num_workers=NUM_WORKERS,
)

print(f"✅ Data loaded: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val / {len(test_loader.dataset)} test")

# ==================== MODEL, LOSS, OPTIMIZER, SCHEDULER ====================
model = SimpleAudioAutoencoder(latent_dim=LATENT_DIM)

# Autoencoder: MSE loss — reconstruct the input (no labels!)
loss_function = nn.MSELoss()

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# ==================== MIXED PRECISION ====================
use_amp = is_cuda
scaler = GradScaler() if use_amp else None

# ==================== TRANSFORMATIONS ====================
train_transform, eval_transform = get_transformations()
train_transform = train_transform.to(device)
eval_transform = eval_transform.to(device)


# ==================== TRAINING FUNCTIONS ====================

def train_epoch(model, train_loader, loss_function, optimizer, device, train_transform,
                scaler, use_amp, pbar=None):
    """Train one epoch — reconstruct spectrograms (no labels in loss)."""
    model.train()
    running_loss = 0.0

    for batch_idx, (waveforms, _labels) in enumerate(train_loader):
        waveforms = waveforms.to(device)

        # Transform raw waveforms → mel spectrograms on GPU
        spectrograms = train_transform(waveforms)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast(device_type="cuda"):
                reconstructed = model(spectrograms)
                loss = loss_function(reconstructed, spectrograms)  # target = input!
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            reconstructed = model(spectrograms)
            loss = loss_function(reconstructed, spectrograms)  # target = input!
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * spectrograms.size(0)

        if pbar:
            pbar.update_batch(batch_idx + 1, postfix_dict={"mse": f"{loss.item():.4f}"})

    return running_loss / len(train_loader.dataset)


def validate_epoch(model, val_loader, loss_function, device, eval_transform, pbar=None):
    """Validate one epoch — MSE reconstruction loss only (no accuracy)."""
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (waveforms, _labels) in enumerate(val_loader):
            waveforms = waveforms.to(device)
            spectrograms = eval_transform(waveforms)

            reconstructed = model(spectrograms)
            loss = loss_function(reconstructed, spectrograms)  # target = input!

            running_loss += loss.item() * spectrograms.size(0)

            if pbar:
                pbar.update_batch(batch_idx + 1)

    return running_loss / len(val_loader.dataset)


# ==================== TRAINING LOOP ====================

def training_loop(model, train_loader, val_loader, loss_function, optimizer, scheduler,
                  num_epochs, device, train_transform, eval_transform, scaler, use_amp):
    """
    Full training loop.
    Early stopping on LOWEST val MSE (classifier tracked HIGHEST accuracy).
    """
    model.to(device)

    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    train_losses, val_losses = [], []

    print("\n" + "=" * 70)
    print(f"🚀 AUTOENCODER TRAINING — {MODE.upper()} MODE")
    print(f"   Device: {device} | Epochs: {num_epochs} | Best model → {BEST_MODEL_PATH}")
    print("=" * 70)

    for epoch in range(num_epochs):
        train_pbar = helper_utils.NestedProgressBar(
            total_epochs=num_epochs,
            total_batches=len(train_loader),
            mode="train",
        )
        train_pbar.update_epoch(epoch + 1)

        # ── Train ──
        epoch_loss = train_epoch(model, train_loader, loss_function, optimizer, device,
                                 train_transform, scaler, use_amp, pbar=train_pbar)
        train_pbar.batch_bar.close()

        # ── Validate ──
        val_pbar = helper_utils.NestedProgressBar(
            total_epochs=1,
            total_batches=len(val_loader),
            mode="eval",
        )
        epoch_val_loss = validate_epoch(model, val_loader, loss_function, device,
                                        eval_transform, pbar=val_pbar)
        val_pbar.close()

        train_losses.append(epoch_loss)
        val_losses.append(epoch_val_loss)

        current_lr = scheduler.get_last_lr()[0]

        train_pbar.update_epoch(epoch + 1, postfix_dict={
            "train_mse": f"{epoch_loss:.4f}",
            "val_mse": f"{epoch_val_loss:.4f}",
            "lr": f"{current_lr:.6f}",
        })

        scheduler.step()

        # === Save best model when val loss improves (LOWER is better!) ===
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())

            torch.save({
                'model_state_dict': model.state_dict(),
                'latent_dim': LATENT_DIM,
                'val_mse': best_val_loss,
                'epoch': best_epoch,
                'mode': MODE,
            }, BEST_MODEL_PATH)

            print(f"  → ✅ New best model saved (MSE={best_val_loss:.6f} at epoch {best_epoch})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹️ Early stopping: {patience_counter} epochs without improvement")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, [train_losses, val_losses]


# ==================== RUN ====================
if __name__ == "__main__":
    trained_model, training_metrics = training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        device=device,
        train_transform=train_transform,
        eval_transform=eval_transform,
        scaler=scaler,
        use_amp=use_amp,
    )

    # Plot training curves
    try:
        helper_utils.plot_training_metrics(training_metrics)
    except Exception as e:
        print(f"⚠️ Plotting failed: {e}")

    # Evaluate on test set
    test_loss = validate_epoch(trained_model, test_loader, loss_function, device, eval_transform)
    print(f"\n🎯 Test Set: MSE={test_loss:.6f}")
    print(f"   Best model saved to: {BEST_MODEL_PATH}")
