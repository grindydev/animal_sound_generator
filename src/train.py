"""
train.py — Phase 2: Training Pipeline for Audio Classifier
============================================================

WHAT YOU'LL BUILD:
  • CONFIG dict — lr, weight_decay, batch_size, epochs (like NSFW main.py)
  • Training loop with epoch-level progress output
  • Early stopping + Cosine LR scheduler
  • Best model checkpointing (saved to disk when val accuracy improves)
  • Device-aware: CUDA / MPS / CPU auto-detect
  • test/train mode in CONFIG for fast development vs full training

KEY CONCEPTS:
  • Same training pipeline as NSFW — only the data changes (spectrograms vs images)
  • The classifier trained here becomes the EVALUATOR in Phase 5
  • CONFIG["mode"] = "test" for fast iteration, "train" for full training

COURSE REFERENCE:
  • NSFW main.py — CONFIG dict, training loop pattern
  • L2-M1 scheduler/main.py — Cosine LR, early stopping

CONFIG (adjust these):
  mode = "test" or "train"
  lr = 1e-3
  weight_decay = 0.05
  batch_size = 16
  label_smoothing = 0.1
"""

import copy
import warnings
import torch
from torch import nn
from torch import optim
from torch.amp import autocast, GradScaler

from data_loader import get_dataloaders, get_transformations
from model import SimpleAudioCNN
import helper_utils
warnings.filterwarnings("ignore")

# ==================== CONFIG (EDIT ONLY THIS SECTION) ====================
# Same pattern as NSFW main.py — all settings in one place.
# Switch mode between "test" (fast dev) and "train" (full training).
CONFIG = {
    "mode": "test",                    # "test" = fast dev mode, "train" = full training
    "device": "auto",                  # "auto", "cuda", "mps", or "cpu"
    "val_fraction": 0.15,
    "lr": 1e-3,
    "weight_decay": 0.05,
    "label_smoothing": 0.1,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",

    # Fast development mode — small subset, few epochs
    "test": {
        "num_epochs": 5,
        "batch_size": 16,
        "patience": 3,
    },

    # Full training mode
    "train": {
        "num_epochs": 40,
        "batch_size": 16,
        "patience": 8,
    }
}

# ==================== APPLY CONFIG ====================
MODE = CONFIG["mode"]
SETTINGS = CONFIG[MODE]

NUM_EPOCHS = SETTINGS["num_epochs"]
BATCH_SIZE = SETTINGS["batch_size"]
PATIENCE = SETTINGS["patience"]
VAL_FRACTION = CONFIG["val_fraction"]
LR = CONFIG["lr"]
WEIGHT_DECAY = CONFIG["weight_decay"]
LABEL_SMOOTHING = CONFIG["label_smoothing"]

# Best model saved to disk whenever val accuracy improves
BEST_MODEL_PATH = f"models/best_audio_cnn_{MODE}.pth"

print(f"🔧 CONFIG → {MODE.upper()} MODE")
print(f"   Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR} | Patience: {PATIENCE}")
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
    val_fraction=VAL_FRACTION,
)

print(f"✅ Data loaded: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val / {len(test_loader.dataset)} test")
print(f"   Classes: {num_classes}")

# ==================== MODEL, LOSS, OPTIMIZER, SCHEDULER ====================
model = SimpleAudioCNN(num_classes=num_classes)

# Label smoothing prevents overconfident predictions (same as NSFW)
loss_function = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

# AdamW handles weight decay better than plain Adam
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Cosine annealing: LR starts high, gradually decreases → smoother convergence
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# ==================== MIXED PRECISION ====================
# AMP makes training faster + less memory on CUDA GPUs
# Note: AMP only works reliably on CUDA, not MPS
use_amp = is_cuda
scaler = GradScaler() if use_amp else None

# ==================== TRANSFORMATIONS ====================
# MelSpectrogram + AmplitudeToDB run on GPU as batch transform (Option B)
# Applied AFTER DataLoader, in the training loop — much faster than per-sample
train_transform, eval_transform = get_transformations()
train_transform = train_transform.to(device)
eval_transform = eval_transform.to(device)

# ==================== TRAINING FUNCTIONS ====================

def train_epoch(model, train_loader, loss_function, optimizer, device, train_transform, scaler, use_amp):
    """
    Train one epoch.
    
    Flow per batch:
      1. Move raw waveforms to device [batch, 1, max_samples]
      2. Apply MelSpectrogram + AmplitudeToDB on GPU → [batch, 1, 128, time]
      3. Forward pass through CNN
      4. Backward pass + optimizer step
    """
    model.train()
    running_loss = 0.0

    for waveforms, labels in train_loader:
        waveforms, labels = waveforms.to(device), labels.to(device)

        # Transform raw waveforms → mel spectrograms on GPU
        waveforms = train_transform(waveforms)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast(device_type="cuda"):
                outputs = model(waveforms)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(waveforms)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * waveforms.size(0)

    return running_loss / len(train_loader.dataset)


def validate_epoch(model, val_loader, loss_function, device, eval_transform):
    """
    Validate one epoch.
    
    torch.no_grad() — don't build computation graph (saves memory, faster)
    model.eval() — disable dropout, use running batchnorm stats
    """
    model.eval()
    running_loss = 0.0
    correct = total = 0

    with torch.no_grad():
        for waveforms, labels in val_loader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            waveforms = eval_transform(waveforms)

            outputs = model(waveforms)
            loss = loss_function(outputs, labels)

            running_loss += loss.item() * waveforms.size(0)
            # torch.max returns (values, indices) — we want the indices (predicted class)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return (running_loss / len(val_loader.dataset)), (100.0 * correct / total)


# ==================== TRAINING LOOP ====================

def training_loop(model, train_loader, val_loader, loss_function, optimizer, scheduler,
                  num_epochs, device, train_transform, eval_transform, scaler, use_amp):
    """
    Full training loop with:
      • Early stopping (patience from CONFIG)
      • Best model checkpointing (saved to disk immediately when improved)
      • Cosine LR scheduling
      • Progress output per epoch
    """
    model.to(device)

    best_val_accuracy = 0.0
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    train_losses, val_losses, val_accuracies = [], [], []

    print("\n" + "=" * 70)
    print(f"🚀 TRAINING STARTED — {MODE.upper()} MODE")
    print(f"   Device: {device} | Epochs: {num_epochs} | Best model → {BEST_MODEL_PATH}")
    print("=" * 70)

    for epoch in range(num_epochs):
        # Train
        epoch_loss = train_epoch(model, train_loader, loss_function, optimizer, device,
                                 train_transform, scaler, use_amp)
        # Validate
        epoch_val_loss, epoch_accuracy = validate_epoch(model, val_loader, loss_function, device,
                                                        eval_transform)

        train_losses.append(epoch_loss)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_accuracy)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_accuracy:6.2f}% | "
              f"LR: {current_lr:.6f}")

        # Step the LR scheduler
        scheduler.step()

        # === Save best model immediately when val accuracy improves ===
        if epoch_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_accuracy
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())

            # Save checkpoint to disk — even if training crashes, best model is safe
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': num_classes,
                'val_accuracy': best_val_accuracy,
                'epoch': best_epoch,
                'mode': MODE,
            }, BEST_MODEL_PATH)

            print(f"  → ✅ New best model saved ({best_val_accuracy:.2f}% at epoch {best_epoch})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹️ Early stopping: {patience_counter} epochs without improvement")
                break

    # Load best weights back for plotting / further use
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, [train_losses, val_losses, val_accuracies]


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

    # Plot training curves (same helper as NSFW)
    try:
        helper_utils.plot_training_metrics(training_metrics)
    except Exception as e:
        print(f"⚠️ Plotting failed: {e}")

    # Evaluate on test set
    test_loss, test_acc = validate_epoch(trained_model, test_loader, loss_function, device, eval_transform)
    print(f"\n🎯 Test Set: Loss={test_loss:.4f}, Accuracy={test_acc:.2f}%")
    print(f"   Best model saved to: {BEST_MODEL_PATH}")
