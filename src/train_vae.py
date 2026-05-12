"""
train_vae.py — Phase 4: Conditional VAE Training (From Scratch)
===============================================================

Trains the VAE entirely from random initialization — no pretrained weights.

BEST PRACTICES IMPLEMENTED:

  1. KL ANNEALING FROM EPOCH 0 (not a flat β=0 warmup)
     - The "β=0 warmup" is a finetune trick to protect pretrained weights.
     - For from-scratch, the encoder is random — it NEEDS gradient signal.
     - Instead, β starts very small (1e-6) and grows exponentially.
     - This gives tiny KL pressure immediately (stabilizes training)
       while still prioritizing reconstruction in early epochs.

  2. LEARNING RATE WARMUP
     - Random weights → huge initial gradients → instability.
     - LR ramps from 0 → target over lr_warmup_epochs.
     - Then cosine decay takes over.

  3. ADAM (not AdamW)
     - AdamW's weight decay penalizes all weights uniformly.
     - KL divergence already acts as a regularizer on the bottleneck.
     - Weight decay + KL = double-regularization → hurts performance.
     - Adam is the standard optimizer for VAEs in literature.

  4. KL FREE BITS (optional, configurable)
     - Each latent dimension must have KL ≥ free_bits to prevent
       "dead" dimensions that carry zero information (posterior collapse).
     - Set to 0 to disable. Default 0.1 per dimension works well.

  5. EXPONENTIAL β SCHEDULE (not linear ramp)
     - Linear ramp: β grows at constant speed.
     - Exponential: β grows slowly at first, then accelerates.
     - Matches how the model actually learns: slow start, then faster.
     - Formula: β = BETA × (1 - exp(-k × epoch / ramp_epochs))
       where k = 5 controls the curve steepness.

  6. GRADIENT CLIPPING (max_norm=1.0)
     - Prevents exploding gradients during KL ramp.

  7. COSINE LR SCHEDULER (after warmup)
     - Smooth LR decay → fine-tuning in later epochs.

EARLY STOPPING:
  Tracks MSE (not total loss) because total loss changes dramatically
  as β ramps up.

COMPARED TO finetune_vae.py:
  • train_vae.py        → trains ALL layers from random init
  • finetune_vae.py     → loads autoencoder weights, freezes encoder/decoder
                          during warmup, then fine-tunes

COURSE REFERENCE:
  • train_autoencoder.py — same structure
  • L3-M2 stable_diffusion — VAE math
"""

import math
import os
import warnings
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from data_loader import get_dataloaders, get_transformations
from vae import SimpleAudioVAE
import helper_utils

warnings.filterwarnings("ignore")


# ==================== CONFIG (EDIT ONLY THIS SECTION) ====================
CONFIG = {
    "mode": "train",                      # "test" = fast dev, "train" = full training
    "device": "auto",                    # "auto", "cuda", "mps", or "cpu"
    "train_fraction": 0.6,
    "val_fraction": 0.2,
    "lr": 1e-3,
    "lr_warmup_epochs": 3,               # LR ramps 0→target over N epochs
    "latent_dim": 1024,
    "embed_dim": 64,                     # class embedding size
    "beta": 0.01,                        # Target KL weight (higher = better organized latent space)
    "beta_free_epochs": 5,                # Epochs with β=0 (quick MSE head start)
    "beta_ramp_epochs": 15,              # β ramps over N epochs AFTER free epochs
    "beta_schedule": "exponential",      # "exponential" or "linear"
    "beta_k": 3,                         # Curve steepness for exponential
    "free_bits": 0.1,                     # Prevent dead latent dimensions
    "class_loss_weight": 0.5,            # γ — 5x stronger class supervision!
    "classifier_path": "models/best_audio_cnn_train.pth",  # Pretrained classifier for class loss
    "optimizer": "Adam",                 # "Adam" recommended for VAEs (not AdamW)
    "scheduler": "CosineAnnealingLR",

    "test": {
        "num_epochs": 5,
        "batch_size": 16,
        "num_workers": 1,
    },

    "train": {
        "num_epochs": 30,                # from-scratch: tight schedule
        "batch_size": 16,
        "num_workers": 4,
    }
}

# ==================== APPLY CONFIG ====================
MODE = CONFIG["mode"]
SETTINGS = CONFIG[MODE]

NUM_EPOCHS = SETTINGS["num_epochs"]
BATCH_SIZE = SETTINGS["batch_size"]
NUM_WORKERS = SETTINGS["num_workers"]
TRAIN_FRACTION = CONFIG["train_fraction"]
VAL_FRACTION = CONFIG["val_fraction"]
LR = CONFIG["lr"]
LR_WARMUP_EPOCHS = CONFIG["lr_warmup_epochs"]
LATENT_DIM = CONFIG["latent_dim"]
EMBED_DIM = CONFIG["embed_dim"]
BETA = CONFIG["beta"]
BETA_FREE_EPOCHS = CONFIG["beta_free_epochs"]
BETA_RAMP_EPOCHS = CONFIG["beta_ramp_epochs"]
BETA_SCHEDULE = CONFIG["beta_schedule"]
BETA_K = CONFIG["beta_k"]
FREE_BITS = CONFIG["free_bits"]
CLASS_LOSS_WEIGHT = CONFIG["class_loss_weight"]
CLASSIFIER_PATH = CONFIG["classifier_path"]

BEST_MODEL_PATH = f"models/best_vae_scratch_{MODE}.pth"

print(f"🔧 CONFIG → {MODE.upper()} MODE (FROM SCRATCH — BEST PRACTICES)")
print(f"   Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR} (warmup: {LR_WARMUP_EPOCHS})")
print(f"   Latent dim: {LATENT_DIM} | Embed dim: {EMBED_DIM}")
print(f"   β: β=0 for {BETA_FREE_EPOCHS} epochs → exp ramp over {BETA_RAMP_EPOCHS} epochs → target={BETA}")
print(f"   Free bits: {FREE_BITS} | Class loss γ: {CLASS_LOSS_WEIGHT}")
print(f"   No pretrained weights — training VAE from random initialization")
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

# ==================== MODEL ====================
model = SimpleAudioVAE(latent_dim=LATENT_DIM, num_classes=num_classes, embed_dim=EMBED_DIM)

# ── Apply Xavier/Kaiming initialization ──
#
# WHY: Default PyTorch init uses Kaiming Uniform for conv layers, which
# is designed for ReLU. Our encoder/decoder already uses ReLU, so the
# default is fine for those. But fc_mu and fc_log_var are Linear layers
# — Xavier (Glorot) init gives better gradient flow for tanh/sigmoid-like
# bottlenecks.
#
# We also init fc_mu/fc_log_var with tiny weights to prevent KL explosion.
#
def init_weights(m):
    if isinstance(m, nn.Linear):
        if m in (model.fc_mu, model.fc_log_var):
            # Tiny init for bottleneck — prevents KL explosion
            nn.init.normal_(m.weight, std=0.001)
            nn.init.zeros_(m.bias)
        else:
            # Xavier for other Linear layers
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        # Kaiming for conv layers (designed for ReLU)
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.ConvTranspose2d):
        # Kaiming for conv transpose
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        # Small random init for embeddings
        nn.init.normal_(m.weight, std=0.02)

model.apply(init_weights)

print(f"✅ VAE initialized from scratch (Xavier Linear, Kaiming Conv, tiny bottleneck)")
print(f"   Strategy: exp β annealing from epoch 0 + LR warmup + class supervision")

# ── Load pretrained classifier for supervision loss ──
# Freeze it — gradients flow THROUGH to VAE but classifier weights don't update
from model import SimpleAudioCNN
classifier = SimpleAudioCNN(num_classes=num_classes)
if os.path.exists(CLASSIFIER_PATH):
    cls_ckpt = torch.load(CLASSIFIER_PATH, map_location=device, weights_only=True)
    classifier.load_state_dict(cls_ckpt["model_state_dict"])
    classifier.to(device)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    print(f"✅ Classifier loaded: epoch={cls_ckpt.get('epoch', '?')}, "
          f"val_acc={cls_ckpt.get('val_accuracy', '?')}")
else:
    classifier = None
    print(f"⚠️  No classifier at {CLASSIFIER_PATH} — class loss disabled")

# ==================== LOSS, OPTIMIZER, SCHEDULER ====================

# VAE uses MSE for reconstruction — same as autoencoder
reconstruction_loss = nn.MSELoss()

# ── Adam (not AdamW) — recommended for VAEs ──
#
# WHY: AdamW decouples weight decay from gradient computation.
# Weight decay penalizes ALL weights uniformly.
# But KL divergence already acts as a regularizer on the bottleneck.
# Two regularizers fighting = degraded performance.
# Adam is the standard in VAE literature (Kingma & Welling 2014).
#
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - LR_WARMUP_EPOCHS, eta_min=1e-6)

# ==================== MIXED PRECISION ====================
use_amp = is_cuda
scaler = GradScaler() if use_amp else None

# ==================== TRANSFORMATIONS ====================
train_transform, eval_transform = get_transformations()
train_transform = train_transform.to(device)
eval_transform = eval_transform.to(device)


# ==================== BETA SCHEDULE ====================

def get_beta(epoch):
    """
    Calculate β for a given epoch.

    Three phases:
      1. Free (epoch < free_epochs): β = 0 (MSE only)
      2. Ramp (free <= epoch < free+ramp): β grows 0 → target
      3. Full (epoch >= free+ramp): β = target

    Two schedule options for ramp:

    1. Exponential (recommended):
       β = BETA × (1 - exp(-k × progress))
       Starts near 0, grows slowly at first, then accelerates.

    2. Linear:
       β = BETA × progress
       Grows at constant speed.

    Example (BETA=0.005, BETA_K=3, free=10, ramp=30):
      Epoch  5: β = 0.00000  (still free phase)
      Epoch 10: β = 0.00000  (end of free phase)
      Epoch 15: β = 0.00197  (5 epochs into ramp, ~39%)
      Epoch 20: β = 0.00316  (10 into ramp, ~63%)
      Epoch 30: β = 0.00433  (20 into ramp, ~87%)
      Epoch 40: β = 0.00500  (target reached)
    """
    if epoch < BETA_FREE_EPOCHS:
        return 0.0

    ramp_epoch = epoch - BETA_FREE_EPOCHS
    if ramp_epoch >= BETA_RAMP_EPOCHS:
        return BETA

    if BETA_SCHEDULE == "exponential":
        return BETA * (1 - math.exp(-BETA_K * ramp_epoch / BETA_RAMP_EPOCHS))
    else:  # linear
        return BETA * (ramp_epoch / BETA_RAMP_EPOCHS)


# ==================== LR WARMUP ====================

def set_lr_warmup(optimizer, epoch):
    """
    Ramp LR from 0 → target over lr_warmup_epochs.

    WHY: Random weights → huge initial gradients → instability.
    LR warmup gives the model time to find a stable direction
    before learning at full speed.
    """
    if epoch < LR_WARMUP_EPOCHS:
        lr = LR * (epoch + 1) / LR_WARMUP_EPOCHS
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    return None


# ==================== VAE LOSS FUNCTION ====================

def vae_loss(reconstructed, target, mu, log_var, beta, free_bits=0.0,
            classifier=None, labels=None, class_loss_weight=0.0):
    """
    VAE loss = MSE + β·KL + γ·CrossEntropy(classifier(recon), labels)

    The classifier term (γ) pushes the decoder to produce class-recognizable
    outputs. The classifier is frozen — gradients flow through it to the VAE
    but the classifier weights don't change.
    """
    # Reconstruction
    recon_loss = reconstruction_loss(reconstructed, target)

    # KL divergence with clamped log_var for numerical stability
    log_var_clamped = torch.clamp(log_var, min=-10, max=10)
    kl_per_dim = -0.5 * (1 + log_var_clamped - mu.pow(2) - log_var_clamped.exp())

    if free_bits > 0:
        kl_per_dim = torch.max(kl_per_dim, torch.full_like(kl_per_dim, free_bits))

    kl_loss = torch.mean(torch.sum(kl_per_dim, dim=1))

    # Combined
    total = recon_loss + beta * kl_loss

    # Classification supervision loss
    class_loss_val = 0.0
    if classifier is not None and labels is not None and class_loss_weight > 0:
        class_logits = classifier(reconstructed)
        class_loss = F.cross_entropy(class_logits, labels)
        class_loss_val = class_loss.item()
        total = total + class_loss_weight * class_loss

    return total, recon_loss.item(), kl_loss.item(), class_loss_val


# ==================== TRAINING FUNCTIONS ====================

def train_epoch(model, train_loader, optimizer, device, train_transform,
                scaler, use_amp, beta, free_bits, classifier=None,
                class_loss_weight=0.0, pbar=None):
    """Train one epoch."""
    model.train()
    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0

    for batch_idx, (waveforms, labels) in enumerate(train_loader):
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        spectrograms = train_transform(waveforms)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast(device_type="cuda"):
                reconstructed, mu, log_var = model(spectrograms, labels)
                loss, recon_val, kl_val, cls_val = vae_loss(
                    reconstructed, spectrograms, mu, log_var, beta, free_bits,
                    classifier=classifier, labels=labels, class_loss_weight=class_loss_weight
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            reconstructed, mu, log_var = model(spectrograms, labels)
            loss, recon_val, kl_val, cls_val = vae_loss(
                reconstructed, spectrograms, mu, log_var, beta, free_bits,
                classifier=classifier, labels=labels, class_loss_weight=class_loss_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * spectrograms.size(0)
        running_recon += recon_val * spectrograms.size(0)
        running_kl += kl_val * spectrograms.size(0)

        if pbar:
            pbar.update_batch(batch_idx + 1, postfix_dict={
                "loss": f"{loss.item():.4f}",
                "mse": f"{recon_val:.4f}",
                "kl": f"{kl_val:.2f}",
                "β": f"{beta:.5f}",
            })

    n = len(train_loader.dataset)
    return running_loss / n, running_recon / n, running_kl / n


def validate_epoch(model, val_loader, device, eval_transform, beta, free_bits,
                  classifier=None, class_loss_weight=0.0, pbar=None):
    """Validate one epoch — returns total loss, recon loss, KL loss, class loss."""
    model.eval()
    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0
    running_cls = 0.0

    with torch.no_grad():
        for batch_idx, (waveforms, labels) in enumerate(val_loader):
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            spectrograms = eval_transform(waveforms)

            reconstructed, mu, log_var = model(spectrograms, labels)
            loss, recon_val, kl_val, cls_val = vae_loss(
                reconstructed, spectrograms, mu, log_var, beta, free_bits,
                classifier=classifier, labels=labels, class_loss_weight=class_loss_weight
            )

            running_loss += loss.item() * spectrograms.size(0)
            running_recon += recon_val * spectrograms.size(0)
            running_kl += kl_val * spectrograms.size(0)
            running_cls += cls_val * spectrograms.size(0)

            if pbar:
                pbar.update_batch(batch_idx + 1)

    n = len(val_loader.dataset)
    return running_loss / n, running_recon / n, running_kl / n, running_cls / n


# ==================== TRAINING LOOP ====================

def training_loop(model, train_loader, val_loader, optimizer, scheduler,
                  num_epochs, device, train_transform, eval_transform, scaler, use_amp,
                  classifier=None, class_loss_weight=0.0):
    """
    Full training loop for from-scratch VAE training.

    Three things change over time:
      - LR: 0 → target (warmup) → cosine decay
      - β:  tiny → target (exponential ramp)
      - Class loss: constant γ weight applied to CrossEntropy(classifier(recon), label)
    """
    model.to(device)

    train_losses, val_losses = [], []
    train_recons, val_recons = [], []
    train_kls, val_kls = [], []

    print("\n" + "=" * 70)
    print(f"🚀 VAE FROM-SCRATCH TRAINING — {MODE.upper()} MODE")
    print(f"   Device: {device} | Epochs: {num_epochs} | Batch: {BATCH_SIZE}")
    print(f"   LR: warmup {LR_WARMUP_EPOCHS} epochs → cosine decay")
    print(f"   β: β=0 for {BETA_FREE_EPOCHS} epochs → {BETA_SCHEDULE} ramp over {BETA_RAMP_EPOCHS} epochs → {BETA}")
    print(f"   Class loss γ: {class_loss_weight}" if class_loss_weight > 0 else "   Class loss: disabled")
    print(f"   All layers unfrozen from start")
    print(f"   Saving last model → {BEST_MODEL_PATH}")
    print("=" * 70)

    for epoch in range(num_epochs):

        # ── LR warmup ──
        warmup_lr = set_lr_warmup(optimizer, epoch)
        if warmup_lr is not None:
            # During warmup, skip scheduler step (we control LR manually)
            using_warmup = True
        else:
            using_warmup = False

        # ── β schedule ──
        beta_val = get_beta(epoch)

        train_pbar = helper_utils.NestedProgressBar(
            total_epochs=num_epochs,
            total_batches=len(train_loader),
            mode="train",
        )
        train_pbar.update_epoch(epoch + 1)

        # ── Train ──
        epoch_loss, epoch_recon, epoch_kl = train_epoch(
            model, train_loader, optimizer, device, train_transform,
            scaler, use_amp, beta_val, FREE_BITS,
            classifier=classifier, class_loss_weight=class_loss_weight, pbar=train_pbar
        )
        train_pbar.batch_bar.close()

        # ── Validate ──
        val_pbar = helper_utils.NestedProgressBar(
            total_epochs=1,
            total_batches=len(val_loader),
            mode="eval",
        )
        epoch_val_loss, epoch_val_recon, epoch_val_kl, epoch_val_cls = validate_epoch(
            model, val_loader, device, eval_transform, beta_val, FREE_BITS,
            classifier=classifier, class_loss_weight=class_loss_weight, pbar=val_pbar
        )
        val_pbar.close()

        # Track all metrics
        train_losses.append(epoch_loss)
        val_losses.append(epoch_val_loss)
        train_recons.append(epoch_recon)
        val_recons.append(epoch_val_recon)
        train_kls.append(epoch_kl)
        val_kls.append(epoch_val_kl)
        # Note: val_cls is only tracked for logging, not used for early stopping

        current_lr = optimizer.param_groups[0]['lr']

        ramp_end = BETA_FREE_EPOCHS + BETA_RAMP_EPOCHS
        if epoch < BETA_FREE_EPOCHS:
            phase = "β=0"
        elif epoch < ramp_end:
            phase = "β ramp"
        else:
            phase = "β fixed"
        train_pbar.update_epoch(epoch + 1, postfix_dict={
            "phase": phase,
            "train": f"{epoch_loss:.4f}",
            "val_mse": f"{epoch_val_recon:.4f}",
            "kl": f"{epoch_val_kl:.2f}",
            "β": f"{beta_val:.5f}",
            "lr": f"{current_lr:.1e}",
        })

        # Only step the cosine scheduler after warmup is done
        if not using_warmup:
            scheduler.step()

    # ── Save last model (epoch 49) ── the best generative VAE ──
    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_dim': LATENT_DIM,
        'embed_dim': EMBED_DIM,
        'num_classes': num_classes,
        'beta': BETA,
        'mode': MODE,
        'type': 'scratch',
    }, BEST_MODEL_PATH)
    print(f"\n💾 Last epoch model saved to: {BEST_MODEL_PATH}")

    return model, [train_losses, val_losses, train_recons, val_recons, train_kls, val_kls]


# ==================== GENERATION DEMO ====================

def generate_demo(model, device, eval_transform):
    """
    After training, generate sample sounds from each class.
    """
    print("\n" + "=" * 70)
    print("🎨 GENERATION DEMO — Creating new animal sounds!")
    print("=" * 70)

    class_names = ['Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen', 'Noise']

    model.eval()
    for class_idx, class_name in enumerate(class_names):
        spectrograms = model.sample(
            label=class_idx,
            num_samples=3,
            device=device,
        )
        print(f"  {class_name}: generated {spectrograms.shape[0]} spectrograms "
              f"shape={spectrograms.shape[2:]}")

    print("\n  🔀 Interpolation demo: Dog → Cat (5 steps)")
    print("     (Requires test samples — will show in Phase 5 evaluation)")


# ==================== RUN ====================
if __name__ == "__main__":
    trained_model, training_metrics = training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        device=device,
        train_transform=train_transform,
        eval_transform=eval_transform,
        scaler=scaler,
        use_amp=use_amp,
        classifier=classifier,
        class_loss_weight=CLASS_LOSS_WEIGHT,
    )

    # Plot training curves
    try:
        train_losses, val_losses = training_metrics[0], training_metrics[1]
        helper_utils.plot_training_metrics([train_losses, val_losses, val_losses])
    except Exception as e:
        print(f"⚠️ Plotting failed: {e}")

    # Evaluate on test set
    test_loss, test_recon, test_kl, test_cls = validate_epoch(
        trained_model, test_loader, device, eval_transform, BETA, FREE_BITS,
        classifier=classifier, class_loss_weight=CLASS_LOSS_WEIGHT
    )
    print(f"\n🎯 Test Set: Total={test_loss:.6f} | MSE={test_recon:.6f} | KL={test_kl:.6f} | Cls={test_cls:.4f}")
    print(f"   Model saved to: {BEST_MODEL_PATH}")

    # Generate demo sounds!
    generate_demo(trained_model, device, eval_transform)
