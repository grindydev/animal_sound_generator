"""
train_vae.py — Phase 4: Conditional VAE Training Pipeline
==========================================================

COMPARED TO train_autoencoder.py, WHAT CHANGES:

  ┌────────────────────┬──────────────────────────┬───────────────────────────────┐
  │  Aspect            │  Autoencoder (Phase 3)    │  VAE (Phase 4)                │
  ├────────────────────┼──────────────────────────┼───────────────────────────────┤
  │  Model             │  SimpleAudioAutoencoder   │  SimpleAudioVAE               │
  │  forward() input   │  (spectrograms)           │  (spectrograms, labels)       │
  │  forward() output  │  reconstructed             │  reconstructed, μ, log_var   │
  │  Loss              │  MSE only                 │  MSE + β * KL_divergence      │
  │  Early stopping    │  val MSE (lower=better)   │  val total_loss (lower=better)│
  │  After training    │  (nothing — can't generate)│  Generate new sounds!         │
  └────────────────────┴──────────────────────────┴───────────────────────────────┘

THE KEY NEW CONCEPT — KL DIVERGENCE LOSS:

  KL divergence measures how different two probability distributions are.

  We want: q(z|x) ≈ N(0, 1)   (encoder output close to standard normal)
  Why?     So the latent space is SMOOTH and CONTINUOUS:
           - Nearby points → similar sounds
           - Any random point from N(0,1) produces a meaningful sound
           - This is what makes generation possible!

  Formula: KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
  When μ=0 and σ²=1 (perfect standard normal): KL = 0 (minimum)
  The model is penalized when the learned distribution drifts from N(0,1)

  β (beta) controls the TRADEOFF:
    - β too small → good reconstruction, but latent space is messy → can't generate
    - β too large → latent space is very organized, but reconstruction is blurry
    - Typical starting value: 0.01 (reconstruction quality first, organization second)

COURSE REFERENCE:
  • train_autoencoder.py — same structure, this file is adapted from it
  • L3-M2 stable_diffusion — the math behind VAEs and latent spaces
"""

import copy
import os
import warnings
import torch
from torch import nn
from torch import optim
from torch.amp import autocast, GradScaler

from data_loader import get_dataloaders, get_transformations
from vae import SimpleAudioVAE
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
    "embed_dim": 64,                     # NEW: class embedding size
    "beta": 0.01,                        # Target KL weight after warmup
    "beta_start": 0.0,                  # β annealing: start with no KL (pure autoencoder)
    "warmup_epochs": 10,                # β annealing: gradually increase β over N epochs
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",

    "test": {
        "num_epochs": 5,
        "batch_size": 16,
        "patience": 5,
        "num_workers": 1,
    },

    "train": {
        "num_epochs": 80,
        "batch_size": 16,
        "patience": 15,
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
EMBED_DIM = CONFIG["embed_dim"]
BETA = CONFIG["beta"]
BETA_START = CONFIG["beta_start"]
WARMUP_EPOCHS = CONFIG["warmup_epochs"]

BEST_MODEL_PATH = f"models/best_vae_{MODE}.pth"

print(f"🔧 CONFIG → {MODE.upper()} MODE")
print(f"   Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR} | Patience: {PATIENCE}")
print(f"   Latent dim: {LATENT_DIM} | Embed dim: {EMBED_DIM} | β: {BETA_START}→{BETA} over {WARMUP_EPOCHS} epochs")
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
model = SimpleAudioVAE(latent_dim=LATENT_DIM, num_classes=num_classes, embed_dim=EMBED_DIM)

# ── #2: Load pretrained autoencoder encoder weights ──
#
# WHY: The VAE encoder is identical to the autoencoder encoder.
# Instead of learning feature extraction from scratch (slow, harder),
# we start from a trained encoder that already knows how to compress
# spectrograms. Only the VAE-specific parts (μ, log_var, class embedding)
# start random — the model only needs to learn the new probabilistic bottleneck.
#
# WHAT GETS COPIED:
#   encode.*           → encoder conv blocks (feature extraction)
#   decode.*           → decoder conv blocks (spectrogram generation)
#   fc_decode.*        → decoder linear layer
#
# WHAT STAYS RANDOM (VAE-specific, must learn from scratch):
#   fc_mu.*            → mean head (new)
#   fc_log_var.*       → log-variance head (new)
#   class_embed.*      → class embedding table (new)
#   class_project.*    → class projection layer (new)
#
ae_checkpoint_path = f"models/best_autoencoder_{MODE}.pth"
if os.path.exists(ae_checkpoint_path):
    ae_ckpt = torch.load(ae_checkpoint_path, map_location=device, weights_only=True)
    ae_state = ae_ckpt['model_state_dict']
    vae_state = model.state_dict()

    copied, skipped = [], []
    for key in ae_state:
        if key.startswith('encode.') or key.startswith('decode.') or key.startswith('fc_decode.'):
            if key in vae_state and vae_state[key].shape == ae_state[key].shape:
                vae_state[key] = ae_state[key]
                copied.append(key)
            else:
                skipped.append(key)
        else:
            skipped.append(key)

    model.load_state_dict(vae_state)

    print(f"✅ Loaded pretrained autoencoder weights from {ae_checkpoint_path}")
    print(f"   Copied:   {len(copied)} layers ({', '.join(copied[:5])}...)")
    print(f"   Skipped:  {len(skipped)} layers (VAE-specific: fc_mu, fc_log_var, class_embed, class_project)")
    print(f"   Source:   epoch {ae_ckpt.get('epoch', '?')}, val_mse={ae_ckpt.get('val_mse', '?')}")
else:
    print(f"⚠️  No pretrained autoencoder found at {ae_checkpoint_path}")
    print(f"   Run 'python src/train_autoencoder.py' first for best results")
    print(f"   Training VAE from scratch (still works, just needs more epochs)")

# VAE uses MSE for reconstruction — same as autoencoder
reconstruction_loss = nn.MSELoss()

# Optimizer + scheduler — same as autoencoder
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# ==================== MIXED PRECISION ====================
use_amp = is_cuda
scaler = GradScaler() if use_amp else None

# ==================== TRANSFORMATIONS ====================
train_transform, eval_transform = get_transformations()
train_transform = train_transform.to(device)
eval_transform = eval_transform.to(device)


# ==================== VAE LOSS FUNCTION ====================

def vae_loss(reconstructed, target, mu, log_var, beta):
    """
    VAE loss = reconstruction_loss + beta * KL_divergence.

    Compared to autoencoder which only has MSE, VAE adds KL divergence.

    WHY TWO LOSSES:
      - Reconstruction (MSE):  "make the output look like the input"
        → drives the encoder/decoder to preserve spectrogram details
      - KL divergence:         "keep latent space organized as N(0,1)"
        → ensures we can sample random z at generation time

    WHY β (beta):
      These two losses compete — MSE wants z to encode EVERYTHING (max info),
      KL wants z to be standard normal (min info). β balances them.
      Start small (0.01) so reconstruction quality stays good.

    KL formula: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
      When μ=0, σ²=1:  -0.5 * Σ(1 + 0 - 0 - 1) = 0  ← perfect, no penalty
      When μ=5, σ²=1:  -0.5 * Σ(1 + 0 - 25 - 1) = 12.5  ← far from N(0,1), big penalty

    Args:
        reconstructed: model output [B, 1, 64, W]
        target:        original spectrogram [B, 1, 64, W]
        mu:            latent mean [B, latent_dim]
        log_var:       latent log variance [B, latent_dim]
        beta:          weight for KL loss (default from CONFIG)

    Returns:
        total_loss:     scalar (reconstruction + β * KL)
        recon_loss_val: scalar (MSE only, for logging)
        kl_loss_val:    scalar (KL only, for logging)
    """
    # Reconstruction — same as autoencoder
    recon_loss = reconstruction_loss(reconstructed, target)

    # KL divergence — NEW for VAE
    # sum over latent_dim, mean over batch
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    )

    # Combined
    total = recon_loss + beta * kl_loss

    return total, recon_loss.item(), kl_loss.item()


# ==================== TRAINING FUNCTIONS ====================

def train_epoch(model, train_loader, optimizer, device, train_transform,
                scaler, use_amp, beta, pbar=None):
    """
    Train one epoch.

    DIFFERENCES from autoencoder train_epoch:
      1. model(spectrograms, labels) — labels are NOW used (for conditioning)
      2. model returns (reconstructed, mu, log_var) — need all three
      3. vae_loss() instead of simple MSE — includes KL divergence
    """
    model.train()
    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0

    for batch_idx, (waveforms, labels) in enumerate(train_loader):
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        # Transform raw waveforms → mel spectrograms on GPU
        spectrograms = train_transform(waveforms)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast(device_type="cuda"):
                reconstructed, mu, log_var = model(spectrograms, labels)
                loss, recon_val, kl_val = vae_loss(reconstructed, spectrograms, mu, log_var, beta)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            reconstructed, mu, log_var = model(spectrograms, labels)
            loss, recon_val, kl_val = vae_loss(reconstructed, spectrograms, mu, log_var, beta)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * spectrograms.size(0)
        running_recon += recon_val * spectrograms.size(0)
        running_kl += kl_val * spectrograms.size(0)

        if pbar:
            pbar.update_batch(batch_idx + 1, postfix_dict={
                "loss": f"{loss.item():.4f}",
                "mse": f"{recon_val:.4f}",
                "kl": f"{kl_val:.4f}",
            })

    n = len(train_loader.dataset)
    return running_loss / n, running_recon / n, running_kl / n


def validate_epoch(model, val_loader, device, eval_transform, beta, pbar=None):
    """Validate one epoch — returns total loss, recon loss, KL loss."""
    model.eval()
    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0

    with torch.no_grad():
        for batch_idx, (waveforms, labels) in enumerate(val_loader):
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            spectrograms = eval_transform(waveforms)

            reconstructed, mu, log_var = model(spectrograms, labels)
            loss, recon_val, kl_val = vae_loss(reconstructed, spectrograms, mu, log_var, beta)

            running_loss += loss.item() * spectrograms.size(0)
            running_recon += recon_val * spectrograms.size(0)
            running_kl += kl_val * spectrograms.size(0)

            if pbar:
                pbar.update_batch(batch_idx + 1)

    n = len(val_loader.dataset)
    return running_loss / n, running_recon / n, running_kl / n


# ==================== TRAINING LOOP ====================

def training_loop(model, train_loader, val_loader, optimizer, scheduler,
                  num_epochs, device, train_transform, eval_transform, scaler, use_amp):
    """
    Full training loop — same structure as autoencoder, but tracks 3 losses.
    Early stopping on LOWEST total val loss (reconstruction + β * KL).
    """
    model.to(device)

    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    train_losses, val_losses = [], []
    train_recons, val_recons = [], []
    train_kls, val_kls = [], []

    print("\n" + "=" * 70)
    print(f"🚀 VAE TRAINING — {MODE.upper()} MODE")
    print(f"   Device: {device} | Epochs: {num_epochs}")
    print(f"   β annealing: {BETA_START} → {BETA} over {WARMUP_EPOCHS} epochs")
    print(f"   Best model → {BEST_MODEL_PATH}")
    print("=" * 70)

    for epoch in range(num_epochs):

        # ── #1: β annealing (KL warmup) ──
        #
        # WHY: The VAE has two competing objectives — reconstruction (MSE) and
        # latent space organization (KL). Learning both simultaneously is hard.
        #
        # Solution: learn them sequentially.
        #   Epochs 1-10:  β ≈ 0 → pure autoencoder, learn reconstruction first
        #   Epochs 11-20: β ramps up → gradually introduce KL, organize latent space
        #   Epochs 21+:   β = 0.01 → full VAE training
        #
        # This is the standard training approach for VAEs, used by:
        #   - Stable Diffusion (linear KL warmup over first 10K steps)
        #   - VQ-VAE-2 (progressive training)
        #   - NVIDIA NeMo (default audio VAE training)
        #
        if epoch < WARMUP_EPOCHS:
            beta = BETA_START + (BETA - BETA_START) * (epoch / WARMUP_EPOCHS)
        else:
            beta = BETA
        train_pbar = helper_utils.NestedProgressBar(
            total_epochs=num_epochs,
            total_batches=len(train_loader),
            mode="train",
        )
        train_pbar.update_epoch(epoch + 1)

        # ── Train ──
        epoch_loss, epoch_recon, epoch_kl = train_epoch(
            model, train_loader, optimizer, device, train_transform,
            scaler, use_amp, beta, pbar=train_pbar
        )
        train_pbar.batch_bar.close()

        # ── Validate ──
        val_pbar = helper_utils.NestedProgressBar(
            total_epochs=1,
            total_batches=len(val_loader),
            mode="eval",
        )
        epoch_val_loss, epoch_val_recon, epoch_val_kl = validate_epoch(
            model, val_loader, device, eval_transform, beta, pbar=val_pbar
        )
        val_pbar.close()

        # Track all metrics
        train_losses.append(epoch_loss)
        val_losses.append(epoch_val_loss)
        train_recons.append(epoch_recon)
        val_recons.append(epoch_val_recon)
        train_kls.append(epoch_kl)
        val_kls.append(epoch_val_kl)

        current_lr = scheduler.get_last_lr()[0]

        train_pbar.update_epoch(epoch + 1, postfix_dict={
            "train": f"{epoch_loss:.4f}",
            "val": f"{epoch_val_loss:.4f}",
            "mse": f"{epoch_val_recon:.4f}",
            "kl": f"{epoch_val_kl:.4f}",
            "β": f"{beta:.5f}",
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
                'embed_dim': EMBED_DIM,
                'num_classes': num_classes,
                'beta': BETA,
                'val_loss': best_val_loss,
                'epoch': best_epoch,
                'mode': MODE,
            }, BEST_MODEL_PATH)

            print(f"  → ✅ New best model saved (loss={best_val_loss:.6f}, "
                  f"mse={epoch_val_recon:.4f}, kl={epoch_val_kl:.4f} at epoch {best_epoch})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹️ Early stopping: {patience_counter} epochs without improvement")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, [train_losses, val_losses, train_recons, val_recons, train_kls, val_kls]


# ==================== GENERATION DEMO ====================

def generate_demo(model, device, eval_transform):
    """
    After training, generate sample sounds from each class.
    This is the exciting part — the model creates sounds it has NEVER seen!
    """
    print("\n" + "=" * 70)
    print("🎨 GENERATION DEMO — Creating new animal sounds!")
    print("=" * 70)

    class_names = ['Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen', 'Noise']

    model.eval()
    for class_idx, class_name in enumerate(class_names):
        # Generate 3 samples per class to show diversity
        spectrograms = model.sample(
            label=class_idx,
            num_samples=3,
            device=device,
        )
        print(f"  {class_name}: generated {spectrograms.shape[0]} spectrograms "
              f"shape={spectrograms.shape[2:]}")
        # In Phase 5, we'll convert these to playable audio with Griffin-Lim


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
    )

    # Plot training curves
    try:
        helper_utils.plot_training_metrics(training_metrics)
    except Exception as e:
        print(f"⚠️ Plotting failed: {e}")

    # Evaluate on test set
    test_loss, test_recon, test_kl = validate_epoch(
        trained_model, test_loader, device, eval_transform, BETA
    )
    print(f"\n🎯 Test Set: Total={test_loss:.6f} | MSE={test_recon:.6f} | KL={test_kl:.6f}")
    print(f"   Best model saved to: {BEST_MODEL_PATH}")

    # Generate demo sounds!
    generate_demo(trained_model, device, eval_transform)
