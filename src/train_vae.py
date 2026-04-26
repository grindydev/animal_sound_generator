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
  │  Early stopping    │  val MSE (lower=better)   │  val MSE (lower=better)       │
  │  After training    │  (nothing — can't generate)│  Generate new sounds!         │
  └────────────────────┴──────────────────────────┴───────────────────────────────┘

TRAINING STRATEGY (3 phases):

  Phase A — Warmup (epochs 1 to warmup_epochs):
    - Encoder + decoder FROZEN (pretrained autoencoder weights)
    - Only train: fc_mu, fc_log_var, class_embed, class_project
    - β starts at 0, slowly increases
    - WHY: Let the new VAE heads learn to produce μ near 0 and σ near 1
      while keeping the pretrained feature extraction intact.
      Without freezing, the KL gradient would destroy the encoder.

  Phase B — Fine-tune (epochs warmup_epochs+1 to end):
    - All layers unfrozen
    - β at target value
    - WHY: Now that the bottleneck is stable, fine-tune everything together.

  Early stopping tracks MSE (not total loss) because total loss is
  dominated by β × KL which changes during warmup.

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
    "mode": "train",                      # "test" = fast dev, "train" = full training
    "device": "auto",                    # "auto", "cuda", "mps", or "cpu"
    "train_fraction": 0.6,
    "val_fraction": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-3,
    "latent_dim": 1024,
    "embed_dim": 64,                     # class embedding size
    "beta": 0.01,                        # Target KL weight after warmup
    "beta_start": 0.0,                   # β annealing: start with no KL
    "warmup_epochs": 10,                 # frozen encoder/decoder duration
    "ramp_epochs": 30,                   # β ramps 0→target over N epochs AFTER warmup
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",

    "test": {
        "num_epochs": 5,
        "batch_size": 16,
        "patience": 5,
        "num_workers": 1,
    },

    "train": {
        "num_epochs": 100,
        "batch_size": 16,
        "patience": 20,
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
RAMP_EPOCHS = CONFIG["ramp_epochs"]

BEST_MODEL_PATH = f"models/best_vae_{MODE}.pth"

print(f"🔧 CONFIG → {MODE.upper()} MODE")
print(f"   Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR} | Patience: {PATIENCE}")
print(f"   Latent dim: {LATENT_DIM} | Embed dim: {EMBED_DIM} | β: 0→{BETA} over {WARMUP_EPOCHS}+{RAMP_EPOCHS} epochs")
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

# ── Load pretrained autoencoder weights ──
#
# We copy encoder conv blocks + decoder conv blocks + fc_decode from the
# autoencoder. These are the "feature extraction" and "spectrogram generation"
# parts that transfer directly.
#
# We do NOT copy fc_encode → fc_mu because:
#   - fc_encode was trained without KL constraint → produces large μ values (~60)
#   - Large μ → KL = 0.5 × Σ(μ²) ≈ 2,000,000
#   - Even tiny β creates enormous gradients → destroys training
#   - Instead, fc_mu starts random and learns to produce μ near 0
#
# During warmup (first 10 epochs), encoder + decoder are FROZEN.
# Only fc_mu, fc_log_var, class_embed, class_project are trained.
# This lets the new VAE heads adapt to the pretrained features
# without the KL gradient destroying the encoder.
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

    # ── Initialize fc_mu with tiny weights so μ starts near 0 ──
    #
    # WHY: The pretrained encoder produces large features (after BN + ReLU through
    # 4 layers). A random Linear(35840, 1024) with default init produces
    # μ ≈ ±50 → KL = 0.5 × 1024 × 50² ≈ 1.3M. Even β=0.001 × 1.3M = 1,300
    # gradient → instant destruction.
    #
    # Fix: init with very small weights so μ starts near 0 → KL ≈ small.
    # The model will learn the right μ scale during warmup (β=0).
    #
    with torch.no_grad():
        model.fc_mu.weight.normal_(std=0.001)
        model.fc_mu.bias.zero_()
        model.fc_log_var.weight.normal_(std=0.001)
        model.fc_log_var.bias.zero_()

    print(f"✅ Loaded pretrained autoencoder weights from {ae_checkpoint_path}")
    print(f"   Copied:   {len(copied)} layers (encoder + decoder + fc_decode)")
    print(f"   Skipped:  {len(skipped)} layers (fc_encode → not copied, VAE heads init tiny)")
    print(f"   fc_mu:    tiny init (std=0.001) → μ starts near 0 → KL starts small")
    print(f"   Source:   epoch {ae_ckpt.get('epoch', '?')}, val_mse={ae_ckpt.get('val_mse', '?')}")
else:
    print(f"⚠️  No pretrained autoencoder found at {ae_checkpoint_path}")
    print(f"   Run 'python src/train_autoencoder.py' first for best results")
    print(f"   Training VAE from scratch (still works, just needs more epochs)")

# ==================== LOSS, OPTIMIZER, SCHEDULER ====================

# VAE uses MSE for reconstruction — same as autoencoder
reconstruction_loss = nn.MSELoss()

# Optimizer + scheduler
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

    The log_var is clamped to [-10, 10] to prevent float16 overflow under AMP.
    Without clamping, exp(log_var) overflows float16 when log_var > 11.

    Returns:
        total_loss, recon_loss_val, kl_loss_val
    """
    # Reconstruction
    recon_loss = reconstruction_loss(reconstructed, target)

    # KL divergence with clamped log_var for numerical stability
    log_var_clamped = torch.clamp(log_var, min=-10, max=10)
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + log_var_clamped - mu.pow(2) - log_var_clamped.exp(), dim=1)
    )

    # Combined
    total = recon_loss + beta * kl_loss

    return total, recon_loss.item(), kl_loss.item()


# ==================== FREEZE / UNFREEZE HELPERS ====================

def freeze_pretrained(model):
    """
    Freeze encoder + decoder during warmup.

    WHY: The pretrained encoder/decoder already work well. If we train them
    from epoch 1, the KL gradient (even with small β) will distort the feature
    extraction before the VAE heads have learned to produce reasonable μ/σ.

    By freezing, the VAE heads (fc_mu, fc_log_var, class_embed, class_project)
    learn to map the pretrained features to a N(0,1)-compatible latent space
    without disturbing the feature extraction.
    """
    for param in model.encode.parameters():
        param.requires_grad = False
    for param in model.decode.parameters():
        param.requires_grad = False
    for param in model.fc_decode.parameters():
        param.requires_grad = False
    print("❄️  Frozen: encoder + decoder + fc_decode (only training VAE heads)")


def unfreeze_all(model):
    """
    Unfreeze all layers after warmup.

    WHY: Now that the VAE heads produce reasonable μ/σ, we can fine-tune
    the entire model end-to-end. The encoder/decoder adapt to work with
    the new probabilistic bottleneck.
    """
    for param in model.parameters():
        param.requires_grad = True
    print("🔥 Unfrozen: all layers (full fine-tuning)")


# ==================== TRAINING FUNCTIONS ====================

def train_epoch(model, train_loader, optimizer, device, train_transform,
                scaler, use_amp, beta, pbar=None):
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
    Full training loop with 3-phase strategy:

      Phase A (warmup): Frozen encoder/decoder, β ramps 0→target
      Phase B (finetune): All unfrozen, β at target
      Early stopping on val MSE (not total loss!)
    """
    model.to(device)

    # ── Phase A: Freeze pretrained layers during warmup ──
    if os.path.exists(ae_checkpoint_path):
        freeze_pretrained(model)

    best_val_mse = float("inf")
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    train_losses, val_losses = [], []
    train_recons, val_recons = [], []
    train_kls, val_kls = [], []

    print("\n" + "=" * 70)
    print(f"🚀 VAE TRAINING — {MODE.upper()} MODE")
    print(f"   Device: {device} | Epochs: {num_epochs}")
    print(f"   β schedule: {WARMUP_EPOCHS} warmup (frozen, β=0) → {RAMP_EPOCHS} ramp → β={BETA}")
    print(f"   Early stopping: val MSE (patience={PATIENCE})")
    print(f"   Best model → {BEST_MODEL_PATH}")
    print("=" * 70)

    for epoch in range(num_epochs):

        # ── β schedule ──
        #
        # Three phases:
        #   Epochs  1-10:  β=0,      frozen (learn VAE heads for reconstruction)
        #   Epochs 11-40:  β 0→0.01, unfrozen (gradual transition, decoder adapts)
        #   Epochs 41+:    β=0.01,   full VAE training
        #
        # WHY 30 epochs for ramp (not 10):
        #   During warmup fc_mu learns large μ (KL≈5M, no penalty since β=0).
        #   When β starts, KL gradient pushes μ toward 0. If β ramps too fast
        #   (10 epochs), μ collapses in 1-2 epochs → decoder suddenly gets
        #   completely different z → MSE jumps 4×. With 30 epochs, μ shrinks
        #   gradually giving the decoder time to adapt to the changing z.
        #
        if epoch < WARMUP_EPOCHS:
            beta = 0.0
        elif epoch < WARMUP_EPOCHS + RAMP_EPOCHS:
            ramp_progress = (epoch - WARMUP_EPOCHS) / RAMP_EPOCHS
            beta = BETA * ramp_progress
        else:
            beta = BETA

        # ── Phase transition: unfreeze after warmup ──
        if epoch == WARMUP_EPOCHS and os.path.exists(ae_checkpoint_path):
            unfreeze_all(model)

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

        phase = "❄️ warmup" if epoch < WARMUP_EPOCHS else "🔥 finetune"
        train_pbar.update_epoch(epoch + 1, postfix_dict={
            "phase": phase,
            "train": f"{epoch_loss:.4f}",
            "val_mse": f"{epoch_val_recon:.4f}",
            "kl": f"{epoch_val_kl:.2f}",
            "β": f"{beta:.5f}",
        })

        scheduler.step()

        # === Early stopping on val MSE (not total loss!) ===
        #
        # WHY track MSE not total loss:
        #   total = MSE + β × KL
        #   During warmup β goes 0→0.01, so total loss keeps increasing
        #   even if MSE improves. Epoch 1 (β=0) would always "win".
        #   MSE is the actual reconstruction quality — that's what we care about.
        #
        if epoch_val_recon < best_val_mse:
            best_val_mse = epoch_val_recon
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())

            torch.save({
                'model_state_dict': model.state_dict(),
                'latent_dim': LATENT_DIM,
                'embed_dim': EMBED_DIM,
                'num_classes': num_classes,
                'beta': BETA,
                'val_mse': best_val_mse,
                'epoch': best_epoch,
                'mode': MODE,
            }, BEST_MODEL_PATH)

            print(f"  → ✅ New best model saved (mse={best_val_mse:.6f}, "
                  f"kl={epoch_val_kl:.2f}, β={beta:.5f} at epoch {best_epoch})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹️ Early stopping: {patience_counter} epochs without MSE improvement")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

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

    # Interpolation demo: dog → cat
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
    )

    # Plot training curves
    try:
        train_losses, val_losses = training_metrics[0], training_metrics[1]
        helper_utils.plot_training_metrics([train_losses, val_losses, val_losses])
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
