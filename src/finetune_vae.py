"""
finetune_vae.py — Phase 4: Conditional VAE Fine-Tuning Pipeline
===============================================================

Loads pretrained autoencoder weights, then trains the VAE bottleneck on top.

STRATEGY (2 phases):
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

COMPARED TO train_vae.py (from-scratch):
  • train_vae.py        → trains all layers from random init (no pretrained)
  • finetune_vae.py     → loads autoencoder weights, then adapts VAE heads

COURSE REFERENCE:
  • train_autoencoder.py — same structure, this file is adapted from it
  • L3-M2 stable_diffusion — the math behind VAEs and latent spaces
"""

import math
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
    "latent_dim": 1024,
    "embed_dim": 64,                     # class embedding size
    "beta": 0.002,                       # Target KL weight after warmup (lower = more class distinct)
    "free_bits": 0.0,                    # Disabled — let KL flow freely with lower β
    "warmup_epochs": 8,                  # Frozen encoder/decoder, β=0
    "ramp_epochs": 27,                  # β exponential ramp (8+27=35, then 15 epochs full β)
    "beta_k": 3,                         # Curve steepness for exponential ramp
    "class_loss_weight": 0.1,            # γ — weight for classification supervision loss
    "classifier_path": "models/best_audio_cnn_train.pth",
    "optimizer": "Adam",
    "scheduler": "CosineAnnealingLR",

    "test": {
        "num_epochs": 5,
        "batch_size": 16,
        "num_workers": 1,
    },

    "train": {
        "num_epochs": 50,
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
LATENT_DIM = CONFIG["latent_dim"]
EMBED_DIM = CONFIG["embed_dim"]
BETA = CONFIG["beta"]
FREE_BITS = CONFIG["free_bits"]
CLASS_LOSS_WEIGHT = CONFIG["class_loss_weight"]
CLASSIFIER_PATH = CONFIG["classifier_path"]
WARMUP_EPOCHS = CONFIG["warmup_epochs"]
RAMP_EPOCHS = CONFIG["ramp_epochs"]
BETA_K = CONFIG["beta_k"]

BEST_MODEL_PATH = f"models/best_vae_finetune_{MODE}.pth"

print(f"🔧 CONFIG → {MODE.upper()} MODE")
print(f"   Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}")
print(f"   Latent dim: {LATENT_DIM} | β: 0→{BETA} over {WARMUP_EPOCHS}+{RAMP_EPOCHS} epochs (exp ramp, k={BETA_K})")
print(f"   Free bits: {FREE_BITS} | γ={CLASS_LOSS_WEIGHT} | Optimizer: {CONFIG['optimizer']}")
print(f"   Model saved to: {BEST_MODEL_PATH}")

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

# ── Load pretrained classifier for supervision loss ──
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

# Adam (not AdamW) — weight decay conflicts with KL regularization
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# ==================== MIXED PRECISION ====================
use_amp = is_cuda
scaler = GradScaler() if use_amp else None

# ==================== TRANSFORMATIONS ====================
train_transform, eval_transform = get_transformations()
train_transform = train_transform.to(device)
eval_transform = eval_transform.to(device)


# ==================== VAE LOSS FUNCTION ====================

def vae_loss(reconstructed, target, mu, log_var, beta, free_bits=0.0,
            classifier=None, labels=None, class_loss_weight=0.0):
    """
    VAE loss = MSE + β·KL + γ·CrossEntropy(classifier(recon), labels)
    """
    recon_loss = reconstruction_loss(reconstructed, target)

    log_var_clamped = torch.clamp(log_var, min=-10, max=10)
    kl_per_dim = -0.5 * (1 + log_var_clamped - mu.pow(2) - log_var_clamped.exp())
    kl_per_sample = torch.sum(kl_per_dim, dim=1)

    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        kl_loss = torch.mean(torch.sum(kl_per_dim, dim=1))
    else:
        kl_loss = torch.mean(kl_per_sample)

    total = recon_loss + beta * kl_loss

    class_loss_val = 0.0
    if classifier is not None and labels is not None and class_loss_weight > 0:
        class_logits = classifier(reconstructed)
        class_loss = F.cross_entropy(class_logits, labels)
        class_loss_val = class_loss.item()
        total = total + class_loss_weight * class_loss

    return total, recon_loss.item(), kl_loss.item(), class_loss_val


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
                scaler, use_amp, beta, free_bits=0.0, classifier=None,
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
                loss, recon_val, kl_val, cls_val = vae_loss(reconstructed, spectrograms, mu, log_var, beta, free_bits,
                    classifier=classifier, labels=labels, class_loss_weight=class_loss_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            reconstructed, mu, log_var = model(spectrograms, labels)
            loss, recon_val, kl_val, cls_val = vae_loss(reconstructed, spectrograms, mu, log_var, beta, free_bits,
                classifier=classifier, labels=labels, class_loss_weight=class_loss_weight)
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


def validate_epoch(model, val_loader, device, eval_transform, beta, free_bits=0.0,
                  classifier=None, class_loss_weight=0.0, pbar=None):
    """Validate — returns total loss, recon loss, KL loss, class loss."""
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
            loss, recon_val, kl_val, cls_val = vae_loss(reconstructed, spectrograms, mu, log_var, beta, free_bits,
                classifier=classifier, labels=labels, class_loss_weight=class_loss_weight)

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
    Full training loop with 3-phase β schedule:

      Phase A — Warmup (epochs 0 to warmup_epochs-1):
        Encoder + decoder FROZEN (protect pretrained weights)
        β = 0 (MSE only — VAE heads learn to reconstruct)

      Phase B — Ramp (warmup_epochs to warmup_epochs+ramp_epochs-1):
        All layers UNFROZEN
        β grows 0 → target via EXPONENTIAL curve (gentle start)

      Phase C — Full VAE (remaining epochs):
        β = target, full VAE training

    No early stopping — saves the last epoch model (best generative VAE).
    """
    model.to(device)

    # ── Phase A: Freeze pretrained layers during warmup ──
    if os.path.exists(ae_checkpoint_path):
        freeze_pretrained(model)

    train_losses, val_losses = [], []
    train_recons, val_recons = [], []
    train_kls, val_kls = [], []

    print("\n" + "=" * 70)
    print(f"🚀 VAE TRAINING — {MODE.upper()} MODE")
    print(f"   Device: {device} | Epochs: {num_epochs}")
    print(f"   β schedule: {WARMUP_EPOCHS} frozen (β=0) → {RAMP_EPOCHS} exp ramp → β={BETA}")
    print(f"   γ={class_loss_weight} | Optimizer: Adam (no weight decay)")
    print(f"   No early stopping — saving last epoch model")
    print("=" * 70)

    for epoch in range(num_epochs):

        # ── β schedule (exponential ramp) ──
        #
        # Compared to linear: exp starts much gentler, giving the decoder
        # time to adapt as KL pressure slowly increases. Critical for the
        # frozen→unfrozen transition — the pretrained weights need time
        # to adjust without shock.
        if epoch < WARMUP_EPOCHS:
            beta = 0.0
        elif epoch < WARMUP_EPOCHS + RAMP_EPOCHS:
            ramp_epoch = epoch - WARMUP_EPOCHS
            beta = BETA * (1 - math.exp(-BETA_K * ramp_epoch / RAMP_EPOCHS))
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
            scaler, use_amp, beta, FREE_BITS,
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
            model, val_loader, device, eval_transform, beta, FREE_BITS,
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

        current_lr = scheduler.get_last_lr()[0]

        if epoch < WARMUP_EPOCHS:
            phase = "❄️ warmup"
        elif epoch < WARMUP_EPOCHS + RAMP_EPOCHS:
            phase = "β ramp"
        else:
            phase = "β fixed"
        train_pbar.update_epoch(epoch + 1, postfix_dict={
            "phase": phase,
            "train": f"{epoch_loss:.4f}",
            "val_mse": f"{epoch_val_recon:.4f}",
            "kl": f"{epoch_val_kl:.2f}",
            "β": f"{beta:.5f}",
        })

        scheduler.step()

    # ── Save last model — the best generative VAE ──
    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_dim': LATENT_DIM,
        'embed_dim': EMBED_DIM,
        'num_classes': num_classes,
        'beta': BETA,
        'mode': MODE,
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
