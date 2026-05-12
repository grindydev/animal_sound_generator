"""
finetune_vae.py — VAE v2 Fine-Tuning Pipeline.

Loads pretrained ImprovedAutoencoder weights, then trains the ImprovedVAE
with FiLM class conditioning on top.

Strategy:
  Phase A — Warmup: freeze encoder/decoder, train only VAE heads & FiLM
  Phase B — Fine-tune: unfreeze all, full training
"""
import math
import os
import sys
import warnings
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_loader import get_dataloaders, get_transformations
from src.vae.model import ImprovedVAE
from src.vae.autoencoder import ImprovedAutoencoder
import helper_utils

warnings.filterwarnings("ignore")


# ==================== CONFIG ====================
CONFIG = {
    "mode": "train",           # "test" = 5 epoch smoke test | "train" = full training
    "device": "auto",
    "train_fraction": 0.8,     # 80% train (test folded in)
    "val_fraction": 0.2,
    "lr": 5e-4,                # lower LR for fine-tuning
    "latent_dim": 2048,        # 2× larger
    "embed_dim": 128,          # class embedding for FiLM (was 64)
    "beta": 0.01,              # target KL weight
    "free_bits": 0.1,          # prevent dead latent dims
    "warmup_epochs": 5,        # frozen encoder/decoder, β=0
    "ramp_epochs": 15,         # β exponential ramp
    "beta_k": 3,               # curve steepness
    "class_loss_weight": 0.5,  # classifier supervision weight
    "classifier_path": "models/best_audio_cnn_train.pth",
    "optimizer": "Adam",
    "scheduler": "CosineAnnealingLR",
    "ae_checkpoint": "models/best_autoencoder_train.pth",  # pretrained autoencoder

    "test": {
        "num_epochs": 5,
        "batch_size": 8,       # smaller batch for bigger model
        "num_workers": 1,
    },

    "train": {
        "num_epochs": 40,
        "batch_size": 8,
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
AE_CHECKPOINT = CONFIG["ae_checkpoint"]

BEST_MODEL_PATH = f"models/best_vae_finetune_{MODE}.pth"
CHECKPOINT_DIR = f"models/vae_checkpoints/{MODE}"

print(f"🔧 CONFIG → {MODE.upper()} MODE (Improved V2)")
print(f"   Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR} | Latent: {LATENT_DIM}")
print(f"   β: {BETA} | free_bits: {FREE_BITS} | class_loss: {CLASS_LOSS_WEIGHT}")
print(f"   Warmup: {WARMUP_EPOCHS} epochs frozen → ramp {RAMP_EPOCHS} epochs → full β")
print(f"   Split: {TRAIN_FRACTION*100:.0f}% train / {VAL_FRACTION*100:.0f}% val")

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
print(f"✅ Data: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val")

# ==================== MODEL ====================
model = ImprovedVAE(latent_dim=LATENT_DIM, num_classes=num_classes, embed_dim=EMBED_DIM)

# Load pretrained autoencoder weights
if os.path.exists(AE_CHECKPOINT):
    ae_ckpt = torch.load(AE_CHECKPOINT, map_location=device, weights_only=True)
    ae_state = ae_ckpt["model_state_dict"]

    # Map autoencoder weights to VAE (encoder layers have same names)
    vae_state = model.state_dict()
    matched = 0
    for k in vae_state:
        if k in ae_state and vae_state[k].shape == ae_state[k].shape:
            vae_state[k] = ae_state[k]
            matched += 1

    model.load_state_dict(vae_state)
    print(f"✅ Loaded pretrained AE from {AE_CHECKPOINT}")
    print(f"   Matched: {matched} layers")
    print(f"   AE val_mse: {ae_ckpt.get('val_mse', '?')}")
else:
    print(f"⚠️  No AE checkpoint at {AE_CHECKPOINT} — training VAE from scratch")

n_params = sum(p.numel() for p in model.parameters())
print(f"✅ Model: {n_params:,} params ({n_params/1e6:.1f}M)")

# ==================== CLASSIFIER ====================
from model import SimpleAudioCNN
classifier = SimpleAudioCNN(num_classes=num_classes)
if os.path.exists(CLASSIFIER_PATH):
    cls_ckpt = torch.load(CLASSIFIER_PATH, map_location=device, weights_only=True)
    classifier.load_state_dict(cls_ckpt["model_state_dict"])
    classifier.to(device)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    print(f"✅ Classifier loaded (val_acc={cls_ckpt.get('val_accuracy', '?')}%)")
else:
    classifier = None
    print(f"⚠️  No classifier — class loss disabled")

# ==================== TRAINING SETUP ====================
reconstruction_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)

scaler = GradScaler() if use_amp else None

# ==================== CHECKPOINTING ====================

def save_checkpoint(model, optimizer, scheduler, epoch):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"vae_{epoch:06d}.pth")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }, path)


def load_checkpoint(model, optimizer, scheduler):
    if not os.path.isdir(CHECKPOINT_DIR):
        return 0
    files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("vae_")])
    if not files:
        return 0
    path = os.path.join(CHECKPOINT_DIR, files[-1])
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    print(f"   📂 Resumed from {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


train_transform, eval_transform = get_transformations()
train_transform = train_transform.to(device)
eval_transform = eval_transform.to(device)


# ==================== BETA SCHEDULE ====================

def get_beta(epoch):
    if epoch < WARMUP_EPOCHS:
        return 0.0
    ramp_epoch = epoch - WARMUP_EPOCHS
    if ramp_epoch >= RAMP_EPOCHS:
        return BETA
    return BETA * (1 - math.exp(-BETA_K * ramp_epoch / RAMP_EPOCHS))


# ==================== VAE LOSS ====================

def vae_loss(reconstructed, target, mu, log_var, beta, free_bits=0.0,
             classifier=None, labels=None, class_loss_weight=0.0):
    recon_loss = reconstruction_loss(reconstructed, target)

    log_var_clamped = torch.clamp(log_var, min=-10, max=10)
    kl_per_dim = -0.5 * (1 + log_var_clamped - mu.pow(2) - log_var_clamped.exp())

    if free_bits > 0:
        kl_per_dim = torch.max(kl_per_dim, torch.full_like(kl_per_dim, free_bits))

    kl_loss = torch.mean(torch.sum(kl_per_dim, dim=1))
    total = recon_loss + beta * kl_loss

    class_loss_val = 0.0
    if classifier is not None and labels is not None and class_loss_weight > 0:
        class_logits = classifier(reconstructed)
        class_loss = F.cross_entropy(class_logits, labels)
        class_loss_val = class_loss.item()
        total = total + class_loss_weight * class_loss

    return total, recon_loss.item(), kl_loss.item(), class_loss_val


# ==================== TRAINING FUNCTIONS ====================

def train_epoch(model, train_loader, optimizer, device, train_tfm,
                scaler, use_amp, beta, free_bits, classifier, class_loss_weight, pbar=None):
    model.train()
    running_loss, running_recon, running_kl = 0.0, 0.0, 0.0

    for batch_idx, (waveforms, labels) in enumerate(train_loader):
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        spectrograms = train_tfm(waveforms)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast(device_type="cuda"):
                reconstructed, mu, log_var = model(spectrograms, labels)
                loss, recon_val, kl_val, cls_val = vae_loss(
                    reconstructed, spectrograms, mu, log_var, beta, free_bits,
                    classifier=classifier, labels=labels, class_loss_weight=class_loss_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            reconstructed, mu, log_var = model(spectrograms, labels)
            loss, recon_val, kl_val, cls_val = vae_loss(
                reconstructed, spectrograms, mu, log_var, beta, free_bits,
                classifier=classifier, labels=labels, class_loss_weight=class_loss_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * spectrograms.size(0)
        running_recon += recon_val * spectrograms.size(0)
        running_kl += kl_val * spectrograms.size(0)

        if pbar:
            pbar.update_batch(batch_idx + 1, postfix_dict={
                "loss": f"{loss.item():.4f}", "mse": f"{recon_val:.4f}",
                "kl": f"{kl_val:.1f}", "β": f"{beta:.4f}",
            })

    n = len(train_loader.dataset)
    return running_loss / n, running_recon / n, running_kl / n


def validate_epoch(model, val_loader, device, eval_tfm, beta, free_bits,
                   classifier, class_loss_weight, pbar=None):
    model.eval()
    running_loss, running_recon, running_kl, running_cls = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for batch_idx, (waveforms, labels) in enumerate(val_loader):
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            spectrograms = eval_tfm(waveforms)

            reconstructed, mu, log_var = model(spectrograms, labels)
            loss, recon_val, kl_val, cls_val = vae_loss(
                reconstructed, spectrograms, mu, log_var, beta, free_bits,
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

def freeze_except_vae_heads(model):
    """Freeze encoder + decoder, keep VAE heads + FiLM trainable."""
    frozen_prefixes = ('enc1', 'enc2', 'enc3', 'enc4', 'attn',
                       'dec4', 'dec3', 'dec2', 'dec1', 'output_conv')
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in frozen_prefixes):
            param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def training_loop():
    model.to(device)

    # Resume
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    start_epoch = load_checkpoint(model, optimizer, scheduler)

    # Handle freeze state on resume
    if start_epoch < WARMUP_EPOCHS:
        freeze_except_vae_heads(model)
        print(f"\n❄️  Phase A — Warmup: encoder+decoder FROZEN for {WARMUP_EPOCHS-start_epoch} more epochs")
    else:
        unfreeze_all(model)
        print(f"\n🔥 All layers unfrozen (resumed after warmup)")

    print(f"\n{'='*70}")
    print(f"🚀 VAE V2 FINE-TUNING — {MODE.upper()} MODE")
    print(f"   Device: {device} | Epochs: {start_epoch+1}-{NUM_EPOCHS}")
    print(f"{'='*70}\n")

    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            # Unfreeze at end of warmup
            if epoch == WARMUP_EPOCHS:
                unfreeze_all(model)
                print(f"\n🔥 Unfrozen: all layers (full fine-tuning)\n")

            beta_val = get_beta(epoch)

            train_pbar = helper_utils.NestedProgressBar(
                total_epochs=NUM_EPOCHS, total_batches=len(train_loader), mode="train")
            train_pbar.update_epoch(epoch + 1)

            epoch_loss, epoch_recon, epoch_kl = train_epoch(
                model, train_loader, optimizer, device, train_transform,
                scaler, use_amp, beta_val, FREE_BITS,
                classifier=classifier, class_loss_weight=CLASS_LOSS_WEIGHT, pbar=train_pbar)
            train_pbar.batch_bar.close()

            val_pbar = helper_utils.NestedProgressBar(
                total_epochs=1, total_batches=len(val_loader), mode="eval")
            _, epoch_val_recon, epoch_val_kl, epoch_val_cls = validate_epoch(
                model, val_loader, device, eval_transform, beta_val, FREE_BITS,
                classifier=classifier, class_loss_weight=CLASS_LOSS_WEIGHT, pbar=val_pbar)
            val_pbar.close()

            current_lr = optimizer.param_groups[0]['lr']

            ramp_end = WARMUP_EPOCHS + RAMP_EPOCHS
            if epoch < WARMUP_EPOCHS:
                phase = "warmup"
            elif epoch < ramp_end:
                phase = "β ramp"
            else:
                phase = "β fixed"

            train_pbar.update_epoch(epoch + 1, postfix_dict={
                "phase": phase,
                "train": f"{epoch_loss:.4f}",
                "val_mse": f"{epoch_val_recon:.4f}",
                "kl": f"{epoch_val_kl:.1f}",
                "β": f"{beta_val:.5f}",
                "lr": f"{current_lr:.1e}",
            })

            if epoch >= WARMUP_EPOCHS:
                scheduler.step()

            # Save checkpoint EVERY epoch (safe for Ctrl+C)
            save_checkpoint(model, optimizer, scheduler, epoch + 1)
    except KeyboardInterrupt:
        print(f"\n⏸️  Interrupted at epoch {epoch+1}. Checkpoint saved — resume anytime.")

    # Save final
    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_dim': LATENT_DIM,
        'embed_dim': EMBED_DIM,
        'num_classes': num_classes,
        'beta': BETA,
        'mode': MODE,
    }, BEST_MODEL_PATH)
    print(f"\n💾 Model saved → {BEST_MODEL_PATH}")

    return model


if __name__ == "__main__":
    training_loop()
