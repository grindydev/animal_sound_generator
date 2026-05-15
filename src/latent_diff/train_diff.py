"""
train_diff.py — Phase 2: Train Tiny Diffusion UNet on Compressed Latents.

Encodes all real mels → spatial latents [B, 16, 4, 35] → train UNet to denoise.

Usage:
    python src/latent_diff/train_diff.py
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.latent_diff.config import config as cfg
from src.latent_diff.decoder import LatentDecoder, ChannelReducer, ChannelExpander
from src.latent_diff.unet import LatentUNet
from src.vae.autoencoder import ImprovedAutoencoder
from src.diffusion.diffusion import DiffusionProcess
from src.diffusion.train import DiffusionDataset

# ═══════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
use_amp = (DEVICE.type == "cuda")

NUM_EPOCHS = 100
BATCH_SIZE = 32
GRAD_ACCUM = 4  # effective batch = 128
NUM_WORKERS = 4 if DEVICE.type == "cuda" else 0
LR = 2e-4
SEGMENT_FRAMES = cfg.segment_frames
EMA_DECAY = 0.9999

os.makedirs(cfg.model_dir, exist_ok=True)


# ═══════════════════════════════════════════════════════════
#  LOAD FROZEN ENCODER + REDUCER + DECODER
# ═══════════════════════════════════════════════════════════

def load_encoder_and_reducer():
    """Load pre-trained encoder and channel reducer (both frozen)."""
    ckpt_path = cfg.encoder_ckpt
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(cfg.model_dir, "best_autoencoder_train.pth")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    state_dict = ckpt.get('model_state_dict', ckpt.get('model', {}))
    if 'config' in ckpt:
        base_ch = ckpt['config'].get('base_channels', 32)
    elif state_dict:
        fc_w = state_dict.get('fc_encode.weight')
        flat_dim = fc_w.shape[1] if fc_w is not None else 0
        c4 = flat_dim // (4 * 35)
        base_ch = c4 // 8 if c4 > 0 else 32
    else:
        base_ch = 32

    encoder = ImprovedAutoencoder(latent_dim=2048, base_channels=base_ch)
    encoder.load_state_dict(state_dict)
    encoder.to(DEVICE)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    bottleneck_ch = encoder.c4

    # Load trained reducer if available, else create new
    reducer_path = os.path.join(cfg.model_dir, "latent_decoder_best.pth")
    reducer = ChannelReducer(in_ch=bottleneck_ch, out_ch=cfg.latent_channels).to(DEVICE)

    if os.path.exists(reducer_path):
        dec_ckpt = torch.load(reducer_path, map_location=DEVICE, weights_only=True)
        reducer.load_state_dict(dec_ckpt['reducer'])
        reducer.eval()
        print(f"✅ Reducer loaded from: {reducer_path}")
    else:
        print(f"⚠️  Reducer not found at {reducer_path} — using untrained")

    for p in reducer.parameters():
        p.requires_grad_(False)

    return encoder, reducer, bottleneck_ch


# ═══════════════════════════════════════════════════════════
#  TRAIN
# ═══════════════════════════════════════════════════════════

def train():
    print(f"\n🔧 Latent Diffusion Training — Phase 2")
    print(f"   Device: {DEVICE} | Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} (eff: {BATCH_SIZE*GRAD_ACCUM})")
    print(f"   LR: {LR} | AMP: {use_amp} | Loss: {cfg.loss_type}")

    encoder, reducer, bottleneck_ch = load_encoder_and_reducer()

    # Create UNet and diffusion
    unet = LatentUNet(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in unet.parameters())
    diffusion = DiffusionProcess.__new__(DiffusionProcess)
    DiffusionProcess.__init__(diffusion, cfg)
    diffusion.to(DEVICE)
    diffusion.num_classes = cfg.num_classes  # needed for CFG
    print(f"   UNet params: {n_params:,} ({n_params/1e6:.1f}M)")

    # EMA model
    ema_model = LatentUNet(cfg).to(DEVICE)
    ema_model.load_state_dict(unet.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LR, weight_decay=cfg.adam_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=NUM_EPOCHS // 4, T_mult=2, eta_min=1e-6
    )

    # Data
    train_ds = DiffusionDataset(cfg.data_dir, SEGMENT_FRAMES, split="train")
    val_ds = DiffusionDataset(cfg.data_dir, SEGMENT_FRAMES, split="val")
    pin_mem = NUM_WORKERS > 0 and DEVICE.type == "cuda"
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                               pin_memory=pin_mem, drop_last=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=pin_mem)
    print(f"   Data: {len(train_ds)} train / {len(val_ds)} val")

    # Loss function
    loss_fn = F.l1_loss if cfg.loss_type == 'l1' else F.mse_loss

    best_val = float('inf')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()

        # ── Train ────────────────────────────────────
        unet.train()
        total_loss = 0.0
        optimizer.zero_grad()
        accum_count = 0

        pbar = tqdm(train_loader, desc=f"  Train {epoch+1}/{NUM_EPOCHS}", leave=False)
        for mel_batch, labels in pbar:
            mel_batch = mel_batch.to(DEVICE)
            labels = labels.to(DEVICE)
            B = mel_batch.shape[0]

            # Encode to latent
            with torch.no_grad():
                features = encoder.encode_spatial(mel_batch)
                latent = reducer(features)  # [B, 16, 4, 35]

            # Unconditional training (CFG)
            if cfg.uncond_prob > 0 and np.random.random() < cfg.uncond_prob:
                labels = torch.full_like(labels, cfg.num_classes)

            # Sample timestep
            t = torch.randint(0, cfg.timesteps, (B,), device=DEVICE)
            noise = torch.randn_like(latent)
            x_t = diffusion.q_sample(latent, t, noise)

            # Predict noise
            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = unet(x_t, t, labels)
                    loss = loss_fn(pred, noise) / GRAD_ACCUM
            else:
                pred = unet(x_t, t, labels)
                loss = loss_fn(pred, noise) / GRAD_ACCUM

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_count += 1
            total_loss += loss.item() * GRAD_ACCUM

            if accum_count >= GRAD_ACCUM:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), cfg.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), cfg.grad_clip_norm)
                    optimizer.step()
                optimizer.zero_grad()
                accum_count = 0

                # EMA update
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), unet.parameters()):
                        ema_p.data.mul_(EMA_DECAY).add_(p.data, alpha=1 - EMA_DECAY)

            pbar.set_postfix({"loss": f"{loss.item()*GRAD_ACCUM:.4f}"})

        if accum_count > 0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), cfg.grad_clip_norm)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        # ── Validate ─────────────────────────────────
        unet.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mel_batch, labels in val_loader:
                mel_batch = mel_batch.to(DEVICE)
                labels = labels.to(DEVICE)
                B = mel_batch.shape[0]

                features = encoder.encode_spatial(mel_batch)
                latent = reducer(features)

                t = torch.randint(0, cfg.timesteps, (B,), device=DEVICE)
                noise = torch.randn_like(latent)
                x_t = diffusion.q_sample(latent, t, noise)

                pred = ema_model(x_t, t, labels)
                val_loss += loss_fn(pred, noise).item()

        val_loss /= max(len(val_loader), 1)

        marker = "📉" if val_loss < best_val else "➡️"
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'unet': ema_model.state_dict(),
                'val_loss': val_loss,
                'epoch': epoch,
                'config': cfg,
            }, os.path.join(cfg.model_dir, "latent_diffusion_best.pth"))
            print(f"   💾 Best model saved (val={val_loss:.4f})")

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        print(f"── Epoch {epoch+1:3d}/{NUM_EPOCHS} ({elapsed:.0f}s) ── loss={avg_loss:.4f} val={val_loss:.4f} {marker} lr={lr:.2e}")

    print(f"\n✅ Diffusion training complete. Best val_loss: {best_val:.4f}")


if __name__ == "__main__":
    train()
