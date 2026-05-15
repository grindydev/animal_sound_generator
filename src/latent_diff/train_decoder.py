"""
train_decoder.py — Phase 1: Train Channel Reducers + Small Decoder.

Freeze autoencoder encoder. Train:
  - conv_reduce: 256 → 16 channels (compresses spatial bottleneck)
  - conv_expand: 16 → 256 channels (decompresses after diffusion)
  - LatentDecoder: 256ch × 4×35 → upsample → mel [1, 64, 552]

No skip connections. Pure upsampling decoder by design.

Usage:
    python src/latent_diff/train_decoder.py
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.latent_diff.config import config as cfg
from src.latent_diff.decoder import LatentDecoder, ChannelReducer, ChannelExpander
from src.vae.autoencoder import ImprovedAutoencoder
from src.diffusion.train import DiffusionDataset, compute_mel

# ═══════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
use_amp = (DEVICE.type == "cuda")

MODE = "train"
NUM_EPOCHS = 30
BATCH_SIZE = 16
NUM_WORKERS = 4 if DEVICE.type == "cuda" else 0
LR = 1e-3
SEGMENT_FRAMES = cfg.segment_frames

os.makedirs(cfg.model_dir, exist_ok=True)


# ═══════════════════════════════════════════════════════════
#  LOAD ENCODER (frozen)
# ═══════════════════════════════════════════════════════════

def load_encoder():
    """Load pre-trained autoencoder encoder (frozen)."""
    # Try loading checkpoint to detect base_channels
    ckpt_path = cfg.encoder_ckpt
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(cfg.model_dir, "best_autoencoder_train.pth")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)

    # Auto-detect base_channels from checkpoint state dict
    state_dict = ckpt.get('model_state_dict', ckpt.get('model', {}))
    if 'config' in ckpt:
        base_ch = ckpt['config'].get('base_channels', 32)
    elif state_dict:
        # Probe enc4 channels from fc_encode weight: flat_dim = c4 * 4 * 35, c4 = base_ch * 8
        fc_w = state_dict.get('fc_encode.weight')
        flat_dim = fc_w.shape[1] if fc_w is not None else 0
        c4 = flat_dim // (4 * 35)  # enc4 output channels
        base_ch = c4 // 8 if c4 > 0 else 32
    else:
        base_ch = 32

    model = ImprovedAutoencoder(latent_dim=2048, base_channels=base_ch)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # Freeze all encoder params
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"✅ Encoder loaded (base_ch={base_ch}, bottleneck={model.c4}ch)")
    return model, model.c4


# ═══════════════════════════════════════════════════════════
#  TRAIN
# ═══════════════════════════════════════════════════════════

def train():
    print(f"\n🔧 Latent Decoder Training — Phase 1")
    print(f"   Device: {DEVICE} | Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"   LR: {LR} | AMP: {use_amp}")

    # Load encoder
    encoder, bottleneck_ch = load_encoder()

    # Create reducer/expander/decoder
    reducer = ChannelReducer(in_ch=bottleneck_ch, out_ch=cfg.latent_channels).to(DEVICE)
    expander = ChannelExpander(in_ch=cfg.latent_channels, out_ch=bottleneck_ch).to(DEVICE)
    decoder = LatentDecoder(bottleneck_ch=bottleneck_ch, config=cfg).to(DEVICE)

    n_params = sum(p.numel() for p in decoder.parameters())
    n_params += sum(p.numel() for p in reducer.parameters())
    n_params += sum(p.numel() for p in expander.parameters())
    print(f"   Trainable params: {n_params:,} ({n_params/1e6:.1f}M)")

    # Optimizer
    trainable = list(reducer.parameters()) + list(expander.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Data
    train_ds = DiffusionDataset(cfg.data_dir, SEGMENT_FRAMES, split="train")
    val_ds = DiffusionDataset(cfg.data_dir, SEGMENT_FRAMES, split="val")
    pin_mem = NUM_WORKERS > 0 and DEVICE.type == "cuda"
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                               pin_memory=pin_mem, drop_last=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=pin_mem)
    print(f"   Data: {len(train_ds)} train / {len(val_ds)} val")

    # Train
    best_val = float('inf')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()

        # ── Train ────────────────────────────────────
        reducer.train()
        expander.train()
        decoder.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for mel, _ in pbar:
            mel = mel.to(DEVICE)  # [B, 1, 64, T]

            # Encode to spatial features (frozen encoder)
            with torch.no_grad():
                features = encoder.encode_spatial(mel)  # [B, bottleneck_ch, 4, 35]

            # Reduce → Expand → Decode
            if use_amp:
                with torch.cuda.amp.autocast():
                    latent = reducer(features)
                    expanded = expander(latent)
                    output = decoder(expanded, target_size=(64, SEGMENT_FRAMES))
                    loss = F.mse_loss(output, mel)
            else:
                latent = reducer(features)
                expanded = expander(latent)
                output = decoder(expanded, target_size=(64, SEGMENT_FRAMES))
                loss = F.mse_loss(output, mel)

            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)

        # ── Validate ─────────────────────────────────
        reducer.eval()
        expander.eval()
        decoder.eval()
        val_loss = 0.0

        with torch.no_grad():
            for mel, _ in val_loader:
                mel = mel.to(DEVICE)
                features = encoder.encode_spatial(mel)
                latent = reducer(features)
                expanded = expander(latent)
                output = decoder(expanded, target_size=(64, SEGMENT_FRAMES))
                val_loss += F.mse_loss(output, mel).item()

        val_loss /= max(len(val_loader), 1)
        scheduler.step()

        marker = "📉" if val_loss < best_val else "➡️"
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'reducer': reducer.state_dict(),
                'expander': expander.state_dict(),
                'decoder': decoder.state_dict(),
                'bottleneck_ch': bottleneck_ch,
                'val_loss': val_loss,
            }, os.path.join(cfg.model_dir, "latent_decoder_best.pth"))
            print(f"   💾 Best model saved (val={val_loss:.4f})")

        elapsed = time.time() - t0
        print(f"── Epoch {epoch+1:3d}/{NUM_EPOCHS} ({elapsed:.0f}s) ── loss={avg_loss:.4f} val={val_loss:.4f} {marker}")

    print(f"\n✅ Decoder training complete. Best val_loss: {best_val:.4f}")


if __name__ == "__main__":
    train()
