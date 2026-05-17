"""
train_v14.py — V14: Latent Diffusion Training
==============================================

Trains in TWO phases:
  Phase 1: VAE (compression) — encode mel → z[256] → decode
  Phase 2: Latent Diffusion — train diffusion on VAE latents

Industry standard pipeline (AudioLDM, Stable Audio):
  Mel → VAE Encoder → z → Diffusion → z → VAE Decoder → Mel → Audio

Usage:
  python src/train_v14.py                     # Full training
  python src/train_v14.py --phase 1           # VAE only
  python src/train_v14.py --phase 2           # Diffusion only
  python src/train_v14.py --mode test         # Quick test (5 epochs each)
"""
import os
import sys
import argparse
import time
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.v14_vae import V14VAE
from src.v14_ldm import LatentDiffusionModel, LatentDiffusion
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    "mode": "train",          # "test" = quick | "train" = full
    "device": "auto",         # "auto", "cuda", "mps", "cpu"
    "data_dir": "data/esc50",

    # VAE
    "vae_latent_dim": 256,
    "vae_class_emb_dim": 32,
    "vae_lr": 1e-4,
    "vae_beta": 0.0001,       # KL weight (very small)
    "vae_epochs_test": 5,
    "vae_epochs_train": 50,

    # Latent Diffusion
    "ldm_latent_dim": 256,
    "ldm_time_emb_dim": 64,
    "ldm_class_emb_dim": 32,
    "ldm_hidden_dim": 512,
    "ldm_num_blocks": 4,
    "ldm_timesteps": 1000,
    "ldm_lr": 1e-3,
    "ldm_epochs_test": 10,
    "ldm_epochs_train": 100,
    "ldm_uncond_prob": 0.1,  # 10% unconditional for CFG

    # Shared
    "batch_size": 16,
    "num_workers": 2,
    "segment_frames": 552,
    "sample_rate": 22050,
    "n_mels": 64,

    # Paths
    "model_dir": "models",
}

CLASSES = ['Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# ═══════════════════════════════════════════════════════════════
#  DEVICE
# ═══════════════════════════════════════════════════════════════

def get_device():
    d = CONFIG["device"]
    if d == "auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        elif torch.backends.mps.is_available(): return torch.device("mps")
        else: return torch.device("cpu")
    return torch.device(d)


# ═══════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════

class MelDataset(Dataset):
    """Load audio, compute mel spectrogram."""
    def __init__(self, data_dir, split="train", segment_frames=552):
        self.segment_frames = segment_frames
        self.samples = []

        for cls_name in sorted(os.listdir(data_dir)):
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_dir) or cls_name not in CLASS_TO_IDX:
                continue
            cls_idx = CLASS_TO_IDX[cls_name]
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith('.wav'):
                    self.samples.append((os.path.join(cls_dir, fname), cls_idx))

        np.random.seed(42)
        np.random.shuffle(self.samples)
        split_idx = int(len(self.samples) * 0.9)
        self.samples = self.samples[:split_idx] if split == "train" else self.samples[split_idx:]

        # Pre-create transforms
        self.mel_tfm = MelSpectrogram(
            sample_rate=22050, n_fft=1024, hop_length=200,
            n_mels=64, f_min=0, f_max=11025, power=2,
        )
        self.db_tfm = AmplitudeToDB(top_db=80)
        self.norm_mean = -18.4903
        self.norm_std = 19.8031

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            wav, sr = torchaudio.load(path)
            if sr != 22050:
                wav = torchaudio.transforms.Resample(sr, 22050)(wav)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
        except Exception:
            return torch.zeros(1, 64, self.segment_frames), 0

        # Pad/crop
        target_samples = self.segment_frames * 200
        if wav.shape[-1] < target_samples:
            wav = F.pad(wav, (0, target_samples - wav.shape[-1]))
        else:
            wav = wav[:, :target_samples]

        # Compute mel
        spec = self.mel_tfm(wav)
        mel = (self.db_tfm(spec) - self.norm_mean) / self.norm_std
        mel = mel.squeeze(0)  # [1, 64, T] → [64, T]
        if mel.shape[-1] > self.segment_frames:
            mel = mel[..., :self.segment_frames]
        elif mel.shape[-1] < self.segment_frames:
            mel = F.pad(mel, (0, self.segment_frames - mel.shape[-1]))

        return mel, label


# ═══════════════════════════════════════════════════════════════
#  PHASE 1: VAE TRAINING
# ═══════════════════════════════════════════════════════════════

def train_vae(device, mode="train"):
    cfg = CONFIG
    epochs = cfg["vae_epochs_train"] if mode == "train" else cfg["vae_epochs_test"]
    bs = cfg["batch_size"]

    print(f"\n{'='*60}")
    print(f"🚀 PHASE 1: VAE Training ({epochs} epochs)")
    print(f"{'='*60}")

    train_ds = MelDataset(cfg["data_dir"], "train", cfg["segment_frames"])
    val_ds = MelDataset(cfg["data_dir"], "val", cfg["segment_frames"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=cfg["num_workers"], drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = V14VAE(
        latent_dim=cfg["vae_latent_dim"],
        num_classes=7,
        class_emb_dim=cfg["vae_class_emb_dim"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   VAE params: {n_params:,} ({n_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["vae_lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(cfg["model_dir"], exist_ok=True)
    best_path = os.path.join(cfg["model_dir"], "v14_vae.pth")
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0

        pbar = tqdm(train_loader, desc=f"  VAE Epoch {epoch+1}/{epochs}", leave=False)
        for mel, labels in pbar:
            mel = mel.unsqueeze(1).to(device)  # [B, 1, 64, 552]
            labels = labels.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(mel, labels)

            # Loss
            recon_loss = F.l1_loss(recon, mel)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + cfg["vae_beta"] * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mel, labels in val_loader:
                mel = mel.unsqueeze(1).to(device)
                labels = labels.to(device)
                recon, mu, logvar = model(mel, labels)
                recon_loss = F.l1_loss(recon, mel)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                val_loss += (recon_loss + cfg["vae_beta"] * kl_loss).item()

        n_batches = len(train_loader)
        val_loss /= max(len(val_loader), 1)
        avg_train = train_loss / n_batches

        print(f"  VAE Epoch {epoch+1:3d} | "
              f"train={avg_train:.4f} (rec={train_recon/n_batches:.4f} kl={train_kl/n_batches:.4f}) | "
              f"val={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"    💾 Saved (val={best_loss:.4f})")

    torch.save(model.state_dict(), best_path)
    print(f"✅ VAE training complete → {best_path} (val={best_loss:.4f})")
    return model


# ═══════════════════════════════════════════════════════════════
#  PHASE 2: LATENT DIFFUSION TRAINING
# ═══════════════════════════════════════════════════════════════

def train_ldm(device, mode="train"):
    cfg = CONFIG
    epochs = cfg["ldm_epochs_train"] if mode == "train" else cfg["ldm_epochs_test"]
    bs = cfg["batch_size"]

    print(f"\n{'='*60}")
    print(f"🚀 PHASE 2: Latent Diffusion Training ({epochs} epochs)")
    print(f"{'='*60}")

    # Load trained VAE
    vae_path = os.path.join(cfg["model_dir"], "v14_vae.pth")
    if not os.path.exists(vae_path):
        print(f"❌ VAE checkpoint not found: {vae_path}")
        print("   Run Phase 1 first!")
        return None

    vae = V14VAE(
        latent_dim=cfg["vae_latent_dim"],
        num_classes=7,
        class_emb_dim=cfg["vae_class_emb_dim"],
    ).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"✅ VAE loaded from {vae_path}")

    # Encode all training data to latents
    print("   Encoding training data to latents...")
    train_ds = MelDataset(cfg["data_dir"], "train", cfg["segment_frames"])
    val_ds = MelDataset(cfg["data_dir"], "val", cfg["segment_frames"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    train_latents = []
    train_labels = []

    with torch.no_grad():
        for mel, labels in tqdm(train_loader, desc="  Encoding train"):
            mel = mel.unsqueeze(1).to(device)
            z = vae.encode_to_latent(mel)
            train_latents.append(z.cpu())
            train_labels.append(labels)
        for mel, labels in tqdm(val_loader, desc="  Encoding val"):
            mel = mel.unsqueeze(1).to(device)
            z = vae.encode_to_latent(mel)
            train_latents.append(z.cpu())
            train_labels.append(labels)

    all_latents = torch.cat(train_latents, dim=0)
    all_labels = torch.cat(train_labels, dim=0)
    print(f"   Latents: {all_latents.shape}, mean={all_latents.mean():.3f}, std={all_latents.std():.3f}")

    # Shuffle and split
    n = len(all_latents)
    indices = np.random.permutation(n)
    split = int(n * 0.9)
    train_idx, val_idx = indices[:split], indices[split:]

    train_lat, train_lab = all_latents[train_idx], all_labels[train_idx]
    val_lat, val_lab = all_latents[val_idx], all_labels[val_idx]

    train_ds2 = torch.utils.data.TensorDataset(train_lat, train_lab)
    val_ds2 = torch.utils.data.TensorDataset(val_lat, val_lab)
    train_loader2 = DataLoader(train_ds2, batch_size=bs, shuffle=True, drop_last=True)
    val_loader2 = DataLoader(val_ds2, batch_size=bs, shuffle=False)

    print(f"   Train: {len(train_ds2)} | Val: {len(val_ds2)}")

    # Create model
    model = LatentDiffusionModel(
        latent_dim=cfg["ldm_latent_dim"],
        class_emb_dim=cfg["ldm_class_emb_dim"],
        time_emb_dim=cfg["ldm_time_emb_dim"],
        hidden_dim=cfg["ldm_hidden_dim"],
        num_blocks=cfg["ldm_num_blocks"],
        num_classes=7,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   LDM params: {n_params:,} ({n_params/1e6:.2f}M)")

    diffusion = LatentDiffusion(timesteps=cfg["ldm_timesteps"]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["ldm_lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_path = os.path.join(cfg["model_dir"], "v14_ldm.pth")
    best_loss = float("inf")

    null_label = model.null_label

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader2, desc=f"  LDM Epoch {epoch+1}/{epochs}", leave=False)
        for z_clean, labels in pbar:
            z_clean = z_clean.to(device)
            labels = labels.to(device)
            B = z_clean.shape[0]

            # Random timesteps
            t = torch.randint(0, diffusion.timesteps, (B,), device=device)

            # Unconditional training (for CFG)
            if np.random.random() < cfg["ldm_uncond_prob"]:
                labels = torch.full_like(labels, null_label)

            # Add noise
            noise = torch.randn_like(z_clean)
            z_t = diffusion.q_sample(z_clean, t, noise)

            # Predict clean latent
            optimizer.zero_grad()
            pred_z0 = model(z_t, t, labels)
            loss = F.l1_loss(pred_z0, z_clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for z_clean, labels in val_loader2:
                z_clean = z_clean.to(device)
                labels = labels.to(device)
                B = z_clean.shape[0]
                t = torch.randint(0, diffusion.timesteps, (B,), device=device)
                noise = torch.randn_like(z_clean)
                z_t = diffusion.q_sample(z_clean, t, noise)
                pred_z0 = model(z_t, t, labels)
                val_loss += F.l1_loss(pred_z0, z_clean).item()

        n_batches = len(train_loader2)
        avg_train = total_loss / n_batches
        avg_val = val_loss / max(len(val_loader2), 1)

        print(f"  LDM Epoch {epoch+1:3d} | train={avg_train:.4f} | val={avg_val:.4f}")

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), best_path)
            print(f"    💾 Saved (val={best_loss:.4f})")

    torch.save(model.state_dict(), best_path)
    print(f"✅ LDM training complete → {best_path} (val={best_loss:.4f})")
    return model


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="V14 Latent Diffusion Training")
    parser.add_argument("--phase", type=int, default=0, help="1=VAE, 2=LDM, 0=both")
    parser.add_argument("--mode", type=str, default=None, help="train or test")
    args = parser.parse_args()

    if args.mode:
        CONFIG["mode"] = args.mode

    device = get_device()
    print(f"🚀 Device: {device}")
    print(f"   Mode: {CONFIG['mode']}")

    mode = CONFIG["mode"]

    if args.phase in (0, 1):
        train_vae(device, mode)

    if args.phase in (0, 2):
        train_ldm(device, mode)

    print(f"\n{'='*60}")
    print("✅ V14 Training Complete!")
    print(f"   VAE: models/v14_vae.pth")
    print(f"   LDM: models/v14_ldm.pth")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
