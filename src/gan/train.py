"""
train.py — GAN v16: Class-conditional mel spectrogram generation.

Generator: noise z + class → FiLM UpBlocks → mel [1, 64, 552]
Discriminator: spectral norm + R1 + auxiliary classifier
Loss: hinge adversarial + class prediction + R1 penalty

Usage:
    python src/gan/train.py
"""
import os
import sys
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.gan.config import config as cfg
from src.gan.generator import Generator
from src.gan.discriminator import Discriminator, compute_r1_penalty

# ═══════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
use_amp = (DEVICE.type == "cuda")

NUM_EPOCHS = cfg.epochs
BATCH_SIZE = cfg.batch_size
NUM_WORKERS = 4 if DEVICE.type == "cuda" else 0
SEED = 42

os.makedirs(cfg.model_dir, exist_ok=True)


# ═══════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════

class MelDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, split='train', train_split=0.95, bin_mean=None, bin_std=None):
        self.mel_transform = MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft,
            hop_length=cfg.hop_length, n_mels=cfg.n_mels,
            f_min=cfg.f_min, f_max=cfg.f_max, power=2,
        )
        self.db_transform = AmplitudeToDB(top_db=80)
        self.use_augment = (split == 'train')
        self.bin_mean = bin_mean
        self.bin_std = bin_std

        self.samples = []
        for cls_name, cls_idx in cfg.class_to_idx.items():
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.endswith('.wav'):
                    self.samples.append((os.path.join(cls_dir, fname), cls_idx))

        random.seed(SEED)
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * train_split)
        self.samples = self.samples[:split_idx] if split == 'train' else self.samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)

        target_len = int(cfg.segment_seconds * cfg.sample_rate)
        if wav.shape[1] < target_len:
            wav = F.pad(wav, (0, target_len - wav.shape[1]))
        else:
            start = random.randint(0, wav.shape[1] - target_len)
            wav = wav[:, start:start + target_len]

        spec = self.mel_transform(wav)
        mel_db = self.db_transform(spec)  # dB: [-80, 0]

        # Per-bin z-score normalization on dB values (kills the 38dB gap)
        if self.bin_mean is not None and self.bin_std is not None:
            mel = (mel_db - self.bin_mean) / self.bin_std.clamp(min=1.0)
        else:
            mel = mel_db / 40.0 + 1.0  # fallback for computing stats

        if self.use_augment and random.random() < cfg.aug_prob:
            mel = self._augment(mel)

        return mel, label

    def _augment(self, mel):
        if random.random() < 0.5:
            f = random.randint(1, cfg.aug_freq_mask_max)
            f0 = random.randint(0, mel.shape[-2] - f)
            mel[:, f0:f0 + f, :] = mel.mean()
        if random.random() < 0.5:
            t = random.randint(1, cfg.aug_time_mask_max)
            t0 = random.randint(0, mel.shape[-1] - t)
            mel[:, :, t0:t0 + t] = mel.mean()
        return mel


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════

def pad_mel(mel):
    B, C, H, W = mel.shape
    if W < cfg.mel_pad_width:
        return F.pad(mel, (0, cfg.mel_pad_width - W, 0, 0), mode='reflect')
    return mel[..., :cfg.mel_pad_width]


def d_hinge_loss(real_logits, fake_logits):
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def g_hinge_loss(fake_logits):
    return -fake_logits.mean()


def requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad_(flag)


# ═══════════════════════════════════════════════════════════
#  TRAIN
# ═══════════════════════════════════════════════════════════

def train():
    # Seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print(f"\n🔧 GAN Training — v16")
    print(f"   Device: {DEVICE} | Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"   G LR: {cfg.g_lr} | D LR: {cfg.d_lr} | AMP: {use_amp} | Hinge + R1(γ={cfg.r1_gamma})")

    # ── Data ──
    pin_mem = NUM_WORKERS > 0 and DEVICE.type == "cuda"

    # Compute per-bin stats on RAW dB data (fixes 38dB initialization gap)
    print("   Computing per-bin mel stats...")
    temp_ds = MelDataset(cfg.data_dir, 'train', cfg.train_split, bin_mean=None, bin_std=None)
    temp_loader = DataLoader(temp_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=pin_mem, drop_last=True)
    all_mels = []
    with torch.no_grad():
        for mel_batch, _ in tqdm(temp_loader, desc="   Stats", leave=False):
            all_mels.append(mel_batch)
    all_mels = torch.cat(all_mels, dim=0)  # [N, 1, 64, T]
    # Stats on raw dB: un-fallback the [-1,1] normalization
    all_mels_db = (all_mels - 1.0) * 40.0  # [-1,1] → dB [-80, 0]
    bin_mean = all_mels_db.mean(dim=(0, 3), keepdim=True)  # [1, 1, 64, 1]
    bin_std = all_mels_db.std(dim=(0, 3), keepdim=True)
    print(f"   Bin dB means: [{bin_mean.min():.0f}, {bin_mean.max():.0f}] dB")
    print(f"   Bin dB stds:  [{bin_std.min():.0f}, {bin_std.max():.0f}] dB")
    del all_mels, all_mels_db, temp_ds, temp_loader

    train_ds = MelDataset(cfg.data_dir, 'train', cfg.train_split, bin_mean=bin_mean, bin_std=bin_std)
    val_ds = MelDataset(cfg.data_dir, 'val', cfg.train_split, bin_mean=bin_mean, bin_std=bin_std)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                               pin_memory=pin_mem, drop_last=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=pin_mem)
    print(f"   Data: {len(train_ds)} train / {len(val_ds)} val")

    # ── Models ──
    G = Generator(cfg).to(DEVICE)
    D = Discriminator(cfg).to(DEVICE)
    print(f"   G params: {sum(p.numel() for p in G.parameters()):,} ({sum(p.numel() for p in G.parameters())/1e6:.1f}M)")
    print(f"   D params: {sum(p.numel() for p in D.parameters()):,} ({sum(p.numel() for p in D.parameters())/1e6:.1f}M)")

    # EMA generator
    G_ema = Generator(cfg).to(DEVICE)
    G_ema.load_state_dict(G.state_dict())
    for p in G_ema.parameters():
        p.requires_grad_(False)

    # ── Optimizer + Scheduler (AdamW + cosine → matches train_diff.py) ──
    g_opt = torch.optim.AdamW(G.parameters(), lr=cfg.g_lr,
                               betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.adam_weight_decay)
    d_opt = torch.optim.AdamW(D.parameters(), lr=cfg.d_lr,
                               betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.adam_weight_decay)
    g_sched = torch.optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max=NUM_EPOCHS)
    d_sched = torch.optim.lr_scheduler.CosineAnnealingLR(d_opt, T_max=NUM_EPOCHS)

    # ── AMP ──
    scaler_g = torch.amp.GradScaler('cuda') if use_amp else None
    scaler_d = torch.amp.GradScaler('cuda') if use_amp else None

    # ── Loss ──
    ce_loss = nn.CrossEntropyLoss()

    best_val = 0.0
    step = 0

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()

        # ── Train ────────────────────────────────────
        G.train()
        D.train()
        g_loss_sum = 0.0
        d_loss_sum = 0.0
        d_cls_sum = 0.0
        g_cls_sum = 0.0

        pbar = tqdm(train_loader, desc=f"  Train {epoch+1}/{NUM_EPOCHS}", leave=False)
        for real_mel, real_labels in pbar:
            real_mel = real_mel.to(DEVICE)
            real_labels = real_labels.to(DEVICE)
            B = real_mel.shape[0]
            real_mel = pad_mel(real_mel)

            # ═══ Discriminator ═══
            requires_grad(G, False)
            requires_grad(D, True)
            d_opt.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    real_adv, real_cls = D(real_mel)
                    d_cls_loss = cfg.d_class_weight * ce_loss(real_cls, real_labels)

                    z = torch.randn(B, cfg.latent_dim, device=DEVICE)
                    with torch.no_grad():
                        fake_mel = G(z, real_labels)
                        fake_mel = pad_mel(fake_mel)
                    fake_adv, _ = D(fake_mel.detach())
                    d_loss = d_hinge_loss(real_adv, fake_adv) + d_cls_loss

                    if step % cfg.r1_every == 0:
                        d_loss = d_loss + cfg.r1_gamma * 0.5 * compute_r1_penalty(D, real_mel, real_labels)
            else:
                real_adv, real_cls = D(real_mel)
                d_cls_loss = cfg.d_class_weight * ce_loss(real_cls, real_labels)

                z = torch.randn(B, cfg.latent_dim, device=DEVICE)
                with torch.no_grad():
                    fake_mel = G(z, real_labels)
                    fake_mel = pad_mel(fake_mel)
                fake_adv, _ = D(fake_mel.detach())
                d_loss = d_hinge_loss(real_adv, fake_adv) + d_cls_loss

                if step % cfg.r1_every == 0:
                    d_loss = d_loss + cfg.r1_gamma * 0.5 * compute_r1_penalty(D, real_mel, real_labels)

            if use_amp:
                scaler_d.scale(d_loss).backward()
                scaler_d.unscale_(d_opt)
                torch.nn.utils.clip_grad_norm_(D.parameters(), cfg.grad_clip)
                scaler_d.step(d_opt)
                scaler_d.update()
            else:
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), cfg.grad_clip)
                d_opt.step()

            d_loss_sum += d_loss.item()
            d_cls_sum += (real_cls.argmax(1) == real_labels).float().mean().item()

            # ═══ Generator ═══
            requires_grad(G, True)
            requires_grad(D, False)
            g_opt.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    z = torch.randn(B, cfg.latent_dim, device=DEVICE)
                    fake_mel = G(z, real_labels)
                    fake_mel = pad_mel(fake_mel)
                    fake_adv, fake_cls = D(fake_mel)

                    g_loss = (cfg.g_adv_weight * g_hinge_loss(fake_adv) +
                             cfg.g_class_weight * ce_loss(fake_cls, real_labels))
            else:
                z = torch.randn(B, cfg.latent_dim, device=DEVICE)
                fake_mel = G(z, real_labels)
                fake_mel = pad_mel(fake_mel)
                fake_adv, fake_cls = D(fake_mel)

                g_loss = (cfg.g_adv_weight * g_hinge_loss(fake_adv) +
                         cfg.g_class_weight * ce_loss(fake_cls, real_labels))

            if use_amp:
                scaler_g.scale(g_loss).backward()
                scaler_g.unscale_(g_opt)
                torch.nn.utils.clip_grad_norm_(G.parameters(), cfg.grad_clip)
                scaler_g.step(g_opt)
                scaler_g.update()
            else:
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), cfg.grad_clip)
                g_opt.step()

            g_loss_sum += g_loss.item()
            g_cls_sum += (fake_cls.argmax(1) == real_labels).float().mean().item()

            step += 1

            # EMA update
            with torch.no_grad():
                for ema_p, p in zip(G_ema.parameters(), G.parameters()):
                    ema_p.data.mul_(cfg.ema_decay).add_(p.data, alpha=1 - cfg.ema_decay)

            pbar.set_postfix({"G": f"{g_loss.item():.4f}", "D": f"{d_loss.item():.4f}"})

        # ── Validate ─────────────────────────────────
        G_ema.eval()
        D.eval()
        val_acc = 0.0
        n_val = 0
        with torch.no_grad():
            for val_mel, val_labels in val_loader:
                val_mel = val_mel.to(DEVICE)
                val_labels = val_labels.to(DEVICE)
                Bv = val_mel.shape[0]
                val_mel = pad_mel(val_mel)
                _, val_cls = D(val_mel)
                val_acc += (val_cls.argmax(1) == val_labels).float().sum().item()
                n_val += Bv

        val_acc = val_acc / n_val if n_val > 0 else 0.0

        # ── Logging ──────────────────────────────────
        avg_g = g_loss_sum / len(train_loader)
        avg_d = d_loss_sum / len(train_loader)
        avg_d_cls = d_cls_sum / len(train_loader) * 100
        avg_g_cls = g_cls_sum / len(train_loader) * 100

        marker = "📉" if val_acc > best_val else "➡️"
        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                'generator': G_ema.state_dict(),
                'discriminator': D.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch,
                'g_params': sum(p.numel() for p in G.parameters()),
                'd_params': sum(p.numel() for p in D.parameters()),
                'bin_mean': bin_mean.cpu(),
                'bin_std': bin_std.cpu(),
            }, os.path.join(cfg.model_dir, "gan_generator_best.pth"))
            print(f"   💾 Best model saved (val_acc={val_acc*100:.1f}%)")

        g_sched.step()
        d_sched.step()

        elapsed = time.time() - t0
        lr = g_opt.param_groups[0]['lr']
        print(f"── Epoch {epoch+1:3d}/{NUM_EPOCHS} ({elapsed:.0f}s) ── "
              f"G={avg_g:.4f} D={avg_d:.4f} "
              f"D_cls={avg_d_cls:.1f}% G_cls={avg_g_cls:.1f}% "
              f"val_acc={val_acc*100:.1f}% {marker} lr={lr:.2e}")

        # ── Checkpoint + Samples (every 10 epochs) ──
        if (epoch + 1) % 10 == 0:
            torch.save({
                'generator': G_ema.state_dict(),
                'discriminator': D.state_dict(),
                'epoch': epoch,
            }, os.path.join(cfg.model_dir, f"gan_e{epoch+1:03d}.pth"))
            generate_samples(G_ema, DEVICE, epoch + 1)

    print(f"\n✅ GAN training complete. Best val_acc: {best_val*100:.1f}%")


# ═══════════════════════════════════════════════════════════
#  GENERATION
# ═══════════════════════════════════════════════════════════

def generate_samples(generator, device, epoch):
    from src.gan.generate import mel_to_audio
    generator.eval()
    with torch.no_grad():
        for cls_name, cls_idx in cfg.class_to_idx.items():
            z = torch.randn(cfg.num_samples, cfg.latent_dim, device=device)
            labels = torch.full((cfg.num_samples,), cls_idx, device=device, dtype=torch.long)
            mels = generator(z, labels)
            audio = mel_to_audio(mels[0:1], bin_mean, bin_std).squeeze()
            torchaudio.save(f"outputs/gan_v16_e{epoch}_{cls_name}.wav",
                            audio.cpu(), cfg.sample_rate)
    print(f"   🎵 Saved samples to outputs/")


if __name__ == "__main__":
    train()
