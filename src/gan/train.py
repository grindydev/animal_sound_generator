"""
GAN Training — Class-Conditional Mel Spectrogram Generation

Usage: python -m src.gan.train
"""
import os, sys, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.gan.config import config as cfg
from src.gan.generator import Generator
from src.gan.discriminator import Discriminator, compute_r1_penalty


# ════════════════════════════════════════════════════════════════════
# Dataset
# ════════════════════════════════════════════════════════════════════

class MelDataset(torch.utils.data.Dataset):
    """Load precomputed mel spectrograms or compute on-the-fly."""

    def __init__(self, data_dir, split='train', train_split=0.95):
        self.mel_transform = MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            power=2,
        )
        self.db_transform = AmplitudeToDB(top_db=80)

        self.samples = []
        for cls_name, cls_idx in cfg.class_to_idx.items():
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.endswith('.wav'):
                    self.samples.append((os.path.join(cls_dir, fname), cls_idx))

        # Shuffle and split
        random.seed(42)
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * train_split)

        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

        self.use_augment = (split == 'train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)

        # Pad or trim to target length
        target_len = int(cfg.segment_seconds * cfg.sample_rate)
        if wav.shape[1] < target_len:
            wav = F.pad(wav, (0, target_len - wav.shape[1]))
        else:
            # Random crop
            start = random.randint(0, wav.shape[1] - target_len)
            wav = wav[:, start:start + target_len]

        # Mel spectrogram
        spec = self.mel_transform(wav)
        mel = self.db_transform(spec)

        # Normalize to [-1, 1] using robust scaling
        # Real mels are roughly [-80, 0] dB → map to [-1, 1]
        mel = mel / 40.0 + 1.0  # [-80,0] → [-1, 1]
        mel = mel.clamp(-1, 1)

        # Mild augmentation for training
        if self.use_augment and random.random() < cfg.aug_prob:
            mel = self._augment(mel)

        return mel, label

    def _augment(self, mel):
        """Mild SpecAugment for GAN training."""
        # Frequency masking
        if random.random() < 0.5:
            freq_mask = random.randint(1, cfg.aug_freq_mask_max)
            f0 = random.randint(0, mel.shape[-2] - freq_mask)
            mel[:, f0:f0 + freq_mask, :] = mel.mean()

        # Time masking
        if random.random() < 0.5:
            time_mask = random.randint(1, cfg.aug_time_mask_max)
            t0 = random.randint(0, mel.shape[-1] - time_mask)
            mel[:, :, t0:t0 + time_mask] = mel.mean()

        return mel


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

def pad_mel(mel):
    """Pad mel to cfg.mel_pad_width for clean conv dimensions."""
    B, C, H, W = mel.shape
    if W < cfg.mel_pad_width:
        pad = cfg.mel_pad_width - W
        return F.pad(mel, (0, pad, 0, 0), mode='reflect')
    return mel[..., :cfg.mel_pad_width]


def requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad_(flag)


# ════════════════════════════════════════════════════════════════════
# Losses
# ════════════════════════════════════════════════════════════════════

def d_hinge_loss(real_logits, fake_logits):
    """Hinge loss for discriminator."""
    real_loss = F.relu(1.0 - real_logits).mean()
    fake_loss = F.relu(1.0 + fake_logits).mean()
    return real_loss + fake_loss


def g_hinge_loss(fake_logits):
    """Hinge loss for generator (non-saturating)."""
    return -fake_logits.mean()


# ════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🚀 GAN Training — v16")
    print(f"   Device: {device} | Epochs: {cfg.epochs} | Batch: {cfg.batch_size}")
    print(f"   G LR: {cfg.g_lr} | D LR: {cfg.d_lr} | R1: γ={cfg.r1_gamma}")

    # Data
    train_set = MelDataset(cfg.data_dir, 'train', cfg.train_split)
    val_set = MelDataset(cfg.data_dir, 'val', cfg.train_split)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=8 if device.type == 'cuda' else 0,
                              pin_memory=(device.type == 'cuda'), drop_last=True,
                              prefetch_factor=4)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, drop_last=True)

    print(f"   Data: {len(train_set)} train / {len(val_set)} val")

    # Models
    G = Generator(cfg).to(device)
    D = Discriminator(cfg).to(device)

    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"   G params: {g_params:,} ({g_params/1e6:.1f}M)")
    print(f"   D params: {d_params:,} ({d_params/1e6:.1f}M)")

    # Optimizers
    g_opt = torch.optim.Adam(G.parameters(), lr=cfg.g_lr, betas=(cfg.beta1, cfg.beta2))
    d_opt = torch.optim.Adam(D.parameters(), lr=cfg.d_lr, betas=(cfg.beta1, cfg.beta2))

    # AMP
    if cfg.use_amp and device.type == 'cuda':
        # Support both old (torch.cuda.amp) and new (torch.amp) APIs
        try:
            scaler_g = torch.amp.GradScaler('cuda')
            scaler_d = torch.amp.GradScaler('cuda')
            autocast_fn = lambda: torch.amp.autocast('cuda')
        except (TypeError, AttributeError):
            scaler_g = torch.cuda.amp.GradScaler()
            scaler_d = torch.cuda.amp.GradScaler()
            autocast_fn = torch.cuda.amp.autocast
    else:
        scaler_g = None
        scaler_d = None
        autocast_fn = torch.enable_grad

    # For G class loss
    ce_loss = nn.CrossEntropyLoss()

    best_val = float('inf')
    step = 0

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    for epoch in range(cfg.epochs):
        epoch_start = time.time()
        g_loss_sum = 0.0
        d_loss_sum = 0.0
        d_cls_acc_sum = 0.0
        g_cls_acc_sum = 0.0

        G.train()
        D.train()

        for real_mel, real_labels in train_loader:
            real_mel = real_mel.to(device)
            real_labels = real_labels.to(device)
            B = real_mel.shape[0]

            # Pad to clean dimensions
            real_mel = pad_mel(real_mel)

            # ═══ Train DISCRIMINATOR ═══
            requires_grad(G, False)
            requires_grad(D, True)

            d_opt.zero_grad()

            with autocast_fn():
                # Real
                real_adv, real_cls = D(real_mel)
                d_real_cls_loss = ce_loss(real_cls, real_labels)

                # Fake
                z = torch.randn(B, cfg.latent_dim, device=device)
                with torch.no_grad():
                    fake_mel = G(z, real_labels)
                    fake_mel = pad_mel(fake_mel)
                fake_adv, _ = D(fake_mel.detach())

                # D losses
                d_adv_loss = d_hinge_loss(real_adv, fake_adv)
                d_loss = d_adv_loss + cfg.d_class_weight * d_real_cls_loss

                # R1 penalty (periodic)
                if step % cfg.r1_every == 0:
                    r1_pen = compute_r1_penalty(D, real_mel, real_labels)
                    d_loss = d_loss + cfg.r1_gamma * 0.5 * r1_pen

            if scaler_d:
                scaler_d.scale(d_loss).backward()
                scaler_d.step(d_opt)
                scaler_d.update()
            else:
                d_loss.backward()
                d_opt.step()

            d_loss_sum += d_loss.item()
            d_cls_acc_sum += (real_cls.argmax(1) == real_labels).float().mean().item()

            # ═══ Train GENERATOR ═══
            requires_grad(G, True)
            requires_grad(D, False)

            g_opt.zero_grad()

            with autocast_fn():
                z = torch.randn(B, cfg.latent_dim, device=device)
                fake_mel = G(z, real_labels)
                fake_mel = pad_mel(fake_mel)

                fake_adv, fake_cls = D(fake_mel)

                g_adv_loss = g_hinge_loss(fake_adv)
                g_cls_loss = ce_loss(fake_cls, real_labels)
                g_loss = cfg.g_adv_weight * g_adv_loss + cfg.g_class_weight * g_cls_loss

            if scaler_g:
                scaler_g.scale(g_loss).backward()
                scaler_g.step(g_opt)
                scaler_g.update()
            else:
                g_loss.backward()
                g_opt.step()

            g_loss_sum += g_loss.item()
            g_cls_acc_sum += (fake_cls.argmax(1) == real_labels).float().mean().item()

            step += 1

        # ═══ Validation ═══
        G.eval()
        D.eval()
        val_cls_acc = 0.0
        n_val = 0
        with torch.no_grad():
            for val_mel, val_labels in val_loader:
                val_mel = val_mel.to(device)
                val_labels = val_labels.to(device)
                Bv = val_mel.shape[0]

                z = torch.randn(Bv, cfg.latent_dim, device=device)
                fake_mel = G(z, val_labels)
                fake_mel = pad_mel(fake_mel)
                _, fake_cls = D(fake_mel)
                val_cls_acc += (fake_cls.argmax(1) == val_labels).float().sum().item()
                n_val += Bv

        val_acc = val_cls_acc / n_val if n_val > 0 else 0.0

        epoch_time = time.time() - epoch_start
        train_g_loss = g_loss_sum / len(train_loader)
        train_d_loss = d_loss_sum / len(train_loader)
        d_cls_acc = d_cls_acc_sum / len(train_loader) * 100
        g_cls_acc = g_cls_acc_sum / len(train_loader) * 100

        # Progress
        marker, status = "", ""
        if val_acc > best_val:
            best_val = val_acc
            marker = "📉"
            status = " BEST"
            torch.save(G.state_dict(), cfg.generator_path)
            torch.save(D.state_dict(), cfg.discriminator_path)
        else:
            marker = "➡️"

        lr = g_opt.param_groups[0]['lr']
        print(f"── Epoch {epoch+1:3d}/{cfg.epochs} ({epoch_time:.0f}s) ── "
              f"G={train_g_loss:.4f} D={train_d_loss:.4f} "
              f"val_acc={val_acc*100:.1f}% {marker} lr={lr:.2e}{status}")

        # Generate samples periodically (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            # Save periodic checkpoint (resume-safe)
            torch.save(G.state_dict(), f"models/gan_generator_e{epoch+1}.pth")
            generate_samples(G, device, epoch + 1)
            # Clean old checkpoints (keep last 3)
            for old in sorted([f for f in os.listdir('models') if f.startswith('gan_generator_e')])[:-3]:
                os.remove(f'models/{old}')

    print(f"\n✅ Training complete. Best val_acc: {best_val*100:.1f}%")


def generate_samples(generator, device, epoch):
    """Generate and save samples for each class."""
    from src.gan.generate import mel_to_audio
    generator.eval()
    with torch.no_grad():
        for cls_name, cls_idx in cfg.class_to_idx.items():
            z = torch.randn(cfg.num_samples, cfg.latent_dim, device=device)
            labels = torch.full((cfg.num_samples,), cls_idx, device=device, dtype=torch.long)
            mels = generator(z, labels)  # [N, 1, 64, 552]
            save_mel_sample(mels.cpu(), cls_name, f"outputs/gan_v16_e{epoch}_{cls_name}.wav")


def save_mel_sample(mels, class_name, out_path):
    """Convert mel → power → Griffin-Lim → audio."""
    from src.gan.generate import mel_to_audio
    audio = mel_to_audio(mels[0:1])
    torchaudio.save(out_path, audio.cpu(), cfg.sample_rate)
    print(f"   🎵 {out_path}")


if __name__ == '__main__':
    train()
