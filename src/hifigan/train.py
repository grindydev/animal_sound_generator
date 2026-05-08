"""
train.py — Phase 7a: HiFi-GAN Neural Vocoder Training
======================================================

Trains a HiFi-GAN vocoder: learns to convert mel spectrograms back to
realistic audio waveforms. Uses smart_crop for energy-based segment
selection, then computes mel on-the-fly.

FRAMEWORK (matches train_vae.py / finetune_vae.py):
  - CONFIG dict at top → mode/test/train settings
  - Device auto-detection (CUDA > MPS > CPU)
  - AMP mixed precision on CUDA
  - Separate functions: train_epoch(), validate(), training_loop()
  - Checkpoint resume support

MODES:
  "test"  — 5 epochs, batch=4  → quick dev smoke test
  "train" — 30 epochs, batch=12 → full training

Run:
    python src/hifigan/train.py
"""
import os
import sys
import time
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.smart_crop import smart_crop
from src.hifigan.config import config as cfg
from src.hifigan.generator import HiFiGANGenerator
from src.hifigan.discriminator import Discriminator
from src.hifigan.losses import MelL1Loss, generator_loss, discriminator_loss
from src.hifigan.utils import save_checkpoint, load_checkpoint


# ═══════════════════════════════════════════════════════════════
#  CONFIG (EDIT ONLY THIS SECTION)
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    "mode": "train",         # "train" = full GAN | "meltrain" = mel-only (debug) | "diag" = quick test
    "device": "auto",        # "auto", "cuda", "mps", or "cpu"

    # ── Shared ──────────────────────────────────────────
    "data_dir": cfg.data_dir,
    "segment_size": cfg.segment_size,
    "save_interval": cfg.save_interval,

    "diag": {
        "num_epochs": 5,
        "batch_size": 16,
        "num_workers": 2,
    },

    "test": {
        "num_epochs": 5,
        "batch_size": 4,
        "num_workers": 1,
    },

    "train": {
        "num_epochs": 30,
        "batch_size": 8,       # GAN needs less — discriminators use more memory
        "num_workers": 0,       # 0 = no multiprocessing → avoid silent data bugs
    },

    "meltrain": {
        "num_epochs": 30,
        "batch_size": 16,
        "num_workers": 0,       # 0 = no multiprocessing → avoids silent data loading bugs on CUDA
    },
}

# ═══════════════════════════════════════════════════════════════
#  APPLY CONFIG
# ═══════════════════════════════════════════════════════════════

MODE = CONFIG["mode"]
if MODE not in CONFIG:
    SETTINGS = CONFIG["train"]  # meltrain reuses train settings
else:
    SETTINGS = CONFIG[MODE]

NUM_EPOCHS = SETTINGS["num_epochs"]
BATCH_SIZE = SETTINGS["batch_size"]
NUM_WORKERS = SETTINGS["num_workers"]
SEGMENT_SIZE = CONFIG["segment_size"]

BEST_MODEL_PATH = f"models/hifigan_generator_{MODE}.pth"
CHECKPOINT_DIR = os.path.join(cfg.checkpoint_dir, MODE)

# ═══════════════════════════════════════════════════════════════
#  DEVICE SETUP (matches train_vae.py)
# ═══════════════════════════════════════════════════════════════

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
scaler = GradScaler() if use_amp else None

# ═══════════════════════════════════════════════════════════════
#  AUDIO LOADER (torchaudio → soundfile fallback)
# ═══════════════════════════════════════════════════════════════

def _load_audio(path: str):
    """Load audio file. Tries torchaudio first, falls back to soundfile.

    torchaudio 2.11+ requires libtorchcodec which needs FFmpeg shared libs.
    On systems without FFmpeg, soundfile works directly (reads WAV natively).
    """
    # ── Try torchaudio first (fast, native) ──
    try:
        return torchaudio.load(path)
    except Exception:
        pass

    # ── Fallback: soundfile (no FFmpeg needed for WAV) ──
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype='float32')
        # soundfile returns [samples] or [samples, channels]
        wav = torch.from_numpy(data)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # [1, samples]
        else:
            wav = wav.transpose(0, 1)  # [channels, samples]
        return wav, sr
    except Exception as e:
        import warnings
        warnings.warn(f"⚠️  Audio load FAILED (all backends): {path} → {e}")
        return None, None


# ═══════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════

class HiFiGANDataset(Dataset):
    """Loads audio files, applies smart_crop for energy-based selection."""

    def __init__(self, data_dir: str, segment_size: int, split: str = "train"):
        self.data_dir = data_dir
        self.segment_size = segment_size
        self.files = []

        for cls_name in sorted(os.listdir(data_dir)):
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith('.wav'):
                    # ABSOLUTE PATHS — relative paths break in multiprocessing workers
                    self.files.append(os.path.abspath(os.path.join(cls_dir, fname)))

        np.random.seed(42)
        np.random.shuffle(self.files)
        split_idx = int(len(self.files) * 0.9)
        self.files = self.files[:split_idx] if split == "train" else self.files[split_idx:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        wav, sr = _load_audio(path)
        if wav is None:
            return torch.zeros(1, self.segment_size)

        if sr != cfg.sample_rate:
            wav = torchaudio.transforms.Resample(sr, cfg.sample_rate)(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if wav.shape[-1] <= self.segment_size:
            pad = torch.zeros(1, self.segment_size - wav.shape[-1])
            wav = torch.cat([wav, pad], dim=-1)
        else:
            crops = smart_crop(
                wav, crop_samples=self.segment_size,
                threshold_db=cfg.smart_crop_threshold_db,
                num_crops=1, merge_gap_samples=cfg.smart_crop_merge_gap,
            )
            wav = crops[0]

        return wav  # [1, segment_size]


# ═══════════════════════════════════════════════════════════════
#  MEL COMPUTATION (on-the-fly)
# ═══════════════════════════════════════════════════════════════

def compute_mel(audio: torch.Tensor) -> torch.Tensor:
    """Compute normalized mel spectrogram matching VAE training."""
    from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

    d = audio.device
    mel_tfm = MelSpectrogram(
        sample_rate=cfg.sample_rate, n_fft=cfg.n_fft,
        hop_length=cfg.hop_length, win_length=cfg.win_length,
        n_mels=cfg.n_mels, f_min=cfg.f_min, f_max=cfg.f_max, power=2,
    ).to(d)

    db_tfm = AmplitudeToDB(stype='power', top_db=None).to(d)
    spec = mel_tfm(audio.squeeze(1))
    return (db_tfm(spec) - cfg.norm_mean) / cfg.norm_std


# ═══════════════════════════════════════════════════════════════
#  TRAIN / VALIDATE
# ═══════════════════════════════════════════════════════════════

def train_epoch_mel_only(generator, train_loader, opt_g, mel_loss_fn):
    """Train one epoch — MEL ONLY, no discriminator. Returns avg_mel."""
    from tqdm import tqdm

    generator.train()
    total_mel = 0.0
    pbar = tqdm(train_loader, desc="  Train", leave=False)

    for audio in pbar:
        audio = audio.to(device)
        real_mel = compute_mel(audio)
        n_frames = real_mel.shape[-1]
        target_len = n_frames * cfg.hop_length

        if audio.shape[-1] < target_len:
            real_trim = F.pad(audio, (0, target_len - audio.shape[-1]))
        else:
            real_trim = audio[..., :target_len]

        fake = generator(real_mel, target_length=target_len)
        mel_loss = mel_loss_fn(fake, real_trim)
        time_loss = F.l1_loss(fake, real_trim)  # time-domain: prevents tanh saturation
        loss = mel_loss + 1.0 * time_loss

        opt_g.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt_g.step()

        total_mel += mel_loss.item()
        pbar.set_postfix({"mel": f"{mel_loss.item():.1f}"})

    return total_mel / len(train_loader)


def train_epoch(generator, discriminator, train_loader,
                opt_g, opt_d, mel_loss_fn):
    """Train one epoch. Returns (avg_g, avg_d, avg_mel)."""
    from tqdm import tqdm

    generator.train()
    discriminator.train()

    total_g, total_d, total_mel = 0.0, 0.0, 0.0
    pbar = tqdm(train_loader, desc="  Train", leave=False)

    for audio in pbar:
        audio = audio.to(device)

        real_mel = compute_mel(audio)
        n_frames = real_mel.shape[-1]
        target_len = n_frames * cfg.hop_length

        # Pad real audio to match exact mel frame count
        if audio.shape[-1] < target_len:
            real_trim = F.pad(audio, (0, target_len - audio.shape[-1]))
        else:
            real_trim = audio[..., :target_len]

        fake = generator(real_mel, target_length=target_len)

        # ── Discriminator ───────────────────────────────
        opt_d.zero_grad()

        # Add noise to D input → harder to memorize
        d_real = real_trim + torch.randn_like(real_trim) * 0.01
        d_fake = fake.detach() + torch.randn_like(fake) * 0.01

        if use_amp:
            with autocast(device_type="cuda"):
                r_score, r_feat = discriminator(d_real)
                f_score_d, _ = discriminator(d_fake)
                d_loss, d_dict = discriminator_loss(r_score, f_score_d)
            scaler.scale(d_loss).backward()
            scaler.unscale_(opt_d)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            scaler.step(opt_d)
            scaler.update()
        else:
            r_score, r_feat = discriminator(d_real)
            f_score_d, _ = discriminator(d_fake)
            d_loss, d_dict = discriminator_loss(r_score, f_score_d)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            opt_d.step()

        # ── Generator ───────────────────────────────────
        opt_g.zero_grad()

        if use_amp:
            with autocast(device_type="cuda"):
                f_score_g, f_feat_g = discriminator(fake)
                g_loss, g_dict = generator_loss(
                    fake, real_trim, f_score_g, f_feat_g, r_feat, mel_loss_fn,
                    lambda_mel=cfg.lambda_mel,
                    lambda_fm=cfg.lambda_fm,
                    lambda_adv=cfg.lambda_adv,
                )
            scaler.scale(g_loss).backward()
            scaler.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            scaler.step(opt_g)
            scaler.update()
        else:
            f_score_g, f_feat_g = discriminator(fake)
            g_loss, g_dict = generator_loss(
                fake, real_trim, f_score_g, f_feat_g, r_feat, mel_loss_fn,
                lambda_mel=cfg.lambda_mel,
                lambda_fm=cfg.lambda_fm,
                lambda_adv=cfg.lambda_adv,
            )
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt_g.step()

        total_g += g_dict["g_total"]
        total_d += d_dict["d_total"]
        total_mel += g_dict["g_mel"]

        pbar.set_postfix({
            "G": f"{g_loss.item():.2f}",
            "D": f"{d_loss.item():.2f}",
            "mel": f"{g_dict['g_mel']:.1f}",
        })

    n = len(train_loader)
    return total_g / n, total_d / n, total_mel / n


def validate(generator, val_loader, mel_loss_fn):
    """Validate — returns average mel loss."""
    generator.eval()
    total_mel = 0.0

    with torch.no_grad():
        for va in val_loader:
            va = va.to(device)
            vm = compute_mel(va)
            vt = vm.shape[-1] * cfg.hop_length
            vf = generator(vm, target_length=vt)
            if va.shape[-1] < vt:
                va_trim = F.pad(va, (0, vt - va.shape[-1]))
            else:
                va_trim = va[..., :vt]
            total_mel += mel_loss_fn(vf, va_trim).item()

    return total_mel / max(len(val_loader), 1)


# ═══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def training_loop():
    """Full HiFi-GAN training with test/train mode, AMP, checkpoint resume."""

    # ═════════════════════════════════════════════════════════
    #  BANNER (inside training_loop to avoid multiprocessing duplication)
    # ═════════════════════════════════════════════════════════
    print(f"\n🔧 HiFi-GAN → {MODE.upper()} MODE")
    if is_cuda:
        print(f"   GPU:    {torch.cuda.get_device_name(0)}")
    print(f"   Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | Workers: {NUM_WORKERS}")
    print(f"   Segment: {SEGMENT_SIZE} samples (~{SEGMENT_SIZE/cfg.sample_rate:.1f}s)")
    print(f"   Mixed precision: {'yes' if use_amp else 'no'}")
    print(f"   Best model → {BEST_MODEL_PATH}")

    # ═════════════════════════════════════════════════════════
    #  DIAGNOSTIC: mel-only, no discriminator, small segment
    # ═════════════════════════════════════════════════════════
    if MODE == "diag":
        diag_segment = 8192  # 0.37s, ~4× faster
        print(f"\n🔬 DIAGNOSTIC: Mel-only training, segment={diag_segment}")
        print(f"   No discriminator. If mel drops → GAN setup is broken.")
        print(f"   If mel flatlines → generator is too weak.\n")

        train_ds = HiFiGANDataset(CONFIG["data_dir"], diag_segment, split="train")
        val_ds = HiFiGANDataset(CONFIG["data_dir"], diag_segment, split="val")

        train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=False, drop_last=True)
        val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=False)

        generator = HiFiGANGenerator(cfg).to(device)
        opt_g = torch.optim.Adam(generator.parameters(), lr=cfg.learning_rate, betas=cfg.adam_betas)
        mel_loss_fn = MelL1Loss(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft,
            hop_length=cfg.hop_length, n_mels=cfg.n_mels,
            f_min=cfg.f_min, f_max=cfg.f_max,
        ).to(device)

        # ── DATA VALIDATION ─────────────────────────────────
        _check = next(iter(train_loader))
        _check_max = _check.abs().max().item()
        print(f"   📊 First batch: shape={tuple(_check.shape)}, "
              f"min={_check.min():.4f}, max={_check.max():.4f}, "
              f"mean_abs={_check.abs().mean():.4f}")
        if _check_max < 1e-6:
            print("   🚨 ALL ZEROS — audio files not loading correctly.")
            return None

        prev_val = float("inf")
        for epoch in range(NUM_EPOCHS):
            t0 = time.time()
            avg_mel = train_epoch_mel_only(generator, train_loader, opt_g, mel_loss_fn)
            val_mel = validate(generator, val_loader, mel_loss_fn)
            dt = time.time() - t0
            trend = "📉" if val_mel < prev_val else "➡️"
            print(f"── Epoch {epoch+1:3d}/{NUM_EPOCHS} ({dt:.0f}s) ── mel={avg_mel:.4f} val={val_mel:.4f} {trend}")
            prev_val = val_mel

        print(f"\n✅ Diagnostic complete.")
        Verdict = "DROPS" if val_mel < 0.5 else "FLAT"
        print(f"   Verdict: Mel {Verdict} → {'GAN setup broken' if Verdict == 'DROPS' else 'generator too weak'}")
        return None

    # ═════════════════════════════════════════════════════════
    #  MEL-ONLY TRAINING: no discriminator, pure mel→audio
    # ═════════════════════════════════════════════════════════
    if MODE == "meltrain":
        print(f"\n🎵 MEL-ONLY TRAINING — No GAN, pure reconstruction")
        print(f"   Segment: {SEGMENT_SIZE} | Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | Workers: {NUM_WORKERS}\n")

        train_ds = HiFiGANDataset(CONFIG["data_dir"], SEGMENT_SIZE, split="train")
        val_ds   = HiFiGANDataset(CONFIG["data_dir"], SEGMENT_SIZE, split="val")
        print(f"   Data: {len(train_ds)} train / {len(val_ds)} val files")

        # pin_memory only useful with workers > 0
        pin_mem = NUM_WORKERS > 0 and is_cuda
        train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=pin_mem, drop_last=True)
        val_loader   = DataLoader(val_ds, BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=pin_mem)

        # ── DATA VALIDATION ─────────────────────────────────
        # Catch silent zero-data bugs BEFORE wasting epochs
        _check = next(iter(train_loader))
        _check_max = _check.abs().max().item()
        print(f"   📊 First batch: shape={tuple(_check.shape)}, "
              f"min={_check.min():.4f}, max={_check.max():.4f}, "
              f"mean_abs={_check.abs().mean():.4f}")
        if _check_max < 1e-6:
            print("   🚨 ALL ZEROS — audio files not loading correctly. Check data_dir and file format.")
            print("   🚨 Training stopped to save time.")
            return None

        generator = HiFiGANGenerator(cfg).to(device)
        print(f"   Generator: {sum(p.numel() for p in generator.parameters()):,} params")

        opt_g = torch.optim.Adam(generator.parameters(), lr=cfg.learning_rate, betas=cfg.adam_betas)
        sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=cfg.lr_decay)
        mel_loss_fn = MelL1Loss(cfg.sample_rate, cfg.n_fft, cfg.hop_length, cfg.n_mels,
                                cfg.f_min, cfg.f_max).to(device)

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cfg.model_dir, exist_ok=True)
        best_val = float("inf")
        BEST_PATH = os.path.join(cfg.model_dir, f"hifigan_generator_{MODE}_best.pth")

        for epoch in range(NUM_EPOCHS):
            t0 = time.time()
            avg_mel = train_epoch_mel_only(generator, train_loader, opt_g, mel_loss_fn)
            sched_g.step()
            val_mel = validate(generator, val_loader, mel_loss_fn)
            dt = time.time() - t0

            trend = "📉" if val_mel < best_val else "➡️"
            print(f"── Epoch {epoch+1:3d}/{NUM_EPOCHS} ({dt:.0f}s) ── mel={avg_mel:.4f} val={val_mel:.4f} {trend} lr={sched_g.get_last_lr()[0]:.2e}")

            if val_mel < best_val:
                best_val = val_mel
                torch.save({"generator": generator.state_dict(), "config": cfg.__dict__}, BEST_PATH)

        torch.save({"generator": generator.state_dict(), "config": cfg.__dict__}, BEST_MODEL_PATH)
        print(f"\n💾 Saved: {BEST_MODEL_PATH}")
        print(f"   Best val mel: {best_val:.4f}")
        return None

    # ═════════════════════════════════════════════════════════
    #  GAN TRAINING (test / train modes)
    # ═════════════════════════════════════════════════════════

    # ── Models ──────────────────────────────────────────
    generator = HiFiGANGenerator(cfg).to(device)
    discriminator = Discriminator().to(device)

    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"\n   Generator:     {g_params:,} params")
    print(f"   Discriminator: {d_params:,} params")
    print(f"   Total:         {g_params + d_params:,} params")

    # ── Optimizers ──────────────────────────────────────
    opt_g = torch.optim.Adam(generator.parameters(), lr=cfg.learning_rate, betas=cfg.adam_betas)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.learning_rate, betas=cfg.adam_betas)

    sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=cfg.lr_decay)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=cfg.lr_decay)

    # ── Loss ────────────────────────────────────────────
    mel_loss_fn = MelL1Loss(
        sample_rate=cfg.sample_rate, n_fft=cfg.n_fft,
        hop_length=cfg.hop_length, n_mels=cfg.n_mels,
        f_min=cfg.f_min, f_max=cfg.f_max,
    ).to(device)

    # ── Data ────────────────────────────────────────────
    train_ds = HiFiGANDataset(CONFIG["data_dir"], SEGMENT_SIZE, split="train")
    val_ds = HiFiGANDataset(CONFIG["data_dir"], SEGMENT_SIZE, split="val")

    print(f"\n✅ Data loaded: {len(train_ds)} train / {len(val_ds)} val")

    # pin_memory only useful with workers > 0
    pin_mem = NUM_WORKERS > 0 and is_cuda
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin_mem, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin_mem, drop_last=False,
    )

    # ── DATA VALIDATION ─────────────────────────────────
    _check = next(iter(train_loader))
    _check_max = _check.abs().max().item()
    print(f"   📊 First batch: shape={tuple(_check.shape)}, "
          f"min={_check.min():.4f}, max={_check.max():.4f}, "
          f"mean_abs={_check.abs().mean():.4f}")
    if _check_max < 1e-6:
        print("   🚨 ALL ZEROS — audio files not loading correctly. Check data_dir and file format.")
        print("   🚨 Training stopped to save time.")
        return None

    # ── Resume ──────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)
    start_epoch = load_checkpoint(
        generator, discriminator, opt_g, opt_d,
        CHECKPOINT_DIR, device,
    )
    if start_epoch > 0:
        print(f"   Resumed from epoch {start_epoch}")

    # ── Train ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"🚀 HiFi-GAN TRAINING — {MODE.upper()} MODE")
    print(f"   Device: {device} | Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"   Saving last model → {BEST_MODEL_PATH}")
    print(f"{'='*60}\n")

    best_val_mel = float("inf")
    BEST_PATH = os.path.join(cfg.model_dir, f"hifigan_generator_{MODE}_best.pth")

    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()

        avg_g, avg_d, avg_mel = train_epoch(
            generator, discriminator, train_loader,
            opt_g, opt_d, mel_loss_fn,
        )

        sched_g.step()
        sched_d.step()
        lr = sched_g.get_last_lr()[0]

        val_mel = validate(generator, val_loader, mel_loss_fn)

        dt = time.time() - t0
        print(
            f"── Epoch {epoch+1:3d}/{NUM_EPOCHS} "
            f"({dt:.0f}s) ── "
            f"G={avg_g:.4f} D={avg_d:.4f} "
            f"mel={avg_mel:.4f} val={val_mel:.4f} lr={lr:.2e}"
        )

        if epoch % CONFIG["save_interval"] == 0 or epoch == NUM_EPOCHS - 1:
            save_checkpoint(
                generator, discriminator, opt_g, opt_d,
                epoch + 1, CHECKPOINT_DIR,
            )

        if val_mel < best_val_mel:
            best_val_mel = val_mel
            torch.save(
                {"generator": generator.state_dict(), "config": cfg.__dict__},
                BEST_PATH,
            )

    torch.save(
        {"generator": generator.state_dict(), "config": cfg.__dict__},
        BEST_MODEL_PATH,
    )
    print(f"\n💾 Final model saved to: {BEST_MODEL_PATH}")
    print("✅ Training complete!")

    return generator


if __name__ == "__main__":
    trained_generator = training_loop()
