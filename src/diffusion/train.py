"""
train.py — V9: Diffusion U-Net Training (x₀ + GAN + class-balanced)
===============================================

Trains a U-Net diffusion model to denoise spectrograms. The U-Net learns
to predict noise at various timesteps, enabling img2img refinement of
VAE-generated spectrograms at inference time.

TRAINING APPROACH (Standard DDPM on real spectrograms):
  1. Load real audio → compute mel spectrogram → x_0 (clean)
  2. Pick random timestep t → add noise → x_t
  3. U-Net(x_t, t, class_label) → predicted noise
  4. Loss = MSE(predicted_noise, actual_noise)

INFERENCE (img2img refinement):
  1. VAE generates blurry spectrogram → x_vae
  2. Add noise at t = strength * T → x_t
  3. Denoise N steps → sharp spectrogram
  4. HiFi-GAN converts → crisp audio

FRAMEWORK (matches src/hifigan/train.py):
  - CONFIG dict at top → mode/test/train settings
  - Device auto-detection (CUDA > MPS > CPU)
  - Separate functions: train_epoch(), validate(), training_loop()
  - Checkpoint resume support
  - Progress tracking with 📉/➡️ indicators

MODES:
  "test"  — 5 epochs, batch=4  → quick dev smoke test
  "train" — 50 epochs, batch=8 → full training

Run:
    python src/diffusion/train.py
"""
import os
import sys
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.diffusion.config import config as cfg
from src.diffusion.unet import SpectrogramUNet
from src.diffusion.diffusion import DiffusionProcess
from smart_crop import smart_crop


# ═══════════════════════════════════════════════════════════════
#  CONFIG (EDIT ONLY THIS SECTION)
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    "mode": "train",         # "test" = 5 epoch smoke test | "train" = full training
    "device": "auto",        # "auto", "cuda", "mps", or "cpu"

    # ── Shared ──────────────────────────────────────────
    "data_dir": cfg.data_dir,
    "segment_frames": cfg.segment_frames,
    "save_interval": cfg.save_interval,
    "log_interval": cfg.log_interval,

    # ── Training mode: DDPM on REAL mels (not VAE refinement) ──
    # vae_mix_ratio=0 → train on real mel spectrograms only
    # This makes diffusion the PRIMARY generator (not a refiner)
    "vae_checkpoint": None,         # no VAE needed for DDPM training
    "vae_mix_ratio": 0.0,           # 0% VAE = 100% real mels

    "test": {
        "num_epochs": 5,
        "batch_size": 8,            # v6: 15M UNet (was 4)
        "num_workers": 2,
    },

    "train": {
        "num_epochs": 150,           # v10: 150 for ESC-50
        "batch_size": 8,            # v6: 15M UNet (was 4)
        "num_workers": 4,
        "gradient_accumulation_steps": 4,  # effective batch = 32
    },
}

# ═══════════════════════════════════════════════════════════════
#  APPLY CONFIG
# ═══════════════════════════════════════════════════════════════

MODE = CONFIG["mode"]
SETTINGS = CONFIG[MODE] if MODE in CONFIG else CONFIG["train"]

NUM_EPOCHS = SETTINGS["num_epochs"]
BATCH_SIZE = SETTINGS["batch_size"]
NUM_WORKERS = SETTINGS["num_workers"]
GRADIENT_ACCUMULATION_STEPS = SETTINGS.get("gradient_accumulation_steps", 1)
SEGMENT_FRAMES = CONFIG["segment_frames"]
LOG_INTERVAL = CONFIG["log_interval"]
EMA_DECAY = 0.9999

BEST_MODEL_PATH = f"models/diffusion_unet_{MODE}.pth"


# ═══════════════════════════════════════════════════════════════
#  DEVICE SETUP (matches hifigan/train.py)
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

use_amp = is_cuda  # mixed precision only supported on CUDA



# ═══════════════════════════════════════════════════════════════
#  AUDIO LOADER
# ═══════════════════════════════════════════════════════════════

def _load_audio(path: str):
    """Load audio file. Tries torchaudio first, falls back to soundfile."""
    try:
        import torchaudio
        return torchaudio.load(path)
    except Exception:
        pass

    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype='float32')
        wav = torch.from_numpy(data)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.transpose(0, 1)
        return wav, sr
    except Exception as e:
        import warnings
        warnings.warn(f"⚠️  Audio load FAILED: {path} → {e}")
        return None, None


# ═══════════════════════════════════════════════════════════════
#  CLASS MAPPING
# ═══════════════════════════════════════════════════════════════

CLASSES = ['Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen', 'Noise']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}


# ═══════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════

class DiffusionDataset(Dataset):
    """
    Loads audio files, computes mel spectrograms, pairs with class labels.

    Supports mixed training: randomly returns VAE-reconstructed mels instead of
    real mels, teaching the diffusion model to denoise VAE-quality data.

    Returns:
        mel: normalized mel spectrogram [1, 64, segment_frames]
        label: class index [1]
    """

    def __init__(self, data_dir: str, segment_frames: int, split: str = "train",
                 vae_model=None, vae_mix_ratio: float = 0.0):
        """
        Args:
            data_dir: path to animal_audio directory
            segment_frames: number of mel time frames per sample
            split: "train" or "val"
            vae_model: trained SimpleAudioVAE instance (None = no VAE mixing)
            vae_mix_ratio: fraction of samples to replace with VAE reconstructions
                           (0.0 = all real, 0.3 = 30% VAE, 1.0 = all VAE)
        """
        self.data_dir = data_dir
        self.segment_frames = segment_frames
        self.vae_model = vae_model
        self.vae_mix_ratio = vae_mix_ratio
        self.samples = []  # list of (path, class_idx)

        for cls_name in sorted(os.listdir(data_dir)):
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            if cls_name not in CLASS_TO_IDX:
                continue
            cls_idx = CLASS_TO_IDX[cls_name]
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith('.wav'):
                    self.samples.append((
                        os.path.abspath(os.path.join(cls_dir, fname)),
                        cls_idx,
                    ))

        # Shuffle and split
        np.random.seed(42)
        np.random.shuffle(self.samples)
        split_idx = int(len(self.samples) * getattr(cfg, 'train_fraction', 0.9))
        self.samples = self.samples[:split_idx] if split == "train" else self.samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        wav, sr = _load_audio(path)

        if wav is None:
            return torch.zeros(1, 64, self.segment_frames), torch.tensor(0, dtype=torch.long)

        # Resample if needed
        if sr != cfg.sample_rate:
            import torchaudio
            wav = torchaudio.transforms.Resample(sr, cfg.sample_rate)(wav)

        # Mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Smart crop: pick energy-rich region before mel (matches HiFi-GAN)
        crop_samples = self.segment_frames * cfg.hop_length
        if wav.shape[-1] <= crop_samples:
            pad = torch.zeros(1, crop_samples - wav.shape[-1])
            wav = torch.cat([wav, pad], dim=-1)
        else:
            crops = smart_crop(
                wav, crop_samples=crop_samples, threshold_db=-30.0,
                num_crops=1, merge_gap_samples=4410,
            )
            wav = crops[0]

        # Compute mel
        mel = compute_mel(wav)  # [1, 64, ~segment_frames]

        # V8: Strong augmentation (freq mask, time mask, noise, gain, dropout)
        if getattr(cfg, 'augment', False):
            mel = strong_augment_mel(mel)

        # Trim/pad to exact segment_frames (mel is 3D: [1, 64, T] = [ch, freq, time])
        T = mel.shape[-1]
        if T > self.segment_frames:
            mel = mel[..., :self.segment_frames]
        elif T < self.segment_frames:
            pad_shape = list(mel.shape)
            pad_shape[-1] = self.segment_frames - T
            pad = torch.zeros(pad_shape)
            mel = torch.cat([mel, pad], dim=-1)

        # VAE mix-in: randomly replace real mel with VAE-GENERATED sample.
        # Uses vae.sample() (generation) NOT vae.forward() (reconstruction) —
        # this matches what the diffusion will see at inference time.
        if self.vae_model is not None and self.vae_mix_ratio > 0:
            if np.random.random() < self.vae_mix_ratio:
                vae_device = next(self.vae_model.parameters()).device
                with torch.no_grad():
                    # Generate from scratch (same as generate.py does at inference)
                    vae_gen = self.vae_model.sample(
                        label, num_samples=1, device=vae_device, temperature=0.7,
                    )  # [1, 1, 64, T]
                    # Remove channel dim to match dataset format [1, 64, T]
                    vae_gen = torch.nn.functional.interpolate(
                        vae_gen, size=(64, self.segment_frames), mode='bilinear'
                    )
                    mel = vae_gen[:, 0, :, :].cpu()  # [1, 64, segment_frames]

        return mel, torch.tensor(label, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════
#  V9: CLASS-BALANCED DATASET (oversample rare classes)
# ═══════════════════════════════════════════════════════════════

class BalancedDiffusionDataset(Dataset):
    """
    Class-balanced dataset: rare classes get equal exposure.
    
    Each epoch: every class appears `max_per_class` times.
    - Dog (750) appears naturally
    - Frog (61) repeated 12× with different augmentation each pass
    - Noise capped at 200 (Option B)
    
    Effective: ~1,400 samples per epoch (200 × 7 classes).
    """
    def __init__(self, data_dir: str, segment_frames: int, split: str = "train"):
        self.data_dir = data_dir
        self.segment_frames = segment_frames
        self.class_samples = {c: [] for c in range(cfg.num_classes)}

        # Collect samples grouped by class
        all_samples = []
        for cls_name in os.listdir(data_dir):
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_dir) or cls_name not in CLASS_TO_IDX:
                continue
            cls_idx = CLASS_TO_IDX[cls_name]
            for fname in os.listdir(cls_dir):
                if fname.endswith('.wav'):
                    all_samples.append((
                        os.path.abspath(os.path.join(cls_dir, fname)),
                        cls_idx,
                    ))

        # Split train/val PER CLASS (preserves rare classes in both)
        np.random.seed(42)
        train_samples = {}
        val_samples = {}
        for c in range(cfg.num_classes):
            class_files = [s for s in all_samples if s[1] == c]
            np.random.shuffle(class_files)
            split_idx = int(len(class_files) * cfg.train_fraction)
            if split == "train":
                self.class_samples[c] = class_files[:split_idx]
            else:
                self.class_samples[c] = class_files[split_idx:]
            
            # V9 Option B: cap Noise class
            if c == CLASS_TO_IDX.get('Noise', 7) and cfg.noise_max_samples > 0:
                self.class_samples[c] = self.class_samples[c][:cfg.noise_max_samples]

        # Max samples in any class → target length per epoch
        self.max_per_class = max(len(v) for v in self.class_samples.values())
        self.num_classes = cfg.num_classes

    def __len__(self):
        return self.max_per_class * self.num_classes

    def _load_mel(self, path, label):
        """Load audio, compute mel, apply augmentation."""
        wav, sr = _load_audio(path)
        if wav is None:
            return torch.zeros(1, 64, self.segment_frames), torch.tensor(0, dtype=torch.long)

        if sr != cfg.sample_rate:
            import torchaudio
            wav = torchaudio.transforms.Resample(sr, cfg.sample_rate)(wav)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        crop_samples = self.segment_frames * cfg.hop_length
        if wav.shape[-1] <= crop_samples:
            pad = torch.zeros(1, crop_samples - wav.shape[-1])
            wav = torch.cat([wav, pad], dim=-1)
        else:
            crops = smart_crop(wav, crop_samples=crop_samples, threshold_db=-30.0,
                               num_crops=1, merge_gap_samples=4410)
            wav = crops[0]

        mel = compute_mel(wav)

        # Strong augmentation (different each oversample pass)
        if getattr(cfg, 'augment', False):
            mel = strong_augment_mel(mel)

        T = mel.shape[-1]
        if T > self.segment_frames:
            mel = mel[..., :self.segment_frames]
        elif T < self.segment_frames:
            mel = torch.nn.functional.pad(mel, (0, self.segment_frames - T))

        return mel, torch.tensor(label, dtype=torch.long)

    def __getitem__(self, idx):
        cls = idx % self.num_classes
        samples = self.class_samples[cls]
        # Wrap around: rare classes repeat, each pass gets different augmentation
        s_idx = (idx // self.num_classes) % len(samples)
        path, label = samples[s_idx]
        return self._load_mel(path, label)


# ═══════════════════════════════════════════════════════════════
#  MEL COMPUTATION (on-the-fly, matches data_loader.py format)
# ═══════════════════════════════════════════════════════════════

# Pre-create mel transform (cached per device for efficiency)
_mel_tfm_cache = {}
_db_tfm_cache = {}


def compute_mel(audio: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized mel spectrogram matching VAE training format.
    Uses the same params as data_loader.py and hifigan/config.py.

    MelSpectrogram on [time] returns [1, n_mels, time] (includes batch dim).
    We add unsqueeze(0) → [1, 1, n_mels, time] so after DataLoader collation:
    [B, 1, n_mels, T] = [B, spec_channels, n_mels, segment_frames] — U-Net expects this.
    """
    from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

    norm_mean = -18.4903
    norm_std = 19.8031
    d = audio.device

    if d not in _mel_tfm_cache:
        _mel_tfm_cache[d] = MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft,
            hop_length=cfg.hop_length, win_length=cfg.n_fft,
            n_mels=cfg.n_mels, f_min=cfg.f_min, f_max=cfg.f_max, power=2,
        ).to(d)
        _db_tfm_cache[d] = AmplitudeToDB(stype='power', top_db=80).to(d)

    mel_tfm = _mel_tfm_cache[d]
    db_tfm = _db_tfm_cache[d]

    # audio is [1, samples] — squeeze batch dim for MelSpectrogram (expects 1D [time])
    spec = mel_tfm(audio.squeeze(0))  # [n_mels, time_frames]
    mel = (db_tfm(spec) - norm_mean) / norm_std  # [n_mels, time_frames]
    # Return [1, n_mels, time_frames] — DataLoader collates to [B, 1, n_mels, T] for U-Net
    return mel.unsqueeze(0)  # [1, n_mels, time_frames]


# ═══════════════════════════════════════════════════════════════
#  DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════

def spec_augment(mel: torch.Tensor, freq_mask: int = 8, time_mask: int = 50) -> torch.Tensor:
    """SpecAugment: randomly mask frequency and time bands."""
    # mel: [B, 1, F, T]
    B, C, F, T = mel.shape
    # Frequency masking
    for _ in range(2):
        f = np.random.randint(0, freq_mask + 1)
        if f > 0:
            f0 = np.random.randint(0, F - f)
            mel[:, :, f0:f0+f, :] = 0
    # Time masking
    for _ in range(2):
        t = np.random.randint(0, time_mask + 1)
        if t > 0:
            t0 = np.random.randint(0, T - t)
            mel[:, :, :, t0:t0+t] = 0
    return mel


# ═══════════════════════════════════════════════════════════════
#  V8: STRONG AUGMENTATION (freq mask, time mask, noise, gain, dropout)
# ═══════════════════════════════════════════════════════════════

def strong_augment_mel(mel: torch.Tensor) -> torch.Tensor:
    """
    Strong mel augmentation: model never sees same sample twice.
    - Frequency masking: zero 2-8 consecutive mel bins
    - Time masking: zero 20-80 consecutive frames
    - Gaussian noise: σ=0.05 additive noise
    - Gain jitter: multiply by [0.8, 1.2]
    - Bin dropout: random 10% of bins zeroed
    """
    if not getattr(cfg, 'strong_augment', False):
        return mel

    mel = mel.clone()  # Don't modify original
    n_freq, n_time = mel.shape[-2], mel.shape[-1]

    # 1. Frequency masking (2-8 consecutive mel bins → zero)
    if np.random.random() < 0.7:
        f_mask = np.random.randint(2, 9)
        f0 = np.random.randint(0, n_freq - f_mask)
        mel[..., f0:f0+f_mask, :] = 0

    # 2. Time masking (20-80 consecutive frames → zero)
    if np.random.random() < 0.7:
        t_mask = np.random.randint(20, 81)
        t0 = np.random.randint(0, n_time - t_mask)
        mel[..., :, t0:t0+t_mask] = 0

    # 3. Gaussian noise (σ=0.05)
    if np.random.random() < 0.5:
        mel = mel + torch.randn_like(mel) * 0.05

    # 4. Gain jitter
    gain = np.random.uniform(0.8, 1.2)
    mel = mel * gain

    # 5. Random bin dropout (10%)
    if np.random.random() < 0.3:
        mask_bins = torch.rand(n_freq, device=mel.device) < 0.1
        mel[..., mask_bins, :] = 0

    return mel


def freq_weighted_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    L1 loss weighted by frequency — higher frequencies matter 3× more.
    Forces model to learn harmonics, not just low-frequency average.
    """
    weight_max = getattr(cfg, 'freq_weight_max', 1.0)
    if weight_max <= 1.0:
        return F.l1_loss(pred, target)

    n_mels = cfg.n_mels
    weights = torch.linspace(1.0, weight_max, n_mels, device=pred.device)
    weights = weights.view(1, 1, n_mels, 1)  # [1,1,F,1] for broadcasting

    error = (pred - target).abs()
    weighted_error = error * weights
    return weighted_error.mean()


# ═══════════════════════════════════════════════════════════════
#  V8 FIX C: PATCHGAN DISCRIMINATOR
# ═══════════════════════════════════════════════════════════════

class MelDiscriminator(nn.Module):
    """
    PatchGAN discriminator for mel spectrograms.
    
    Input:  [B, 1, 64, T]  mel spectrogram
    Output: [B, 1, H', W']  real/fake per patch (each patch ≈ 32×32 mel bins)
    
    Architecture: 5-layer conv net → 2M params.
    """
    def __init__(self, base_channels: int = 64):
        super().__init__()
        ch = base_channels
        self.layers = nn.Sequential(
            # Layer 1: [B, 1, 64, T] → [B, 64, 32, T/2]
            nn.Conv2d(1, ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: [B, 64, 32, T/2] → [B, 128, 16, T/4]
            nn.Conv2d(ch, ch*2, 4, 2, 1),
            nn.BatchNorm2d(ch*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: [B, 128, 16, T/4] → [B, 256, 8, T/8]
            nn.Conv2d(ch*2, ch*4, 4, 2, 1),
            nn.BatchNorm2d(ch*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: [B, 256, 8, T/8] → [B, 512, 4, T/16]
            nn.Conv2d(ch*4, ch*8, 4, 2, 1),
            nn.BatchNorm2d(ch*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 5: [B, 512, 4, T/16] → [B, 1, 2, T/32+]
            nn.Conv2d(ch*8, 1, (4, 4), (2, 2), (1, 1)),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)  # [B, 1, H', W']


def gan_loss_disc(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
    """Discriminator loss: push real→1, fake→0."""
    real_loss = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
    fake_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
    return (real_loss + fake_loss) * 0.5


def gan_loss_gen(fake_pred: torch.Tensor) -> torch.Tensor:
    """Generator loss: push fake→1 (fool discriminator)."""
    return F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))


# ═══════════════════════════════════════════════════════════════
#  TRAIN / VALIDATE
# ═══════════════════════════════════════════════════════════════

def get_loss_fn():
    """Return loss function based on config."""
    if cfg.loss_type == "l1":
        return F.l1_loss
    elif cfg.loss_type == "huber":
        return lambda pred, tgt: F.smooth_l1_loss(pred, tgt, beta=0.1)
    return F.mse_loss


def train_epoch(model, diffusion, train_loader, optimizer, ema_model=None, discriminator=None, disc_optimizer=None):
    """Train one epoch. Returns average loss. Optional GAN discriminator training."""
    loss_fn = get_loss_fn()
    uncond_prob = getattr(cfg, 'uncond_prob', 0.0)
    use_gan = discriminator is not None and disc_optimizer is not None
    gan_weight = getattr(cfg, 'gan_weight', 0.1)
    model.train()
    if discriminator is not None:
        discriminator.train()
    total_loss = 0.0
    total_l1 = 0.0
    total_gan = 0.0
    pbar = tqdm(train_loader, desc="  Train", leave=False)
    optimizer.zero_grad()
    if disc_optimizer is not None:
        disc_optimizer.zero_grad()
    accum_count = 0

    for mel, labels in pbar:
        mel = mel.to(device)         # [B, 1, 64, T]
        labels = labels.to(device)   # [B]
        B = mel.shape[0]

        # SpecAugment: mask freq/time bands (prevents memorization)
        if cfg.dropout > 0:
            mel = spec_augment(mel)

        # Unconditional training: randomly drop labels for CFG
        if uncond_prob > 0 and np.random.random() < uncond_prob:
            labels = torch.full_like(labels, cfg.num_classes)

        # Sample random timesteps
        t = torch.randint(0, diffusion.timesteps, (B,), device=device)

        # Add noise: x_t = sqrt(α_cumprod) * x_0 + sqrt(1-α_cumprod) * ε
        noise = torch.randn_like(mel)
        x_t = diffusion.q_sample(mel, t, noise)

        # V7: x₀-prediction (model predicts clean mel, not noise)
        if getattr(cfg, 'predict_x0', False):
            pred = model(x_t, t, labels)
            l1_loss = freq_weighted_loss(pred, mel)

            # V8 Fix C: GAN loss
            if use_gan:
                # Discriminator update: real → 1, fake → 0
                disc_optimizer.zero_grad()
                real_score = discriminator(mel.detach())
                fake_score = discriminator(pred.detach())
                d_loss = gan_loss_disc(real_score, fake_score)
                d_loss.backward()
                disc_optimizer.step()

                # Generator GAN loss: fake → 1 (fool discriminator)
                g_gan = gan_weight * gan_loss_gen(discriminator(pred))
                loss = (l1_loss + g_gan) / GRADIENT_ACCUMULATION_STEPS
                total_gan += g_gan.item()
            else:
                loss = l1_loss / GRADIENT_ACCUMULATION_STEPS

            total_l1 += l1_loss.item()
        else:
            pred = model(x_t, t, labels)
            loss = loss_fn(pred, noise) / GRADIENT_ACCUMULATION_STEPS

        loss.backward()
        accum_count += 1
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

        # Step only after accumulation
        if accum_count >= GRADIENT_ACCUMULATION_STEPS:
            # Relaxed gradient clipping for diffusion models
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0

            # Update EMA if enabled
            if ema_model is not None:
                with torch.no_grad():
                    for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.data.mul_(EMA_DECAY).add_(model_p.data, alpha=1 - EMA_DECAY)

        pbar.set_postfix({"loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}"})

    # Handle leftover accumulation at end of epoch
    if accum_count > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad()

        if ema_model is not None:
            with torch.no_grad():
                for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(EMA_DECAY).add_(model_p.data, alpha=1 - EMA_DECAY)

    avg_loss = total_loss / len(train_loader)
    if use_gan:
        return avg_loss, total_l1 / len(train_loader), total_gan / len(train_loader)
    return avg_loss


def validate(model, diffusion, val_loader, ema_model=None):
    """Validate — returns average loss using configured loss_fn. Uses EMA weights if provided."""
    loss_fn = get_loss_fn()
    eval_model = ema_model if ema_model is not None else model
    eval_model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for mel, labels in val_loader:
            mel = mel.to(device)
            labels = labels.to(device)
            B = mel.shape[0]

            t = torch.randint(0, diffusion.timesteps, (B,), device=device)
            noise = torch.randn_like(mel)
            x_t = diffusion.q_sample(mel, t, noise)
            pred = eval_model(x_t, t, labels)
            if getattr(cfg, 'predict_x0', False):
                loss = freq_weighted_loss(pred, mel).item()
            else:
                loss = loss_fn(pred, noise).item()
            total_loss += loss

    return total_loss / max(len(val_loader), 1)


# ═══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def training_loop():
    """Full diffusion training with test/train mode, checkpoint resume."""

    # ═════════════════════════════════════════════════════════
    #  BANNER
    # ═════════════════════════════════════════════════════════
    model = SpectrogramUNet(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n🔧 Diffusion Refinement → {MODE.upper()} MODE")
    if is_cuda:
        print(f"   GPU:    {torch.cuda.get_device_name(0)}")
    eff_batch = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    print(f"   Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | Accum: {GRADIENT_ACCUMULATION_STEPS} | Effective: {eff_batch} | Workers: {NUM_WORKERS}")
    print(f"   Timesteps: {cfg.timesteps} | Model: {n_params:,} ({n_params/1e6:.1f}M) | Loss: {cfg.loss_type}")
    pred_mode = "x₀" if getattr(cfg, 'predict_x0', False) else "ε"
    print(f"   Prediction: {pred_mode} | Augment: {'yes' if getattr(cfg,'augment',False) else 'no'} | CFG: uncond_prob={cfg.uncond_prob}, scale={cfg.cfg_scale}")
    print(f"   Segment: {SEGMENT_FRAMES} mel frames | EMA: {EMA_DECAY} | AdamW weight_decay: {cfg.adam_weight_decay}")
    print(f"   Scheduler: CosineAnnealingWarmRestarts | Mixed precision: {'yes' if use_amp else 'no'}")
    print(f"   VAE mix-in ratio: {CONFIG.get('vae_mix_ratio', 0)*100:.0f}% | VAE ckpt: {CONFIG.get('vae_checkpoint', 'N/A')}")
    print(f"   Best model → {BEST_MODEL_PATH}")

    # ═════════════════════════════════════════════════════════
    #  DATA
    # ═════════════════════════════════════════════════════════

    # Load VAE for mix-in training (teaches diffusion to handle VAE-quality data)
    vae_mix_ratio = CONFIG.get("vae_mix_ratio", 0.0)
    vae_model_for_dataset = None
    if vae_mix_ratio > 0:
        vae_ckpt_path = CONFIG.get("vae_checkpoint", "models/best_vae_finetune_train.pth")
        if os.path.exists(vae_ckpt_path):
            from src.vae import SimpleAudioVAE
            vae_device_for_data = torch.device("cpu")  # VAE runs on CPU for dataset
            vae_model_for_dataset = SimpleAudioVAE(latent_dim=1024, num_classes=8, embed_dim=64)
            vae_model_for_dataset.load_state_dict(
                torch.load(vae_ckpt_path, map_location=vae_device_for_data, weights_only=True)["model_state_dict"]
            )
            vae_model_for_dataset.eval()
            for p in vae_model_for_dataset.parameters():
                p.requires_grad_(False)
            print(f"✅ VAE loaded for mix-in from: {vae_ckpt_path}")
        else:
            print(f"⚠️  VAE checkpoint not found at {vae_ckpt_path} — disabling VAE mix-in")
            vae_mix_ratio = 0.0

    # V9: class-balanced or standard dataset
    if getattr(cfg, 'balance_classes', False):
        balanced = getattr(cfg, 'noise_max_samples', 0) > 0
        print(f"   ⚖️  Class-balanced sampling: {'Noise capped at '+str(cfg.noise_max_samples) if balanced else 'all classes equal'}")
        train_ds = BalancedDiffusionDataset(CONFIG["data_dir"], SEGMENT_FRAMES, split="train")
        val_ds = BalancedDiffusionDataset(CONFIG["data_dir"], SEGMENT_FRAMES, split="val")
        
        # Show class distribution
        class_counts = {c: len(train_ds.class_samples[c]) for c in range(cfg.num_classes)}
        class_names = {i: n for n, i in CLASS_TO_IDX.items()}
        dist_str = ' | '.join(f"{class_names.get(c,'?'+str(c))}:{n}" for c, n in sorted(class_counts.items()))
        print(f"   Class distribution: {dist_str}")
        print(f"   Epoch size: {len(train_ds)} samples ({train_ds.max_per_class} per class)")
    else:
        train_ds = DiffusionDataset(
            CONFIG["data_dir"], SEGMENT_FRAMES, split="train",
            vae_model=vae_model_for_dataset, vae_mix_ratio=vae_mix_ratio,
        )
        val_ds = DiffusionDataset(
            CONFIG["data_dir"], SEGMENT_FRAMES, split="val",
            vae_model=vae_model_for_dataset, vae_mix_ratio=vae_mix_ratio,
        )

    print(f"\n✅ Data loaded: {len(train_ds)} train / {len(val_ds)} val | VAE mix-in: {vae_mix_ratio*100:.0f}%")

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
    _check_mel, _check_labels = next(iter(train_loader))
    print(f"   📊 First batch: mel shape={tuple(_check_mel.shape)}, "
          f"min={_check_mel.min():.4f}, max={_check_mel.max():.4f}, "
          f"mean_abs={_check_mel.abs().mean():.4f}")
    if _check_mel.abs().max() < 1e-6:
        print("   🚨 ALL ZEROS — mel data not loading correctly.")
        print("   🚨 Training stopped to save time.")
        return None
    print(f"   Labels: {_check_labels.tolist()} → {[CLASSES[l] for l in _check_labels.tolist()]}")

    # ═════════════════════════════════════════════════════════
    #  MODELS
    # ═════════════════════════════════════════════════════════
    model = model.to(device)
    diffusion = DiffusionProcess(cfg).to(device)

    # ═════════════════════════════════════════════════════════
    #  OPTIMIZER
    # ═════════════════════════════════════════════════════════
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, betas=cfg.adam_betas, weight_decay=cfg.adam_weight_decay
    )
    # Cosine annealing with warm restarts — better for diffusion than exponential decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=NUM_EPOCHS // 4, T_mult=2, eta_min=1e-6
    )

    # EMA model (shadow copy of weights for smoother inference)
    ema_model = SpectrogramUNet(cfg).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)

    # ═════════════════════════════════════════════════════════
    #  V8 FIX C: DISCRIMINATOR (PatchGAN)
    # ═════════════════════════════════════════════════════════
    discriminator = None
    disc_optimizer = None
    if getattr(cfg, 'gan_weight', 0) > 0:
        discriminator = MelDiscriminator(base_channels=cfg.disc_channels).to(device)
        disc_optimizer = torch.optim.AdamW(
            discriminator.parameters(), lr=cfg.disc_lr, betas=cfg.adam_betas, weight_decay=0.0
        )
        n_disc = sum(p.numel() for p in discriminator.parameters())
        print(f"   🎭 PatchGAN discriminator: {n_disc:,} ({n_disc/1e6:.1f}M) — λ={cfg.gan_weight}")

    # ═════════════════════════════════════════════════════════
    #  TRAIN (no periodic checkpoints — only best model saved)
    # ═════════════════════════════════════════════════════════
    os.makedirs(cfg.model_dir, exist_ok=True)

    # ═════════════════════════════════════════════════════════
    #  TRAIN
    # ═════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"🚀 DIFFUSION TRAINING v9 — {MODE.upper()} MODE")
    print(f"   Prediction: x₀ + freq-weighted + GAN + class-balanced + 100 epochs")
    print(f"   Device: {device} | Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} (eff: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"   Saving last model → {BEST_MODEL_PATH}")
    print(f"{'='*60}\n")

    best_val = float("inf")
    start_epoch = 0
    BEST_PATH = os.path.join(cfg.model_dir, f"diffusion_unet_{MODE}_best.pth")

    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            t0 = time.time()

            result = train_epoch(model, diffusion, train_loader, optimizer, ema_model,
                                discriminator, disc_optimizer)
            scheduler.step()
            lr = scheduler.get_last_lr()[0]

            val_loss = validate(model, diffusion, val_loader, ema_model)

            dt = time.time() - t0
            trend = "📉" if val_loss < best_val else "➡️"

            # Unpack loss components
            if isinstance(result, tuple):
                avg_loss, l1_loss, gan_loss = result
                print(f"── Epoch {epoch+1:3d}/{NUM_EPOCHS} "
                      f"({dt:.0f}s) ── "
                      f"loss={avg_loss:.4f} (L1={l1_loss:.3f}, G={gan_loss:.4f}) val={val_loss:.4f} {trend} lr={lr:.2e}")
            else:
                avg_loss = result
                print(f"── Epoch {epoch+1:3d}/{NUM_EPOCHS} "
                      f"({dt:.0f}s) ── "
                      f"loss={avg_loss:.4f} val={val_loss:.4f} {trend} lr={lr:.2e}")

            # Save best (using EMA weights)
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"unet": ema_model.state_dict(), "config": cfg.__dict__}, BEST_PATH)
                print(f"      💾 Best model saved (val={best_val:.4f})")
    except KeyboardInterrupt:
        print(f"\n⏸️  Interrupted at epoch {epoch+1}. Checkpoint saved — resume anytime.")

    # Save final model (EMA weights)
    torch.save({"unet": ema_model.state_dict(), "config": cfg.__dict__}, BEST_MODEL_PATH)
    print(f"\n💾 Final model saved to: {BEST_MODEL_PATH}")
    print(f"   Best val loss: {best_val:.4f}")
    print("✅ Training complete!")

    return model


if __name__ == "__main__":
    trained_model = training_loop()
