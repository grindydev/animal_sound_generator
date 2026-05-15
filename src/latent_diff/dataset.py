"""
dataset.py — Dataset that encodes real mels to spatial latents.

For decoder training: returns (features, mel) pairs.
For diffusion training: returns (latent, label) pairs.
"""
import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.latent_diff.config import config as cfg


def load_audio(path: str):
    """Load audio file. Tries torchaudio first, falls back to soundfile."""
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)
        return wav, sr
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
        return None, None


# Cache mel transforms per device
_mel_cache = {}


def compute_mel(audio: torch.Tensor) -> torch.Tensor:
    """Compute normalized mel matching existing pipeline."""
    from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

    norm_mean = -18.4903
    norm_std = 19.8031
    d = audio.device

    if d not in _mel_cache:
        _mel_cache[d] = (
            MelSpectrogram(
                sample_rate=cfg.sample_rate,
                n_fft=1024, hop_length=cfg.hop_length,
                n_mels=cfg.n_mels, f_min=0.0, f_max=cfg.sample_rate / 2,
            ).to(d),
            AmplitudeToDB(top_db=80.0).to(d),
        )

    mel_tfm, db_tfm = _mel_cache[d]
    mel = mel_tfm(audio)           # [1, n_mels, T]
    mel = db_tfm(mel)              # dB
    mel = (mel - norm_mean) / norm_std  # normalize
    mel = mel.unsqueeze(1)          # [1, 1, n_mels, T]
    return mel


CLASSES = ['Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen', 'Noise']


class LatentDataset(Dataset):
    """
    Dataset that provides:
      - features: encoder spatial bottleneck [512, 4, 35] (for decoder training)
      - latent: reduced latent [16, 4, 35] (for diffusion training)
      - mel: target spectrogram [1, 64, segment_frames] (for decoder loss)
      - label: class index
    """

    def __init__(self, data_dir: str, segment_frames: int, split: str = "train",
                 encoder=None, reducer=None, device=None):
        self.data_dir = data_dir
        self.segment_frames = segment_frames
        self.encoder = encoder
        self.reducer = reducer
        self.device = device

        # Collect samples
        from src.diffusion.train import smart_crop

        self.samples = []
        split_path = os.path.join(data_dir, f"{split}.txt")
        if os.path.exists(split_path):
            with open(split_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            for line in lines:
                path, label = line.rsplit(' ', 1)
                self.samples.append((os.path.join(data_dir, path), int(label)))
        else:
            for label_idx, cls in enumerate(CLASSES):
                cls_dir = os.path.join(data_dir, cls)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in sorted(os.listdir(cls_dir)):
                    if fname.endswith(('.wav', '.mp3', '.flac')):
                        self.samples.append((os.path.join(cls_dir, fname), label_idx))

        print(f"   {split}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        wav, sr = load_audio(path)

        if wav is None:
            return torch.zeros(1, 64, self.segment_frames), torch.tensor(0, dtype=torch.long)

        # Resample
        if sr != cfg.sample_rate:
            import torchaudio
            wav = torchaudio.transforms.Resample(sr, cfg.sample_rate)(wav)

        # Mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Crop/pad to segment
        crop_samples = self.segment_frames * cfg.hop_length
        if wav.shape[-1] <= crop_samples:
            wav = torch.nn.functional.pad(wav, (0, crop_samples - wav.shape[-1]))
        else:
            from src.diffusion.train import smart_crop
            crops = smart_crop(wav, crop_samples=crop_samples, threshold_db=-30.0,
                               num_crops=1, merge_gap_samples=4410)
            wav = crops[0]

        # Compute mel
        mel = compute_mel(wav)  # [1, 1, 64, T]
        T = mel.shape[-1]
        if T > self.segment_frames:
            mel = mel[..., :self.segment_frames]
        elif T < self.segment_frames:
            mel = torch.nn.functional.pad(mel, (0, self.segment_frames - T))

        # Squeeze channel for decoder training (decoder expects 1ch input to encoder)
        mel_flat = mel.squeeze(1)  # [1, 64, T] for encoder input

        return mel_flat, torch.tensor(label, dtype=torch.long)


def encode_batch(encoder, reducer, mel_batch, device):
    """Encode a batch of mels to latent [B, 16, 4, 35]."""
    with torch.no_grad():
        encoder.eval()
        z, _ = encoder.encode(mel_batch.to(device))  # z is [B, 2048] from fc_encode
        # For spatial features we need the intermediate output.
        # The encoder doesn't expose spatial features directly.
        # We'll encode manually to get the spatial bottleneck.
    return None  # Handled in training loop directly
