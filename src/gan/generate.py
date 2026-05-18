"""
GAN Generation — Generate animal sounds from trained GAN

Usage:
    python -m src.gan.generate                    # generate samples for all classes
    python -m src.gan.generate --label Dog         # generate for specific class
    python -m src.gan.generate --label Dog --n 10  # generate 10
"""
import os, sys, argparse
import torch
import torchaudio
from torchaudio.transforms import InverseMelScale, GriffinLim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.gan.config import config as cfg
from src.gan.generator import Generator


def load_generator(device):
    """Load trained generator."""
    if not os.path.exists(cfg.generator_path):
        raise FileNotFoundError(
            f"No trained generator found at {cfg.generator_path}.\n"
            f"Train first: python src/gan/train.py"
        )
    G = Generator(cfg).to(device)
    state = torch.load(cfg.generator_path, map_location=device, weights_only=True)
    # Support both dict save and raw state_dict
    bin_mean = None
    bin_std = None
    if 'generator' in state:
        G.load_state_dict(state['generator'])
        val_acc = state.get('val_acc', None)
        bin_mean = state.get('bin_mean', None)
        bin_std = state.get('bin_std', None)
        if val_acc is not None:
            print(f"   val_acc at save: {val_acc*100:.1f}%")
    else:
        G.load_state_dict(state)
    G.eval()
    return G, bin_mean, bin_std


def mel_to_audio(mel, bin_mean=None, bin_std=None, n_iter=64):
    """Convert per-bin z-score mel to audio via Griffin-Lim.

    Args:
        mel: [B, 1, H, W] — per-bin z-score normalized
        bin_mean, bin_std: per-bin stats for denormalization
        n_iter: Griffin-Lim iterations
    """
    device = mel.device

    # Denormalize: z-score → dB
    if bin_mean is not None and bin_std is not None:
        bin_mean = bin_mean.to(device)
        bin_std = bin_std.to(device)
        mel_db = mel * bin_std + bin_mean  # [B, 1, H, W]
    else:
        mel_db = (mel - 1.0) * 40.0  # fallback: old [-1,1] mapping

    mel_db = mel_db.clamp(-80, 0)

    # Inverse mel scale
    inv_mel = InverseMelScale(
        n_stft=cfg.n_fft // 2 + 1,
        n_mels=cfg.n_mels,
        sample_rate=cfg.sample_rate,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
    ).to(device)
    spec = inv_mel(mel_power.squeeze(1))  # [B, n_fft//2+1, T]
    spec = spec.clamp(min=0)

    # Griffin-Lim
    gl = GriffinLim(
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.n_fft,
        n_iter=n_iter,
        power=1,
    ).to(device)
    audio = gl(torch.sqrt(spec).unsqueeze(1))  # [B, T]
    return audio.unsqueeze(1)  # [B, 1, T]


def generate():
    parser = argparse.ArgumentParser(description="Generate animal sounds with GAN v16")
    parser.add_argument('--label', type=str, default=None,
                        help=f'Animal class: {", ".join(cfg.CLASSES)} (default: all)')
    parser.add_argument('--n', type=int, default=5, help='Number of samples per class')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')

    G, bin_mean, bin_std = load_generator(device)
    g_params = sum(p.numel() for p in G.parameters())
    print(f"✅ Generator loaded: {g_params:,} params ({g_params/1e6:.1f}M)")
    os.makedirs(args.output_dir, exist_ok=True)

    classes_to_generate = [args.label] if args.label else cfg.CLASSES

    with torch.no_grad():
        for cls_name in classes_to_generate:
            if cls_name not in cfg.class_to_idx:
                print(f"❌ Unknown class: {cls_name}. Options: {', '.join(cfg.CLASSES)}")
                continue

            cls_idx = cfg.class_to_idx[cls_name]
            print(f"🎵 Generating {args.n} × {cls_name}...")

            for i in range(args.n):
                z = torch.randn(1, cfg.latent_dim, device=device)
                labels = torch.tensor([cls_idx], device=device, dtype=torch.long)
                mel = G(z, labels)
                audio = mel_to_audio(mel, bin_mean, bin_std)
                rms = (audio ** 2).mean().sqrt().item()

                out_path = os.path.join(args.output_dir, f"gan_v16_{cls_name}_{i+1:02d}.wav")
                torchaudio.save(out_path, audio.squeeze().cpu(), cfg.sample_rate)
                print(f"   ✅ {out_path} (RMS: {rms:.4f})")

    print(f"\n✨ Done! Listen: {args.output_dir}/")


if __name__ == '__main__':
    generate()
