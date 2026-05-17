"""
generate.py — V15 Latent Diffusion Generation (Griffin-Lim)

Noise → Latent UNet (DDIM) → latent [16, 4, 35]
→ ChannelExpander → spatial [256, 4, 35]
→ LatentDecoder → mel [1, 64, 552]
→ Griffin-Lim → audio

Usage:
    python src/latent_diff/generate.py --label Dog
    python src/latent_diff/generate.py --label Dog --steps 100 --cfg-scale 2.0
"""
import os
import sys
import argparse
import torch
import torchaudio
import soundfile as sf
from datetime import datetime
from torchaudio.transforms import InverseMelScale, GriffinLim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.latent_diff.config import config as cfg
from src.latent_diff.decoder import LatentDecoder, ChannelExpander
from src.latent_diff.unet import LatentUNet
from src.vae.autoencoder import ImprovedAutoencoder
from src.diffusion.diffusion import DiffusionProcess

CLASSES = ['Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen']
_output_dir = "outputs/generated"
os.makedirs(_output_dir, exist_ok=True)


def load_models(device):
    """Load all components for generation."""
    # 1. Encoder (for bottleneck_ch detection)
    encoder_ckpt = cfg.encoder_ckpt
    if not os.path.exists(encoder_ckpt):
        encoder_ckpt = os.path.join(cfg.model_dir, "best_autoencoder_train.pth")
    ckpt = torch.load(encoder_ckpt, map_location='cpu', weights_only=True)
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
    encoder = ImprovedAutoencoder(latent_dim=2048, base_channels=base_ch).to(device)
    encoder.load_state_dict(state_dict)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    bottleneck_ch = encoder.c4
    print(f"✅ Encoder loaded (bottleneck={bottleneck_ch}ch)")

    # 2. Decoder + expander
    decoder_path = cfg.decoder_ckpt
    if not os.path.exists(decoder_path):
        decoder_path = os.path.join(cfg.model_dir, "latent_decoder_best.pth")
    if not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Decoder not found: {decoder_path}. Train Phase 1 first.")

    dec_ckpt = torch.load(decoder_path, map_location=device, weights_only=True)
    loaded_ch = dec_ckpt.get('bottleneck_ch', bottleneck_ch)
    expander = ChannelExpander(in_ch=cfg.latent_channels, out_ch=loaded_ch).to(device)
    expander.load_state_dict(dec_ckpt['expander'])
    expander.eval()
    decoder = LatentDecoder(bottleneck_ch=loaded_ch, config=cfg).to(device)
    decoder.load_state_dict(dec_ckpt['decoder'])
    decoder.eval()
    print(f"✅ Decoder loaded (val_loss={dec_ckpt.get('val_loss', '?'):.4f})")

    # 3. Diffusion UNet
    diff_path = cfg.diffusion_ckpt
    if not os.path.exists(diff_path):
        diff_path = os.path.join(cfg.model_dir, "latent_diffusion_best.pth")
    if not os.path.exists(diff_path):
        raise FileNotFoundError(f"Diffusion UNet not found: {diff_path}. Train Phase 2 first.")

    diff_ckpt = torch.load(diff_path, map_location=device, weights_only=True)
    unet = LatentUNet(cfg).to(device)
    if 'unet' in diff_ckpt:
        unet.load_state_dict(diff_ckpt['unet'])
    else:
        unet.load_state_dict(diff_ckpt)
    unet.eval()
    print(f"✅ Diffusion UNet loaded (val_loss={diff_ckpt.get('val_loss', '?'):.4f})")

    # 4. Diffusion process
    diffusion = DiffusionProcess(cfg).to(device)
    diffusion.num_classes = cfg.num_classes

    return encoder, expander, decoder, unet, diffusion, bottleneck_ch


@torch.no_grad()
def generate_one(label: str, num_steps=100, cfg_scale=2.0, device=None):
    """Generate one animal sound from noise."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")

    encoder, expander, decoder, unet, diffusion, _ = load_models(device)
    label_idx = CLASSES.index(label)

    # DDIM sample: noise → latent
    noise = torch.randn(1, cfg.latent_channels, cfg.latent_height, cfg.latent_width, device=device)
    labels = torch.tensor([label_idx], device=device, dtype=torch.long)
    latent = diffusion.ddim_sample(unet, noise, labels, num_steps=num_steps, eta=0.0, cfg_scale=cfg_scale)

    # Decode: latent → mel
    features = expander(latent)
    mel = decoder(features, target_size=(64, cfg.segment_frames))  # [1, 1, 64, 552]

    # Griffin-Lim: mel → audio (no HiFi-GAN electric noise)
    inv_mel = InverseMelScale(n_stft=513, n_mels=64, sample_rate=22050, f_min=0, f_max=11025)
    gl = GriffinLim(n_fft=1024, hop_length=200, win_length=1024, n_iter=64, power=1)

    mel_db = mel.squeeze() * 19.8031 - 18.4903
    mel_power = 10 ** (mel_db.clamp(-80, 0) / 10.0)
    lin_spec = inv_mel(mel_power.cpu())
    audio = gl(torch.sqrt(lin_spec.clamp(min=0)))

    # Normalize
    peak = audio.abs().max()
    if peak > 0:
        audio = audio / peak * 0.95

    return audio.squeeze(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V15 Latent Diffusion Generation")
    parser.add_argument("--label", type=str, default=None, help=f"Class: {', '.join(CLASSES)}")
    parser.add_argument("--steps", type=int, default=100, help="DDIM steps")
    parser.add_argument("--cfg-scale", type=float, default=2.0, help="CFG scale")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"🚀 Device: {device}")
    labels = [args.label] if args.label else CLASSES
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for label in labels:
        for i in range(args.count):
            print(f"🎵 {label} ({i+1}/{args.count})...", end=" ", flush=True)
            audio = generate_one(label, args.steps, args.cfg_scale, device)
            fname = f"latent_{label}_v15_{ts}_{i+1}.wav"
            path = os.path.join(_output_dir, fname)
            sf.write(path, audio.cpu().numpy(), 22050)
            rms = (audio**2).mean().sqrt()
            print(f"✅ RMS={rms:.3f} → {path}")

    print(f"\n🎉 Done! {_output_dir}/")
