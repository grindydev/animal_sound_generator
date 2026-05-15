"""
generate.py — Full Latent Diffusion Generation Pipeline.

Phase 2 inference:
  Noise → Latent UNet (DDPM/DDIM) → latent [16, 4, 35]
  → ChannelExpander → spatial features [256, 4, 35]
  → LatentDecoder → mel [1, 64, 552]
  → HiFi-GAN → audio .wav

Usage:
    python src/latent_diff/generate.py --label Dog --ddpm
    python src/latent_diff/generate.py --label Cat --steps 100 --cfg-scale 2.0
"""
import os
import sys
import argparse
import torch
import soundfile as sf
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.latent_diff.config import config as cfg
from src.latent_diff.decoder import LatentDecoder, ChannelExpander
from src.latent_diff.unet import LatentUNet
from src.vae.autoencoder import ImprovedAutoencoder
from src.diffusion.diffusion import DiffusionProcess

# ═══════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════

CLASSES = ['Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen', 'Noise']
_output_dir = "outputs/generated"
os.makedirs(_output_dir, exist_ok=True)


def load_models(device):
    """Load all components for generation."""
    # 1. Load encoder (for bottleneck_ch detection)
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
    encoder = ImprovedAutoencoder(latent_dim=2048, base_channels=base_ch)
    encoder.load_state_dict(state_dict)
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    bottleneck_ch = encoder.c4
    print(f"✅ Encoder loaded (bottleneck={bottleneck_ch}ch)")

    # 2. Load decoder + expander
    decoder_path = cfg.decoder_ckpt
    if not os.path.exists(decoder_path):
        decoder_path = os.path.join(cfg.model_dir, "latent_decoder_best.pth")

    if os.path.exists(decoder_path):
        dec_ckpt = torch.load(decoder_path, map_location=device, weights_only=True)
        loaded_ch = dec_ckpt.get('bottleneck_ch', bottleneck_ch)
        expander = ChannelExpander(in_ch=cfg.latent_channels, out_ch=loaded_ch).to(device)
        expander.load_state_dict(dec_ckpt['expander'])
        expander.eval()

        decoder = LatentDecoder(bottleneck_ch=loaded_ch, config=cfg).to(device)
        decoder.load_state_dict(dec_ckpt['decoder'])
        decoder.eval()
        print(f"✅ Decoder loaded (val_loss={dec_ckpt.get('val_loss', '?'):.4f})")
    else:
        raise FileNotFoundError(f"Decoder not found: {decoder_path}")

    # 3. Load diffusion UNet
    diff_path = cfg.diffusion_ckpt
    if not os.path.exists(diff_path):
        diff_path = os.path.join(cfg.model_dir, "latent_diffusion_best.pth")

    if os.path.exists(diff_path):
        # LatentDiffConfig is a custom class — need weights_only=False or add_safe_globals
        diff_ckpt = torch.load(diff_path, map_location=device, weights_only=False)
        unet = LatentUNet(cfg).to(device)
        if 'unet' in diff_ckpt:
            unet.load_state_dict(diff_ckpt['unet'])
        else:
            unet.load_state_dict(diff_ckpt)
        unet.eval()
        print(f"✅ Diffusion UNet loaded (val_loss={diff_ckpt.get('val_loss', '?'):.4f})")
    else:
        raise FileNotFoundError(f"Diffusion UNet not found: {diff_path}")

    # 4. Diffusion process
    diffusion = DiffusionProcess(cfg).to(device)
    diffusion.num_classes = cfg.num_classes

    return encoder, expander, decoder, unet, diffusion, bottleneck_ch


@torch.no_grad()
def generate_from_label(
    label: str,
    num_steps: int = 100,
    use_ddpm: bool = False,
    cfg_scale: float = 2.0,
    device: torch.device = None,
) -> torch.Tensor:
    """Full generation: noise → latent → mel → audio."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")

    encoder, expander, decoder, unet, diffusion, bottleneck_ch = load_models(device)

    label_idx = CLASSES.index(label)
    B = 1

    # Step 1: Generate latent via diffusion
    noise = torch.randn(B, cfg.latent_channels, cfg.latent_height, cfg.latent_width, device=device)
    labels_t = torch.full((B,), label_idx, device=device, dtype=torch.long)

    if use_ddpm:
        latent = diffusion.p_sample_loop(
            unet, (B, cfg.latent_channels, cfg.latent_height, cfg.latent_width),
            labels_t, device, progress=True, cfg_scale=cfg_scale
        )
    else:
        latent = diffusion.ddim_sample(
            unet, noise, labels_t, num_steps=num_steps, eta=0.0, cfg_scale=cfg_scale
        )

    # Step 2: Expand → Decode → mel
    features = expander(latent)  # [1, bottleneck_ch, 4, 35]
    mel = decoder(features, target_size=(64, cfg.segment_frames))  # [1, 1, 64, 552]

    # Step 3: HiFi-GAN → audio
    from src.hifigan.inference import mel_to_waveform
    audio = mel_to_waveform(mel, device=device, use_griffin_lim=False)

    return audio, mel, latent


def generate_and_save(label: str, num_steps: int = 100, use_ddpm: bool = False,
                      cfg_scale: float = 2.0, device: torch.device = None):
    """Generate one sample and save to outputs/generated/."""
    audio, mel, latent = generate_from_label(label, num_steps, use_ddpm, cfg_scale, device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method = "ddpm" if use_ddpm else "ddim"
    fname = f"latent_{label}_{method}_cfg{cfg_scale}_{timestamp}.wav"
    path = os.path.join(_output_dir, fname)

    wav = audio.squeeze().cpu().numpy()
    sf.write(path, wav, cfg.sample_rate)
    print(f"💾 Saved: {path}")
    print(f"   Mel σ: {mel.std():.2f} | Audio RMS: {wav.std():.3f}")
    return path


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent Diffusion Generation")
    parser.add_argument("--label", type=str, default="Dog", choices=CLASSES)
    parser.add_argument("--steps", type=int, default=100, help="DDIM sampling steps")
    parser.add_argument("--ddpm", action="store_true", help="Use full DDPM sampling")
    parser.add_argument("--cfg-scale", type=float, default=2.0, help="CFG scale")
    parser.add_argument("--count", type=int, default=1, help="Number of samples")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"🚀 Device: {device}")
    print(f"🎵 Generating {args.label} ({'DDPM' if args.ddpm else 'DDIM'}, {args.steps} steps, CFG={args.cfg_scale})")

    for i in range(args.count):
        if args.count > 1:
            print(f"\n── Sample {i+1}/{args.count} ──")
        generate_and_save(args.label, args.steps, args.ddpm, args.cfg_scale, device)
