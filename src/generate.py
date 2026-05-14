"""
generate.py — Full Pipeline: VAE → Diffusion → HiFi-GAN → Audio
=================================================================

Generate animal sounds using all three trained models.

Usage:
    python src/generate.py                          # generate all 8 animals
    python src/generate.py --label Dog              # generate one specific animal
    python src/generate.py --label Cat --count 5    # generate 5 cat sounds
    python src/generate.py --label Dog --no-diff    # skip diffusion (VAE only)
    python src/generate.py --label Dog --strength 0.8  # stronger refinement
"""
import os
import sys
import argparse
import torch
import soundfile as sf
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from vae import ImprovedVAE
from diffusion.inference import refine_spectrogram, generate_from_noise
from hifigan.inference import mel_to_waveform


CLASSES = ['Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen', 'Noise']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def load_vae(device):
    """Load the finetuned VAE model. Auto-detects base_channels."""
    ckpt_path = "models/best_vae_finetune_train.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt["model_state_dict"]
    base_ch = state["enc1.main.0.weight"].shape[0]
    embed_dim = state["class_embed.weight"].shape[1]  # auto-detect
    model = ImprovedVAE(latent_dim=2048, num_classes=8, embed_dim=embed_dim, base_channels=base_ch).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"✅ VAE loaded (base_ch={base_ch}, embed_dim={embed_dim})")
    return model


def generate_one(
    vae, label, device,
    temperature=0.7,
    use_diffusion=True,
    strength=0.6,
    diffusion_steps=50,
    use_griffin_lim=False,
    griffin_lim_iters=5,
    from_scratch=False,
):
    """Generate one animal sound through the full pipeline."""
    label_idx = CLASS_TO_IDX[label]

    if from_scratch:
        # Path B: pure diffusion generation from noise (no VAE)
        vae_mel = generate_from_noise(
            label_idx=label_idx, num_samples=1,
            num_steps=diffusion_steps, device=device,
        )
    else:
        # Path A: VAE generation + optional diffusion refinement
        # Step 1: VAE generates rough spectrogram
        vae_mel = vae.sample(label_idx, num_samples=1, device=device, temperature=temperature)

        # Step 1b: Normalize VAE output
        vae_mel = torch.clamp(vae_mel, -4.0, 4.0)
        mel_mean = vae_mel.mean()
        mel_std = vae_mel.std()
        if mel_std > 0.01:
            vae_mel = (vae_mel - mel_mean) / mel_std * 0.7

        # Step 2: Diffusion refinement (optional)
        if use_diffusion:
            vae_mel = refine_spectrogram(
                vae_mel, label_idx=label_idx,
                strength=strength, num_steps=diffusion_steps,
                device=device, use_ddim=True,
            )

    # Step 3: HiFi-GAN converts mel → waveform
    waveform = mel_to_waveform(
        vae_mel, device=device,
        use_griffin_lim=use_griffin_lim,
        griffin_lim_iters=griffin_lim_iters,
    )

    return waveform  # [1, samples]


def main():
    parser = argparse.ArgumentParser(description="Generate animal sounds")
    parser.add_argument("--label", type=str, default=None,
                        help=f"Animal class: {', '.join(CLASSES)} (default: all)")
    parser.add_argument("--count", type=int, default=1,
                        help="Number of sounds per class")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="VAE temperature (0.3=consistent, 0.5=normal, 1.0=wild)")
    parser.add_argument("--strength", type=float, default=0.6,
                        help="Diffusion refinement strength (0.0=none, 1.0=full)")
    parser.add_argument("--diffusion-steps", type=int, default=50,
                        help="DDIM sampling steps (fewer=faster, more=sharper)")
    parser.add_argument("--no-diff", action="store_true",
                        help="Skip diffusion (VAE only)")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Pure diffusion generation from noise (no VAE needed)")
    parser.add_argument("--griffin-lim", action="store_true",
                        help="Enable Griffin-Lim phase refinement (off by default)")
    parser.add_argument("--output-dir", type=str, default="outputs/generated",
                        help="Directory to save audio files")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, mps, cpu")
    args = parser.parse_args()

    # ── Device ──────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"🚀 Device: {device}")

    # ── Output dir ──────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Load models ─────────────────────────────────────
    if args.from_scratch:
        vae = None  # no VAE needed for pure diffusion generation
        print("🎲 Pure diffusion generation from noise (no VAE)")
    else:
        vae = load_vae(device)

    # ── Generate ────────────────────────────────────────
    labels = [args.label] if args.label else CLASSES

    for label in labels:
        for i in range(args.count):
            tag = 'scratch' if args.from_scratch else ('diff' if not args.no_diff else 'vae')
            fname = f"{label}_{tag}_s{args.strength}_t{args.temperature}_{timestamp}_{i+1}.wav"
            out_path = os.path.join(args.output_dir, fname)

            print(f"🎵 Generating {label} ({i+1}/{args.count})...", end=" ", flush=True)

            waveform = generate_one(
                vae, label, device,
                temperature=args.temperature,
                use_diffusion=not args.no_diff,
                strength=args.strength,
                diffusion_steps=args.diffusion_steps,
                use_griffin_lim=args.griffin_lim,
                from_scratch=args.from_scratch,
            )

            # Normalize to [-1, 1]
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak * 0.95

            sf.write(out_path, waveform.squeeze().cpu().numpy(), 22050)
            dur = waveform.shape[-1] / 22050
            print(f"✅ {dur:.1f}s → {out_path}")

    print(f"\n🎉 Done! Files saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
