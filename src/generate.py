"""
generate.py — V14: Animal Sound Generation

Two modes:
  1. v14_latent: VAE + Latent Diffusion (true generation from noise) — INDUSTRY STANDARD
  2. retrieval:   Real mel → perturb → Griffin-Lim (quick workaround)

Usage:
    # After Colab training:
    python src/generate.py --label Dog --v14-latent

    # Quick retrieval (no training needed):
    python src/generate.py --label Dog --retrieval
"""
import os, sys, argparse, torch, torchaudio, random
import soundfile as sf
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

CLASSES = ['Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

_mel_index_cache = {}


# ═══════════════════════════════════════════════════════════════
#  MODE 1: V14 LATENT DIFFUSION (industry standard)
# ═══════════════════════════════════════════════════════════════

def generate_v14_latent(label, device, num_samples=1, cfg_scale=2.0, num_steps=50):
    """True generation: noise → latent diffusion → VAE decode → Griffin-Lim → audio."""
    from src.v14_vae import V14VAE
    from src.v14_ldm import LatentDiffusionModel, LatentDiffusion
    from torchaudio.transforms import InverseMelScale, GriffinLim

    # Load models
    vae = V14VAE(latent_dim=256, num_classes=7, class_emb_dim=32).to(device)
    vae.load_state_dict(torch.load('models/v14_vae.pth', map_location=device, weights_only=True))
    vae.eval()

    ldm = LatentDiffusionModel(
        latent_dim=256, class_emb_dim=32, time_emb_dim=64,
        hidden_dim=512, num_blocks=4, num_classes=7,
    ).to(device)
    ldm.load_state_dict(torch.load('models/v14_ldm.pth', map_location=device, weights_only=True))
    ldm.eval()

    diffusion = LatentDiffusion(timesteps=1000).to(device)

    inv_mel = InverseMelScale(n_stft=513, n_mels=64, sample_rate=22050, f_min=0, f_max=11025)
    gl = GriffinLim(n_fft=1024, hop_length=200, win_length=1024, n_iter=64, power=1)

    label_idx = CLASS_TO_IDX[label]
    labels = torch.full((num_samples,), label_idx, device=device, dtype=torch.long)

    with torch.no_grad():
        z_0 = diffusion.ddim_sample(
            ldm, (num_samples, 256), labels,
            num_steps=num_steps, cfg_scale=cfg_scale, device=device,
        )
        mel = vae.decode(z_0, labels)
        mel_db = mel * 19.8031 - 18.4903
        mel_power = 10 ** (mel_db.clamp(-80, 0) / 10.0)
        # InverseMelScale + Griffin-Lim on CPU (MPS doesn't support linalg.lstsq)
        mel_cpu = mel_power.squeeze(1).cpu()
        lin_spec = inv_mel(mel_cpu)
        audio = gl(torch.sqrt(lin_spec.clamp(min=0)))

    # Normalize
    peak = audio.abs().max()
    if peak > 0:
        audio = audio / peak * 0.95
    return audio


# ═══════════════════════════════════════════════════════════════
#  MODE 2: RETRIEVAL (quick workaround, no training needed)
# ═══════════════════════════════════════════════════════════════

def load_mel_index(cls_name):
    if cls_name in _mel_index_cache:
        return _mel_index_cache[cls_name]
    path = f"data/mel_index/{cls_name}.pt"
    if not os.path.exists(path):
        return None
    mels = torch.load(path, weights_only=True)
    _mel_index_cache[cls_name] = mels
    return mels


def generate_retrieval(label, device, variation=0.3):
    from torchaudio.transforms import InverseMelScale, GriffinLim
    mels = load_mel_index(label)
    if mels is None or len(mels) == 0:
        return None

    if variation > 0.1 and len(mels) >= 2:
        idx1, idx2 = random.sample(range(len(mels)), 2)
        mel = (random.uniform(0.3, 0.7) * mels[idx1] +
               (1 - random.uniform(0.3, 0.7)) * mels[idx2])
    else:
        mel = mels[random.randint(0, len(mels) - 1)].clone()
    if variation > 0:
        mel = mel + torch.randn_like(mel) * variation * 0.1
    mel = torch.clamp(mel, -3.0, 3.0)

    inv_mel = InverseMelScale(n_stft=513, n_mels=64, sample_rate=22050, f_min=0, f_max=11025)
    gl = GriffinLim(n_fft=1024, hop_length=200, win_length=1024, n_iter=64, power=1)

    mel_db = mel * 19.8031 - 18.4903
    mel_power = 10 ** (mel_db.clamp(-80, 0) / 10.0)
    mel_cpu = mel_power.cpu()
    lin_spec = inv_mel(mel_cpu)
    audio = gl(torch.sqrt(lin_spec.clamp(min=0)).unsqueeze(0))

    peak = audio.abs().max()
    if peak > 0:
        audio = audio / peak * 0.95
    return audio


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="V14 Animal Sound Generator")
    parser.add_argument("--label", type=str, default=None, help=f"Class: {', '.join(CLASSES)}")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--v14-latent", action="store_true", help="V14 Latent Diffusion (industry standard)")
    parser.add_argument("--retrieval", action="store_true", help="Retrieval-based (quick, no training needed)")
    parser.add_argument("--variation", type=float, default=0.3, help="Retrieval variation amount")
    parser.add_argument("--cfg-scale", type=float, default=2.0, help="CFG scale for latent diffusion")
    parser.add_argument("--output-dir", type=str, default="outputs/generated")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    device = torch.device(device)
    print(f"🚀 Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    labels = [args.label] if args.label else CLASSES

    for label in labels:
        for i in range(args.count):
            if args.v14_latent:
                wav = generate_v14_latent(label, device, cfg_scale=args.cfg_scale)
                tag = "v14"
            elif args.retrieval:
                wav = generate_retrieval(label, device, args.variation)
                tag = f"ret_v{args.variation}"
            else:
                print("Specify --v14-latent or --retrieval")
                return

            if wav is None:
                print(f"  {label}: ❌ Failed"); continue

            fname = f"{label}_{tag}_{ts}_{i+1}.wav"
            out_path = os.path.join(args.output_dir, fname)
            sf.write(out_path, wav.squeeze().cpu().numpy(), 22050)
            dur = wav.shape[-1] / 22050
            rms = (wav**2).mean().sqrt()
            print(f"  {label}: ✅ {dur:.1f}s RMS={rms:.3f} → {out_path}")

    print(f"\n🎉 Done! {args.output_dir}/")


if __name__ == "__main__":
    main()
