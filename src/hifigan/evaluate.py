"""
evaluate_gan.py — Phase 7a: HiFi-GAN Vocoder Evaluation
========================================================

Compares 3 vocoder approaches on real audio:

  1. Griffin-Lim      — Math-based phase estimation (grainy baseline)
  2. HiFi-GAN melonly — No discriminator, learned phase
  3. HiFi-GAN fullGAN — With discriminator, realistic phase

Metrics:
  • Mel-L1 distance (lower = better match to original)
  • Energy ratio (closer to 1.0 = more realistic amplitude)
  • STFT distance (waveform-level difference)

Output:
  • Audio files in evaluation_gan/ for A/B listening
  • Comparison table with numbers
  • Spectrogram plots (original vs each vocoder)

Usage:
    python src/evaluate_gan.py

Requires:
    • models/hifigan_generator_meltrain_best.pth  (mel-only model)
    • models/hifigan_generator_train.pth           (full GAN model)
    • data/animal_audio/                            (test audio files)
"""
import os
import sys
import warnings
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════

CONFIG = {
    "output_dir": "outputs/evaluation_gan",
    "test_files": [
        "data/animal_audio/Dog/55154.wav",
        "data/animal_audio/Cat/100114.wav",
        "data/animal_audio/Insect/100886.wav",
        "data/animal_audio/Frog/86123.wav",
        "data/animal_audio/Crow/242401.wav",
        "data/animal_audio/Rooster/245668.wav",
        "data/animal_audio/Hen/407490.wav",
    ],
    "duration_sec": 2.0,       # listen to first 2s
    "model_dir": "models",
    "device": "auto",           # "auto" | "cuda" | "mps" | "cpu"
    "griffinlim_iters": 32,    # GL refinement iterations
}

# ═══════════════════════════════════════════════════════════
#  SETUP
# ═══════════════════════════════════════════════════════════

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.hifigan.config import config as cfg
from src.hifigan.generator import HiFiGANGenerator


def get_device():
    if CONFIG["device"] == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def audio_to_mel(audio: torch.Tensor) -> torch.Tensor:
    """Compute normalized mel spectrogram [1, 64, T]."""
    d = audio.device
    mel_fn = T.MelSpectrogram(
        sample_rate=cfg.sample_rate, n_fft=cfg.n_fft,
        hop_length=cfg.hop_length, win_length=cfg.win_length,
        n_mels=cfg.n_mels, f_min=cfg.f_min, f_max=cfg.f_max, power=2,
    ).to(d)
    db_fn = T.AmplitudeToDB(stype="power", top_db=None).to(d)
    spec = mel_fn(audio.squeeze(1))
    mel_db = db_fn(spec)
    return (mel_db - cfg.norm_mean) / cfg.norm_std


def mel_to_audio_raw(mel_norm: torch.Tensor) -> torch.Tensor:
    """Convert normalized mel → magnitude spectrogram for Griffin-Lim."""
    mel_db = mel_norm * cfg.norm_std + cfg.norm_mean
    mel_db = torch.clamp(mel_db, min=-80, max=0)
    # dB → magnitude
    mel_mag = torch.pow(10.0, mel_db / 20.0)  # [1, 64, T]

    # Mel → linear STFT magnitude via filterbank
    n_stft = cfg.n_fft // 2 + 1
    fb = torchaudio.functional.melscale_fbanks(
        n_stft, cfg.f_min, cfg.f_max, cfg.n_mels, cfg.sample_rate
    ).to(mel_mag.device)  # [n_stft, n_mels]

    # [n_stft, n_mels] @ [n_mels, T] → [n_stft, T]
    linear_mag = torch.matmul(fb, mel_mag)
    return linear_mag


def phase_refine(gan_waveform: torch.Tensor, target_mel_norm: torch.Tensor, length: int, device: torch.device, n_iter: int = 5) -> torch.Tensor:
    """Refine HiFi-GAN output using Griffin-Lim to clean up phase artifacts.
    
    Uses HiFi-GAN for amplitude (spectral shape) and Griffin-Lim to smooth the phase.
    This removes the 'electric noise' often produced by GANs.
    """
    # 1. Get Target Magnitude (from Mel)
    mel_db = target_mel_norm * cfg.norm_std + cfg.norm_mean
    mel_db = torch.clamp(mel_db, min=-80, max=0)
    mel_mag = torch.pow(10.0, mel_db / 20.0)  # [1, 64, T]
    n_stft = cfg.n_fft // 2 + 1
    
    fb = torchaudio.functional.melscale_fbanks(
        n_stft, cfg.f_min, cfg.f_max, cfg.n_mels, cfg.sample_rate
    ).to(device)
    
    target_mag = torch.matmul(fb, mel_mag)  # [1, n_stft, T]
    
    # 2. Get GAN Phase
    window = torch.hann_window(cfg.win_length).to(device)
    gan_spec = torch.stft(
        gan_waveform.squeeze(0), n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        win_length=cfg.win_length, window=window, return_complex=True, pad_mode="reflect"
    )
    
    # 3. Initialize with GAN Phase + Target Magnitude
    current_spec = target_mag * torch.exp(1j * torch.angle(gan_spec))
    
    # 4. Run GL iterations to smooth phase
    for _ in range(n_iter):
        current_waveform = torch.istft(
            current_spec, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            win_length=cfg.win_length, window=window, length=length
        )
        current_spec = torch.stft(
            current_waveform, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            win_length=cfg.win_length, window=window, return_complex=True, pad_mode="reflect"
        )
        current_spec = target_mag * torch.exp(1j * torch.angle(current_spec))
        
    # Final ISTFT
    final_waveform = torch.istft(
        current_spec, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        win_length=cfg.win_length, window=window, length=length
    )
    return final_waveform.unsqueeze(0)  # [1, L]


def griffin_lim(mel_norm: torch.Tensor, length: int, device: torch.device) -> torch.Tensor:
    """Convert normalized mel → audio via Griffin-Lim."""
    linear_mag = mel_to_audio_raw(mel_norm)

    gl = T.GriffinLim(
        n_fft=cfg.n_fft, hop_length=cfg.hop_length, power=1,
        n_iter=CONFIG["griffinlim_iters"], momentum=0.99, length=length,
    ).to(device)

    return gl(linear_mag.to(device).squeeze(0)).unsqueeze(0)  # [1, length]


def load_generator(path: str, device: torch.device) -> HiFiGANGenerator:
    """Load a HiFi-GAN checkpoint."""
    if not os.path.exists(path):
        return None
    gen = HiFiGANGenerator(cfg).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = ckpt["generator"] if "generator" in ckpt else ckpt
    gen.load_state_dict(state)
    gen.eval()
    return gen


def load_audio(path: str, device: torch.device) -> torch.Tensor:
    """Load audio, resample to 22050 Hz, trim to CONFIG duration."""
    wav, sr = torchaudio.load(path)
    if sr != cfg.sample_rate:
        wav = torchaudio.transforms.Resample(sr, cfg.sample_rate)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    target = int(cfg.sample_rate * CONFIG["duration_sec"])
    if wav.shape[-1] > target:
        wav = wav[:, :target]
    return wav.to(device)


def save_audio(audio: torch.Tensor, path: str):
    """Save audio at 22050 Hz with industry-standard low-pass filter."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if audio.dim() == 3:
        audio = audio.squeeze(0)
    if audio.dim() == 3:
        audio = audio.squeeze(1)
    
    # Industry Standard: Low-pass filter removes HiFi-GAN electric noise (>11kHz)
    audio = torchaudio.functional.lowpass_biquad(audio, cfg.sample_rate, 11025.0)
    
    torchaudio.save(path, audio, cfg.sample_rate)


def stft_distance(wav_a: torch.Tensor, wav_b: torch.Tensor) -> float:
    """Compute STFT L1 distance between two waveforms."""
    d = wav_a.device
    stft_fn = T.Spectrogram(n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                            win_length=cfg.win_length, power=1).to(d)
    spec_a = stft_fn(wav_a.squeeze(0))
    spec_b = stft_fn(wav_b.squeeze(0))
    return torch.nn.functional.l1_loss(spec_a, spec_b).item()


def mel_l1_distance(wav_a: torch.Tensor, wav_b: torch.Tensor) -> float:
    """Compute mel L1 distance between two waveforms."""
    mel_a = audio_to_mel(wav_a)
    mel_b = audio_to_mel(wav_b)
    return torch.nn.functional.l1_loss(mel_a, mel_b).item()


# ═══════════════════════════════════════════════════════════
#  PLOT
# ═══════════════════════════════════════════════════════════

def plot_spectrograms(spectrograms: dict, title: str, path: str):
    """Plot spectrograms side by side with CONSISTENT color scale."""
    n = len(spectrograms)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n))
    if n == 1:
        axes = [axes]

    # Compute global vmin/vmax for consistent color scale
    all_vals = np.concatenate([s.flatten() for s in spectrograms.values()])
    vmin = np.percentile(all_vals, 2)
    vmax = np.percentile(all_vals, 98)

    for ax, (label, spec) in zip(axes, spectrograms.items()):
        im = ax.imshow(spec, aspect="auto", origin="lower", cmap="magma",
                       vmin=vmin, vmax=vmax)  # consistent scale!
        ax.set_title(label, fontsize=11, weight="bold")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    plt.suptitle(title, fontsize=13, weight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Plot saved: {path}")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    out_dir = CONFIG["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()
    # Force CPU for HiFi-GAN inference (MPS doesn't support large Conv1d)
    device_hifigan = torch.device("cpu")

    print("=" * 70)
    print("🔧 HiFi-GAN Vocoder Evaluation")
    print("=" * 70)
    print(f"   Device (metrics): {device}")
    print(f"   Device (HiFi-GAN): {device_hifigan} (CPU for compatibility)")
    print(f"   Duration per clip: {CONFIG['duration_sec']}s")
    print(f"   GL iterations:     {CONFIG['griffinlim_iters']}")

    # ── Load models ──────────────────────────────────────
    meltrain_path = os.path.join(CONFIG["model_dir"], "hifigan_generator_meltrain_best.pth")
    gantrain_path = os.path.join(CONFIG["model_dir"], "hifigan_generator_train.pth")

    gen_mel = load_generator(meltrain_path, device_hifigan)
    gen_gan = load_generator(gantrain_path, device_hifigan)

    print(f"   Mel-only model:    {'✅' if gen_mel else '❌ not found'}")
    print(f"   GAN model:         {'✅' if gen_gan else '❌ not found'}")

    # ── Collect results ──────────────────────────────────
    all_results = []
    has_mel = gen_mel is not None
    has_gan = gen_gan is not None

    for path in CONFIG["test_files"]:
        if not os.path.exists(path):
            print(f"   ⚠️  Not found: {path}")
            continue

        cls = os.path.basename(os.path.dirname(path))
        base = os.path.splitext(os.path.basename(path))[0]

        print(f"\n{'='*60}")
        print(f"🐾 {cls}/{base}.wav")
        print(f"{'='*60}")

        # Load original
        orig = load_audio(path, device)
        orig_len = orig.shape[-1]
        mel = audio_to_mel(orig).unsqueeze(0)

        # ── 01: Griffin-Lim ──────────────────────────────
        gl = griffin_lim(mel.squeeze(0), orig_len, device).cpu()
        save_audio(gl, f"{out_dir}/{cls}_{base}_01_griffinlim.wav")
        gl_mel = mel_l1_distance(orig, gl.to(device))
        gl_stft = stft_distance(orig, gl.to(device))
        gl_energy = gl.abs().mean().item()
        print(f"  01 Griffin-Lim:  mel-L1={gl_mel:.3f}  STFT={gl_stft:.3f}  energy={gl_energy:.4f}")

        # ── 02: HiFi-GAN mel-only ────────────────────────
        hi_mel = None
        hi_stft = None
        hi_energy = None
        if has_mel:
            mel_cpu = mel.to(device_hifigan)
            with torch.no_grad():
                hi = gen_mel(mel_cpu, target_length=orig_len).cpu()
            save_audio(hi, f"{out_dir}/{cls}_{base}_02_meltrain.wav")
            hi_mel = mel_l1_distance(orig, hi.to(device))
            hi_stft = stft_distance(orig, hi.to(device))
            hi_energy = hi.abs().mean().item()
            print(f"  02 Mel-only:     mel-L1={hi_mel:.3f}  STFT={hi_stft:.3f}  energy={hi_energy:.4f}")

        # ── 03: HiFi-GAN full GAN ────────────────────────
        ga_mel = None
        ga_stft = None
        ga_energy = None
        ga_ref_mel = None
        ga_ref_stft = None
        ga_ref_energy = None
        if has_gan:
            mel_cpu = mel.to(device_hifigan)
            with torch.no_grad():
                ga = gen_gan(mel_cpu, target_length=orig_len).cpu()
            save_audio(ga, f"{out_dir}/{cls}_{base}_03_gantrain.wav")
            ga_mel = mel_l1_distance(orig, ga.to(device))
            ga_stft = stft_distance(orig, ga.to(device))
            ga_energy = ga.abs().mean().item()
            print(f"  03 Full GAN:     mel-L1={ga_mel:.3f}  STFT={ga_stft:.3f}  energy={ga_energy:.4f}")
            
            # ── 04: HiFi-GAN + Phase Refinement (Hybrid) ─
            ga_cpu = ga.clone()
            mel_cpu_ref = mel.squeeze(0).clone().to(device_hifigan)
            ga_ref = phase_refine(ga_cpu, mel_cpu_ref, orig_len, device_hifigan, n_iter=5).cpu()
            save_audio(ga_ref, f"{out_dir}/{cls}_{base}_04_gantrain_refined.wav")
            ga_ref_mel = mel_l1_distance(orig, ga_ref.to(device))
            ga_ref_stft = stft_distance(orig, ga_ref.to(device))
            ga_ref_energy = ga_ref.abs().mean().item()
            print(f"  04 GAN+Refined:  mel-L1={ga_ref_mel:.3f}  STFT={ga_ref_stft:.3f}  energy={ga_ref_energy:.4f}")

        # ── Plot spectrograms ────────────────────────────
        specs = {}
        specs["Original"] = audio_to_mel(orig).squeeze(0).cpu().numpy()
        specs["Griffin-Lim"] = audio_to_mel(gl.to(device)).squeeze(0).cpu().numpy()
        if has_mel and hi is not None:
            specs["HiFi-GAN meltrain"] = audio_to_mel(hi.to(device)).squeeze(0).cpu().numpy()
        if has_gan and ga is not None:
            specs["HiFi-GAN full GAN"] = audio_to_mel(ga.to(device)).squeeze(0).cpu().numpy()
        if has_gan and ga_ref is not None:
            specs["HiFi-GAN + Phase Refine"] = audio_to_mel(ga_ref.to(device)).squeeze(0).cpu().numpy()

        plot_spectrograms(
            specs,
            f"{cls}/{base}.wav — Spectrogram Comparison",
            f"{out_dir}/{cls}_{base}_spectrograms.png",
        )

        # ── Store results ────────────────────────────────
        all_results.append({
            "class": cls,
            "file": base,
            "gl_mel": gl_mel,
            "gl_stft": gl_stft,
            "gl_energy": gl_energy,
            "hi_mel": hi_mel,
            "hi_stft": hi_stft,
            "hi_energy": hi_energy,
            "ga_mel": ga_mel,
            "ga_stft": ga_stft,
            "ga_energy": ga_energy,
            "ga_ref_mel": ga_ref_mel,
            "ga_ref_stft": ga_ref_stft,
            "ga_ref_energy": ga_ref_energy,
            "orig_energy": orig.abs().mean().item(),
        })

    # ═══════════════════════════════════════════════════════
    #  SUMMARY TABLE
    # ═══════════════════════════════════════════════════════

    print(f"\n\n{'='*70}")
    print(f"🏆  COMPARISON TABLE — Lower is better for mel-L1 and STFT")
    print(f"{'='*70}")

    # Header
    print(f"\n  {'File':<20s} {'Vocoder':<14s} {'Mel-L1':>8s} {'STFT':>8s} {'Energy':>8s}")
    print(f"  {'─'*20} {'─'*14} {'─'*8} {'─'*8} {'─'*8}")

    for r in all_results:
        file_label = f"{r['class']}/{r['file']}"
        energy_ref = r['orig_energy']

        # Griffin-Lim
        print(f"  {file_label:<20s} {'Griffin-Lim':<14s} {r['gl_mel']:>8.3f} {r['gl_stft']:>8.3f} {r['gl_energy']:>8.4f}")

        # Mel-only
        if r["hi_mel"] is not None:
            print(f"  {'':<20s} {'Mel-only':<14s} {r['hi_mel']:>8.3f} {r['hi_stft']:>8.3f} {r['hi_energy']:>8.4f}")

        # Full GAN
        if r["ga_mel"] is not None:
            print(f"  {'':<20s} {'Full GAN':<14s} {r['ga_mel']:>8.3f} {r['ga_stft']:>8.3f} {r['ga_energy']:>8.4f}")
            
        # Full GAN + Refined
        if r["ga_ref_mel"] is not None:
            print(f"  {'':<20s} {'GAN+Refined':<14s} {r['ga_ref_mel']:>8.3f} {r['ga_ref_stft']:>8.3f} {r['ga_ref_energy']:>8.4f}")

        # Reference energy
        print(f"  {'':<20s} {'(reference)':<14s} {'':>8s} {'':>8s} {energy_ref:>8.4f}")
        print()

    # ── Averages ─────────────────────────────────────────
    n = len(all_results)
    gl_avg_mel = sum(r["gl_mel"] for r in all_results) / n
    gl_avg_stft = sum(r["gl_stft"] for r in all_results) / n

    hi_avg_mel = sum(r["hi_mel"] for r in all_results if r["hi_mel"] is not None) / max(1, sum(1 for r in all_results if r["hi_mel"] is not None))
    hi_avg_stft = sum(r["hi_stft"] for r in all_results if r["hi_stft"] is not None) / max(1, sum(1 for r in all_results if r["hi_stft"] is not None))

    ga_avg_mel = sum(r["ga_mel"] for r in all_results if r["ga_mel"] is not None) / max(1, sum(1 for r in all_results if r["ga_mel"] is not None))
    ga_avg_stft = sum(r["ga_stft"] for r in all_results if r["ga_stft"] is not None) / max(1, sum(1 for r in all_results if r["ga_stft"] is not None))

    ga_ref_avg_mel = sum(r["ga_ref_mel"] for r in all_results if r["ga_ref_mel"] is not None) / max(1, sum(1 for r in all_results if r["ga_ref_mel"] is not None))
    ga_ref_avg_stft = sum(r["ga_ref_stft"] for r in all_results if r["ga_ref_stft"] is not None) / max(1, sum(1 for r in all_results if r["ga_ref_stft"] is not None))
    
    print(f"  {'AVERAGE':<20s} {'Griffin-Lim':<14s} {gl_avg_mel:>8.3f} {gl_avg_stft:>8.3f}")
    if has_mel:
        print(f"  {'':<20s} {'Mel-only':<14s} {hi_avg_mel:>8.3f} {hi_avg_stft:>8.3f}")
    if has_gan:
        print(f"  {'':<20s} {'Full GAN':<14s} {ga_avg_mel:>8.3f} {ga_avg_stft:>8.3f}")
        print(f"  {'':<20s} {'GAN+Refined':<14s} {ga_ref_avg_mel:>8.3f} {ga_ref_avg_stft:>8.3f}")

    # ── Improvement percentage ───────────────────────────
    if has_gan and gl_avg_mel > 0:
        improvement = (1 - ga_avg_mel / gl_avg_mel) * 100
        print(f"\n  📈 GAN vs Griffin-Lim improvement: {improvement:.1f}% lower mel-L1")
    if has_gan and has_mel and hi_avg_mel > 0:
        improvement = (1 - ga_avg_mel / hi_avg_mel) * 100
        print(f"  📈 GAN vs Mel-only improvement:      {improvement:.1f}% lower mel-L1")

    print(f"\n  📁 Audio files: {out_dir}/")
    print(f"  📊 Plots:       {out_dir}/*_spectrograms.png")
    print(f"\n  🎧 Listen: play files ending in _01, _02, _03, _04 to compare")
    print(f"  ✨ 04 (GAN+Refined) should have less 'electric noise' than 03")


if __name__ == "__main__":
    main()
