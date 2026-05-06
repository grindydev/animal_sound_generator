"""
diagnose_audio.py — Quick test to isolate where the noise comes from.

Tests 3 paths:
  A: Real spectrogram → spectrogram_to_waveform() → .wav
  B: Real audio → VAE encode → VAE decode → spectrogram_to_waveform() → .wav
  C: Random z + class → VAE decode → spectrogram_to_waveform() → .wav
"""
import torch, sys, os
sys.path.insert(0, "src")

from data_loader import AnimalSoundDataset, get_transformations
from vae import SimpleAudioVAE
from audio_utils import spectrogram_to_waveform

device = torch.device("mps")
CLASS_NAMES = ["Dog", "Cat", "Rooster", "Frog", "Crow", "Insect", "Hen", "Noise"]
SAMPLE_RATE = 44100
N_FFT = 400       # Must match MelSpectrogram default
HOP_LENGTH = 200  # Must match MelSpectrogram default
N_MELS = 64

# ── Load models ──────────────────────────────────────────────
_, eval_tfm = get_transformations()
eval_tfm = eval_tfm.to(device)

vae = SimpleAudioVAE(latent_dim=1024, num_classes=8, embed_dim=64).to(device)
ckpt = torch.load("models/best_vae_finetune_train.pth", map_location=device, weights_only=True)
state = ckpt.get("model_state_dict", ckpt)
ms = vae.state_dict()
filt = {k: v for k, v in state.items() if k in ms and v.shape == ms[k].shape}
vae.load_state_dict(filt, strict=False)
vae.eval()
print(f"✅ VAE loaded: {len(filt)} keys")

# ── Get one real sample ──────────────────────────────────────
ds = AnimalSoundDataset("data/animal_audio")
wav, lbl_idx = ds[10]
lbl_idx = int(lbl_idx) if not isinstance(lbl_idx, int) else lbl_idx
cls_name = CLASS_NAMES[lbl_idx]
print(f"\n📁 Test sample: class={cls_name}, label={lbl_idx}")

wav_batch = wav.unsqueeze(0).to(device)  # [1, 1, samples]
real_spec = eval_tfm(wav_batch)           # [1, 1, 64, T]

# ── VAE reconstruction ───────────────────────────────────────
with torch.no_grad():
    recon_spec, mu, log_var = vae(real_spec, torch.tensor([lbl_idx]).to(device))
    gen_spec = vae.sample(label=lbl_idx, num_samples=1, device=device, temperature=0.7)

mse = torch.nn.functional.mse_loss(recon_spec, real_spec).item()
print(f"🔍 Reconstruction MSE: {mse:.6f}")
print(f"   Real spec:  [{real_spec.min():.4f}, {real_spec.max():.4f}]  mean={real_spec.mean():.4f}")
print(f"   Recon spec: [{recon_spec.min():.4f}, {recon_spec.max():.4f}]  mean={recon_spec.mean():.4f}")
print(f"   Gen spec:   [{gen_spec.min():.4f}, {gen_spec.max():.4f}]  mean={gen_spec.mean():.4f}")

# ── Convert to audio ─────────────────────────────────────────
wav_a = spectrogram_to_waveform(real_spec, SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS)
wav_b = spectrogram_to_waveform(recon_spec, SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS)
wav_c = spectrogram_to_waveform(gen_spec, SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS)

# Squeeze to [samples] for saving
wav_a_1d = wav_a.squeeze()
wav_b_1d = wav_b.squeeze()
wav_c_1d = wav_c.squeeze()
wav_orig_1d = wav.squeeze()

print(f"\n🅰️  PATH A (real spec → audio):  range=[{wav_a_1d.min():.4f}, {wav_a_1d.max():.4f}]  rms={wav_a_1d.pow(2).mean().sqrt():.4f}")
print(f"🅱️  PATH B (VAE recon → audio): range=[{wav_b_1d.min():.4f}, {wav_b_1d.max():.4f}]  rms={wav_b_1d.pow(2).mean().sqrt():.4f}")
print(f"🅲  PATH C (VAE gen → audio):   range=[{wav_c_1d.min():.4f}, {wav_c_1d.max():.4f}]  rms={wav_c_1d.pow(2).mean().sqrt():.4f}")
print(f"📀  ORIGINAL audio:             range=[{wav_orig_1d.min():.4f}, {wav_orig_1d.max():.4f}]  rms={wav_orig_1d.pow(2).mean().sqrt():.4f}")

# ── Save ─────────────────────────────────────────────────────
import torchaudio
os.makedirs("diagnosis", exist_ok=True)

torchaudio.save(f"diagnosis/{cls_name}_ORIGINAL.wav", wav_orig_1d.unsqueeze(0).cpu(), SAMPLE_RATE)
torchaudio.save(f"diagnosis/{cls_name}_A_real.wav", wav_a_1d.unsqueeze(0).cpu(), SAMPLE_RATE)
torchaudio.save(f"diagnosis/{cls_name}_B_recon.wav", wav_b_1d.unsqueeze(0).cpu(), SAMPLE_RATE)
torchaudio.save(f"diagnosis/{cls_name}_C_generated.wav", wav_c_1d.unsqueeze(0).cpu(), SAMPLE_RATE)

print(f"\n💾 Saved to diagnosis/:")
print(f"   {cls_name}_ORIGINAL.wav  — original audio (no conversion)")
print(f"   {cls_name}_A_real.wav    — real spec → spectrogram_to_waveform")
print(f"   {cls_name}_B_recon.wav   — encode→decode → spectrogram_to_waveform")
print(f"   {cls_name}_C_generated.wav — random z + class → decode → .wav")
print(f"\n🎧 Listen in order: ORIGINAL → A → B → C")
