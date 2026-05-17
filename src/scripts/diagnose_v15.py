"""
diagnose_v15.py — Check what's broken in the v15 pipeline

Tests each component individually:
  1. Decoder reconstruction: real mel → encode → decode → should be close
  2. Diffusion DDIM: noise → latent → decode → check if all-noise or structured
  3. Compare original vs generated for one sample

Usage: python src/scripts/diagnose_v15.py
"""
import os, sys, torch, torchaudio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, InverseMelScale, GriffinLim

from src.latent_diff.config import config as cfg
from src.latent_diff.decoder import LatentDecoder, ChannelReducer, ChannelExpander
from src.latent_diff.unet import LatentUNet
from src.vae.autoencoder import ImprovedAutoencoder
from src.diffusion.diffusion import DiffusionProcess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load encoder
ckpt = torch.load("models/best_autoencoder_train.pth", map_location='cpu', weights_only=True)
state = ckpt.get('model_state_dict', ckpt.get('model', {}))
fc_w = state.get('fc_encode.weight')
flat_dim = fc_w.shape[1] if fc_w is not None else 0
base_ch = (flat_dim // (4*35)) // 8
encoder = ImprovedAutoencoder(latent_dim=2048, base_channels=base_ch).to(device)
encoder.load_state_dict(state)
encoder.eval()

# Load decoder
dec_ckpt = torch.load("models/latent_decoder_best.pth", map_location=device, weights_only=True)
loaded_ch = dec_ckpt.get('bottleneck_ch', encoder.c4)
reducer = ChannelReducer(in_ch=encoder.c4, out_ch=cfg.latent_channels).to(device)
reducer.load_state_dict(dec_ckpt['reducer'])
reducer.eval()
expander = ChannelExpander(in_ch=cfg.latent_channels, out_ch=loaded_ch).to(device)
expander.load_state_dict(dec_ckpt['expander'])
expander.eval()
decoder = LatentDecoder(bottleneck_ch=loaded_ch, config=cfg).to(device)
decoder.load_state_dict(dec_ckpt['decoder'])
decoder.eval()

# Load diffusion
diff_ckpt = torch.load("models/latent_diffusion_best.pth", map_location=device, weights_only=True)
unet = LatentUNet(cfg).to(device)
unet.load_state_dict(diff_ckpt['unet'])
unet.eval()
latent_mean = diff_ckpt.get('latent_mean', 0.0)
latent_std = diff_ckpt.get('latent_std', 1.0)
diffusion = DiffusionProcess(cfg).to(device)
diffusion.num_classes = cfg.num_classes

print(f"Latent stats: mean={latent_mean:.4f}, std={latent_std:.4f}")

# Load a real dog wav
wav_path = "data/animal1000/Dog/1-100032-A-0.wav"
if not os.path.exists(wav_path):
    wavs = [f for f in os.listdir("data/animal1000/Dog") if f.endswith('.wav') and not f.startswith('aug_')]
    wav_path = f"data/animal1000/Dog/{wavs[0]}"
print(f"\nTest file: {wav_path}")

wav, sr = torchaudio.load(wav_path)
wav = wav[:, :5*22050]
if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)

# Compute mel
mel_tfm = MelSpectrogram(22050, 1024, hop_length=200, n_mels=64, f_min=0, f_max=11025, power=2)
db_tfm = AmplitudeToDB(top_db=80)
spec = mel_tfm(wav)
mel_db = db_tfm(spec)
mel_norm = (mel_db + 18.4903) / 19.8031
mel_in = mel_norm.unsqueeze(0).to(device)  # [1, 1, 64, 552]
print(f"Real mel: mean={mel_in.mean():.3f}, std={mel_in.std():.3f}")

# ═══ TEST 1: Decoder reconstruction ═══
print("\n── TEST 1: Decoder reconstruction ──")
with torch.no_grad():
    features = encoder.encode_spatial(mel_in)
    print(f"  Encoded spatial: {features.shape}, mean={features.mean():.3f}, std={features.std():.3f}")
    
    latent = reducer(features)
    print(f"  Reduced: {latent.shape}, mean={latent.mean():.3f}, std={latent.std():.3f}")
    
    expanded = expander(latent)
    print(f"  Expanded: {expanded.shape}, mean={expanded.mean():.3f}, std={expanded.std():.3f}")
    
    recon = decoder(expanded, target_size=(64, 552))
    print(f"  Reconstructed mel: mean={recon.mean():.3f}, std={recon.std():.3f}")
    print(f"  L1 error vs real: {(recon - mel_in).abs().mean():.4f}")

# ═══ TEST 2: Generation from noise ═══
print("\n── TEST 2: Diffusion generation ──")
with torch.no_grad():
    noise = torch.randn(1, cfg.latent_channels, cfg.latent_height, cfg.latent_width, device=device)
    labels = torch.tensor([0], device=device)
    
    # DDIM x0-prediction
    gen_latent = diffusion.ddim_sample_x0(unet, noise, labels, num_steps=50, eta=0.0, cfg_scale=2.0)
    gen_latent_denorm = gen_latent * latent_std + latent_mean
    
    print(f"  Generated latent: mean={gen_latent.mean():.3f}, std={gen_latent.std():.3f}")
    print(f"  Denormalized: mean={gen_latent_denorm.mean():.3f}, std={gen_latent_denorm.std():.3f}")
    
    gen_features = expander(gen_latent_denorm)
    print(f"  Expanded features: mean={gen_features.mean():.3f}, std={gen_features.std():.3f}")
    
    gen_mel = decoder(gen_features, target_size=(64, 552))
    print(f"  Generated mel: mean={gen_mel.mean():.3f}, std={gen_mel.std():.3f}")

# ═══ Compare with real latents ═══
print("\n── TEST 3: Real vs Generated latent stats ──")
with torch.no_grad():
    latent_real = (latent - latent_mean) / latent_std
    print(f"  Real latent (normalized): mean={latent_real.mean():.3f}, std={latent_real.std():.3f}")
    print(f"  Real latent range: [{latent_real.min():.2f}, {latent_real.max():.2f}]")
    print(f"  Gen latent range: [{gen_latent.min():.2f}, {gen_latent.max():.2f}]")

# ═══ Check if generated mel looks like a real spectrogram ═══
print("\n── TEST 4: Spectrogram structure check ──")
# A real spectrogram has energy at specific frequencies (vertical stripes)
# Noise has flat energy
gen_mel_db = gen_mel.squeeze() * 19.8031 - 18.4903
real_mel_db = mel_db.to(device)

# Energy per frequency band
for name, sl in [("Low (0-2kHz)", slice(0,12)), ("Mid (2-5kHz)", slice(12,30)), ("High (5-11kHz)", slice(30,64))]:
    real_e = real_mel_db[0, sl, :].mean()
    gen_e = gen_mel_db[sl, :].mean()
    print(f"  {name}: real={real_e:.1f}dB, gen={gen_e:.1f}dB")

# ═══ Convert to audio ═══
print("\n── TEST 5: Audio conversion ──")
inv_mel = InverseMelScale(n_stft=513, n_mels=64, sample_rate=22050, f_min=0, f_max=11025)
gl = GriffinLim(n_fft=1024, hop_length=200, win_length=1024, n_iter=64, power=1)

# Real reconstruction audio
recon_db = recon.squeeze() * 19.8031 - 18.4903
recon_power = 10 ** (recon_db.clamp(-80, 0) / 10.0)
recon_audio = gl(torch.sqrt(inv_mel(recon_power.cpu()).clamp(min=0)).unsqueeze(0))

# Generated audio
gen_db = gen_mel.squeeze() * 19.8031 - 18.4903
gen_power = 10 ** (gen_db.clamp(-80, 0) / 10.0)
gen_audio = gl(torch.sqrt(inv_mel(gen_power.cpu()).clamp(min=0)).unsqueeze(0))

print(f"  Recon audio RMS: {(recon_audio**2).mean().sqrt():.3f}")
print(f"  Gen audio RMS: {(gen_audio**2).mean().sqrt():.3f}")

torchaudio.save("outputs/diag_recon.wav", recon_audio.squeeze(0).cpu(), 22050)
torchaudio.save("outputs/diag_gen.wav", gen_audio.squeeze(0).cpu(), 22050)
print("\nListen: outputs/diag_recon.wav (reconstruction)")
print("        outputs/diag_gen.wav (generation)")
