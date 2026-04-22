"""
autoencoder.py — Phase 3: Autoencoder (Compress & Reconstruct Audio)
=====================================================================

WHAT YOU'LL BUILD:
  • Encoder: CNN that compresses spectrogram → latent vector (e.g., 128-dim)
  • Decoder: ConvTranspose2d layers that expand latent vector → spectrogram
  • Reconstruction loss: MSE between input and output spectrogram
  • Griffin-Lim: convert reconstructed spectrogram back to audio
  • Listen to reconstructions — how much detail is preserved?

KEY CONCEPTS:
  • This is the FOUNDATION of your generator — the decoder learns to create spectrograms
  • Encoder = same as classifier CNN but outputs a vector (not a class)
  • Decoder = mirror of encoder using ConvTranspose2d (upsampling)
  • Latent space = compressed representation (128 dims instead of 128×87 = 11,136 values)

COURSE REFERENCE:
  • L3-M2 stable_diffusion — latent space concept, VAE in diffusion models

ARCHITECTURE:
  Encoder:
    [1, 128, 87] → Conv(1→32) → Conv(32→64) → Conv(64→128) → flatten → 128-dim
  Decoder:
    128-dim → reshape → ConvTranspose(128→64) → ConvTranspose(64→32) → ConvTranspose(32→1)
    → [1, 128, 87] reconstructed spectrogram
  Loss:
    MSE(input_spectrogram, output_spectrogram)

IMPORTANT:
  The decoder from this phase becomes your GENERATOR in Phase 4!
  If reconstruction quality is bad → generation will be bad → fix here first.
"""
