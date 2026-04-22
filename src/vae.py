"""
vae.py — Phase 4: Conditional VAE (Generate Animal Sounds by Class)
=====================================================================

WHAT YOU'LL BUILD:
  • Variational Autoencoder with class conditioning
  • Encoder outputs μ (mean) and σ (std) instead of a fixed vector
  • Reparameterization trick: z = μ + σ * ε (ε is random noise)
  • Class embedding: nn.Embedding(num_classes, embed_dim)
  • Conditional decoder: [z + class_embedding] → spectrogram
  • KL divergence loss: keeps latent space organized
  • generate_sound(label) → random z + class → decoder → spectrogram → audio

KEY CONCEPTS:
  • VAE = Autoencoder + randomness + class conditioning
  • Same class, different z each time → different dog barks every time (diversity!)
  • KL loss forces latent space to be a normal distribution (smooth, no gaps)
  • β (KL weight) controls tradeoff: reconstruction quality vs latent organization

COURSE REFERENCE:
  • L2-M3 embeddings/main.py — nn.Embedding for class representation
  • L3-M2 stable_diffusion — text conditioning, latent space sampling
  • L3-M2 stable_diffusion — noise → denoise concept (similar to sampling z)

LOSSES:
  reconstruction_loss = MSE(output_spectrogram, input_spectrogram)
  kl_loss = -0.5 * sum(1 + log(σ²) - μ² - σ²)
  total_loss = reconstruction_loss + β * kl_loss

THIS IS YOUR GENERATOR:
  After training, you only need the DECODER + class embedding for inference.
  Generate: sample random z → add class embedding → decoder → spectrogram → audio
"""
