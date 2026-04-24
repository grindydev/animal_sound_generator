"""
vae.py — Phase 4: Conditional Variational Autoencoder (CVAE)
=============================================================

THE BIG QUESTION: Why can a VAE generate but an autoencoder cannot?

SIMPLE ANALOGY:
  Autoencoder = photocopier
    - You put in a dog photo → it makes a copy
    - It can ONLY copy what it's given — it can't create a new dog from scratch
    - Why? The encoder maps each input to ONE specific point in latent space
    - To generate, you'd need to know WHICH point to pick — but there are
      infinite wrong points for every right one. You're lost in space.

  VAE = artist who learned the "style" of each animal
    - The encoder maps each input to a REGION (not a point) in latent space
    - All dog sounds map to nearby regions → the "dog neighborhood"
    - All cat sounds map to a different neighborhood
    - These regions are organized as smooth Gaussian distributions
    - To generate a new dog: pick any point in the dog neighborhood → decode

  Think of latent space like a map:
    Autoencoder:                    VAE:
      · ·     ·                     ⬭⬭⬭     ⬯⬯⬯
         ·  ·    ·                  ⬭⬭⬭ dog  ⬯⬯⬯ cat
      ·     ·      ·                ⬭⬭⬭      ⬯⬯⬯
        ·      ·                      ← smooth regions →
    Random dots everywhere.            Organized neighborhoods.
    Pick a random point → garbage.    Pick a point in "dog" → dog sound.


THREE THINGS THAT MAKE A VAE DIFFERENT FROM AN AUTOENCODER:

  1. PROBABILISTIC ENCODING (μ, σ instead of fixed z)
     Autoencoder:  encoder(x) → z              (one point)
     VAE:          encoder(x) → μ, log_var → sample z = μ + σ * ε  (a region)

  2. KL DIVERGENCE LOSS (organizes latent space)
     Forces all latent distributions toward a standard normal N(0,1)
     This is what creates the smooth "neighborhoods" in latent space

  3. CLASS CONDITIONING (specify what to generate)
     nn.Embedding learns a vector for each class (dog, cat, etc.)
     This vector gets added to z, steering generation toward that class


HOW THE FULL PIPELINE WORKS:

  TRAINING (learn to reconstruct + organize latent space):
  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │  Dog spectrogram                                         │
  │       ↓                                                  │
  │  Encoder → μ_dog, log_var_dog                            │
  │       ↓                                                  │
  │  Sample z = μ_dog + σ_dog * random_noise                 │
  │       ↓                                                  │
  │  Add class info: z = z + class_embedding("Dog")          │
  │       ↓                                                  │
  │  Decoder → reconstructed spectrogram                     │
  │       ↓                                                  │
  │  Loss = MSE(recon, original) + β * KL_divergence         │
  │         ↑                         ↑                      │
  │    "make it look right"    "keep latent space organized"  │
  │                                                          │
  └──────────────────────────────────────────────────────────┘

  GENERATION (create new sounds from nothing):
  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │  z = random_noise ~ N(0,1)        ← random sample       │
  │       ↓                                                  │
  │  z = z + class_embedding("Dog")   ← specify class       │
  │       ↓                                                  │
  │  Decoder(z) → new spectrogram     ← never-before-seen!  │
  │       ↓                                                  │
  │  Griffin-Lim → waveform            ← playable audio     │
  │                                                          │
  │  Different random noise → different dog sound each time  │
  │  = DIVERSITY                                              │
  │                                                          │
  └──────────────────────────────────────────────────────────┘


ARCHITECTURE (compared to your SimpleAudioAutoencoder):

  ┌─────────────────────┬────────────────────────┬─────────────────────────────┐
  │  Component          │  Autoencoder (Phase 3)  │  VAE (Phase 4)              │
  ├─────────────────────┼────────────────────────┼─────────────────────────────┤
  │  Encoder            │  same 4 blocks          │  same 4 blocks              │
  │  Bottleneck encode  │  Linear → z             │  Linear → μ + Linear → σ    │
  │  Sampling           │  (none — z is fixed)    │  z = μ + σ * ε  (random!)   │
  │  Class conditioning │  (none)                 │  Embedding + add to z       │
  │  Bottleneck decode  │  Linear(flat_dim)       │  Linear(flat_dim)           │
  │  Decoder            │  same 4 blocks          │  same 4 blocks              │
  │  Loss               │  MSE only               │  MSE + β * KL_divergence    │
  │  forward() returns  │  reconstructed           │  reconstructed, μ, log_var │
  └─────────────────────┴────────────────────────┴─────────────────────────────┘

  Same encoder + decoder. Only the bottleneck and loss change.


COURSE REFERENCE:
  • L3-M2 stable_diffusion — latent space, sampling from distributions
  • L2-M3 embeddings — nn.Embedding for categorical features
"""

import torch
import torch.nn as nn
from model import SimpleEncoderBlock, SimpleDecoderBlock


class SimpleAudioVAE(nn.Module):
    """
    Conditional Variational Autoencoder for animal sound spectrograms.

    Compared to SimpleAudioAutoencoder, THREE things change:

      1. Bottleneck outputs μ and log_var (not a single z)
         → enables sampling from a learned distribution
      2. Class embedding injected into z
         → specifies WHICH animal to generate
      3. Forward returns (reconstructed, mu, log_var)
         → training needs μ and log_var for KL loss

    Everything else (encoder conv blocks, decoder conv blocks) is IDENTICAL.
    """

    def __init__(self, latent_dim=1024, num_classes=8, embed_dim=64):
        """
        Args:
            latent_dim:  size of the latent vector z (same as autoencoder)
            num_classes: number of animal classes (8: Dog, Cat, Rooster, ...)
            embed_dim:   size of each class embedding vector
        """
        super().__init__()

        # ── Encoder — EXACTLY the same as SimpleAudioAutoencoder ──
        # Why same? The encoder's job hasn't changed: compress spectrogram → features
        self.encode = nn.Sequential(
            SimpleEncoderBlock(1, 32),
            SimpleEncoderBlock(32, 64),
            SimpleEncoderBlock(64, 128),
            SimpleEncoderBlock(128, 256),
        )

        # After 4 encoder blocks with stride=2: [B, 256, 4, 35]
        self.flat_dim = 256 * 4 * 35  # 35,840

        # ── VAE Bottleneck — THIS IS THE KEY DIFFERENCE ──
        #
        # Autoencoder had:
        #   self.fc_encode = nn.Linear(self.flat_dim, latent_dim)  → z
        #
        # VAE has TWO linear layers instead of one:
        #   self.fc_mu      → μ  (mean of the distribution)
        #   self.fc_log_var → log(σ²)  (log variance of the distribution)
        #
        # Instead of mapping to one fixed point z, the encoder maps to a
        # DISTRIBUTION described by μ and σ².
        #
        # We store log(σ²) instead of σ directly because:
        #   - σ must be positive (it's a standard deviation)
        #   - log(σ²) can be any real number → easier for the network to learn
        #   - To get σ: σ = exp(0.5 * log_var) = sqrt(exp(log_var))
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flat_dim, latent_dim)

        # ── Class Conditioning — ENTIRELY NEW ──
        #
        # nn.Embedding(num_classes, embed_dim) creates a lookup table:
        #   class 0 (Dog)     → [0.3, -0.1, 0.8, ...]   (64-dim vector)
        #   class 1 (Cat)     → [-0.2, 0.5, 0.1, ...]   (different vector)
        #   class 2 (Rooster) → [0.7, 0.3, -0.4, ...]   (different again)
        #
        # These vectors are LEARNED during training — the model figures out
        # what makes each class unique.
        #
        # class_project projects embed_dim → latent_dim so we can ADD it to z.
        # Addition is simpler than concatenation and works well when latent_dim
        # is already large (1024).
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        self.class_project = nn.Linear(embed_dim, latent_dim)

        # ── Decoder — EXACTLY the same as SimpleAudioAutoencoder ──
        # Why same? The decoder's job hasn't changed: latent vector → spectrogram
        # It just receives a slightly different z (conditioned on class)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)
        self.decode = nn.Sequential(
            SimpleDecoderBlock(256, 128),
            SimpleDecoderBlock(128, 64),
            SimpleDecoderBlock(64, 32),
            SimpleDecoderBlock(32, 1, activation=False),
        )

    def reparameterize(self, mu, log_var):
        """
        THE REPARAMETERIZATION TRICK — why VAEs can be trained with backprop.

        We want to sample: z ~ N(μ, σ²)
        Naive approach: z = torch.normal(mu, sigma)
        Problem: sampling is a RANDOM operation → gradients can't flow through it
                 → backprop breaks → model can't learn

        The trick: rewrite sampling as a deterministic operation:
          z = μ + σ * ε     where ε ~ N(0, 1)  (random noise, NOT learned)

        Now μ and σ appear as simple arithmetic — gradients flow through them!
        The randomness (ε) is just an input, like the data itself.

        Analogy:
          "Make x random by adding noise" — we don't need to differentiate
          through the noise source, only through x.

        Why log_var instead of var:
          log_var can be any real number (-∞ to +∞)
          var = exp(log_var) is guaranteed positive (σ² must be positive)
          std = exp(0.5 * log_var) = sqrt(var)

        Args:
            mu:      mean of latent distribution      [B, latent_dim]
            log_var: log variance of latent distribution [B, latent_dim]

        Returns:
            z: sampled latent vector                   [B, latent_dim]
        """
        std = torch.exp(0.5 * log_var)  # σ = sqrt(exp(log_var))
        eps = torch.randn_like(std)      # ε ~ N(0,1), same shape as std
        return mu + std * eps             # z = μ + σ * ε

    def encode_to_params(self, x):
        """
        Encode a spectrogram → (μ, log_var).

        Same encoder as autoencoder, but instead of one z we get two outputs.
        This is used by forward() during training.

        Args:
            x: spectrogram [B, 1, 64, W]

        Returns:
            mu:      [B, latent_dim]
            log_var: [B, latent_dim]
        """
        h = self.encode(x)          # [B, 256, 4, 35]
        h = h.flatten(start_dim=1)   # [B, 35,840]
        mu = self.fc_mu(h)           # [B, latent_dim]
        log_var = self.fc_log_var(h) # [B, latent_dim]
        return mu, log_var

    def decode_from_z(self, z, target_size):
        """
        Decode a latent vector → spectrogram.

        Same decoder as autoencoder. Only z is different (conditioned on class).

        Args:
            z:           latent vector [B, latent_dim]
            target_size: (H, W) to resize output

        Returns:
            reconstructed spectrogram [B, 1, H, W]
        """
        h = self.fc_decode(z)             # [B, 35,840]
        h = h.view(-1, 256, 4, 35)        # [B, 256, 4, 35]
        h = self.decode(h)                 # [B, 1, 64, 560]
        h = nn.functional.interpolate(h, size=target_size, mode='bilinear')
        return h

    def forward(self, x, labels):
        """
        Full forward pass: encode → sample → condition → decode.

        This is called during TRAINING. It uses the encoder to get μ, σ,
        samples z from the distribution, conditions on class, and decodes.

        Args:
            x:      spectrogram [B, 1, 64, W]
            labels: class indices [B] (0=Dog, 1=Cat, ...)

        Returns:
            reconstructed: [B, 1, 64, W]  — the reconstructed spectrogram
            mu:            [B, latent_dim] — mean (needed for KL loss)
            log_var:       [B, latent_dim] — log variance (needed for KL loss)
        """
        target_size = x.shape[2:]

        # 1. Encode → get distribution parameters
        #    Autoencoder: z = self.fc_encode(flatten(encode(x)))
        #    VAE:         mu, log_var = fc_mu(flatten(...)), fc_log_var(flatten(...))
        mu, log_var = self.encode_to_params(x)

        # 2. Sample from the distribution
        #    Autoencoder: z is fixed (no randomness)
        #    VAE:         z is sampled from N(μ, σ²) — introduces randomness!
        z = self.reparameterize(mu, log_var)

        # 3. Add class conditioning
        #    Autoencoder: z goes straight to decoder
        #    VAE:         z is adjusted by class embedding first
        #
        #    class_embed(labels) → [B, embed_dim]   look up class vector
        #    class_project(...) → [B, latent_dim]   resize to match z
        #    z + project(...)   → [B, latent_dim]   inject class info
        #
        #    After this: z contains BOTH the audio content (from encoder)
        #    AND the class identity (from embedding).
        #    The decoder learns: "given this z, generate a spectrogram
        #    that sounds like this class."
        class_emb = self.class_project(self.class_embed(labels))  # [B, latent_dim]
        z = z + class_emb

        # 4. Decode — same as autoencoder from here
        reconstructed = self.decode_from_z(z, target_size)

        return reconstructed, mu, log_var

    @torch.no_grad()
    def sample(self, label, num_samples=1, device='cpu'):
        """
        GENERATE new animal sounds! No encoder needed — just random noise + class.

        This is the whole point of the VAE. During training, the model learned:
          - What the latent space looks like (KL loss pushed it toward N(0,1))
          - How to decode latent vectors into spectrograms
          - What each class "sounds like" (class embeddings)

        At generation time, we skip the encoder entirely and:
          1. Sample random noise from N(0,1) — because KL loss organized
             the latent space to BE approximately N(0,1), random points
             will produce meaningful outputs
          2. Add the class embedding — steers the random point toward
             the "dog neighborhood" or "cat neighborhood"
          3. Decode — the decoder turns this into a spectrogram

        Different random noise → different sound (DIVERSITY)
        Same class label → same animal type (CONDITIONING)

        Args:
            label:       int (0-7) or list of ints, which animal to generate
            num_samples: how many different sounds to generate
            device:      'cpu', 'cuda', or 'mps'

        Returns:
            generated spectrogram [num_samples, 1, 64, 552]
        """
        self.eval()

        # Accept int or list
        if isinstance(label, int):
            labels = torch.full((num_samples,), label, dtype=torch.long, device=device)
        else:
            labels = torch.tensor(label, dtype=torch.long, device=device)
            num_samples = len(labels)

        # 1. Sample random z from standard normal
        #    This works because KL loss trained the latent space to be ~N(0,1)
        z = torch.randn(num_samples, self.fc_mu.out_features, device=device)

        # 2. Add class conditioning (same as in forward())
        class_emb = self.class_project(self.class_embed(labels))
        z = z + class_emb

        # 3. Decode — use a default target size (64 mel bins, 552 time frames)
        #    This matches 5 seconds of audio at 22050 Hz
        generated = self.decode_from_z(z, target_size=(64, 552))

        return generated

    @torch.no_grad()
    def interpolate(self, x1, label1, x2, label2, steps=10, device='cpu'):
        """
        Smoothly morph from one animal sound to another.

        HOW IT WORKS:
          1. Encode both sounds → z1 and z2 (their positions in latent space)
          2. Draw a straight line between z1 and z2
          3. Decode points along that line
          4. First outputs sound like animal1, last like animal2, middle = hybrid

        This works because KL loss made the latent space SMOOTH — nearby points
        produce similar sounds. A straight line from "dog" to "cat" passes through
        points that sound like a smooth blend of both.

        Args:
            x1:     spectrogram [1, 1, 64, W] — first sound
            label1: int — class of first sound
            x2:     spectrogram [1, 1, 64, W] — second sound
            label2: int — class of second sound
            steps:  int — how many interpolation steps (more = smoother)
            device: str

        Returns:
            list of [1, 1, 64, W] spectrograms, from sound1 → sound2
        """
        self.eval()
        x1, x2 = x1.to(device), x2.to(device)

        # Encode both sounds (use μ as the "position", no randomness)
        mu1, _ = self.encode_to_params(x1)
        mu2, _ = self.encode_to_params(x2)

        # Add class embeddings to each
        label1_t = torch.tensor([label1], device=device)
        label2_t = torch.tensor([label2], device=device)
        z1 = mu1 + self.class_project(self.class_embed(label1_t))
        z2 = mu2 + self.class_project(self.class_embed(label2_t))

        # Interpolate: z = (1-α)*z1 + α*z2, for α from 0 to 1
        results = []
        for alpha in torch.linspace(0, 1, steps):
            z = (1 - alpha) * z1 + alpha * z2
            recon = self.decode_from_z(z, target_size=x1.shape[2:])
            results.append(recon)

        return results
