"""
latent_mixing.py — Phase 6a: Mix Multiple Animal Sounds in Latent Space
=========================================================================

WHAT YOU'LL BUILD:
  • Latent space arithmetic: z_mix = α * z_dog + (1-α) * z_cat
  • Interpolation: smooth transition from one animal to another
  • Weighted mixing: "70% dog + 30% cat" hybrid sound
  • Multi-way mixing: dog + cat + bird blended together
  • Listen to the transitions — does it morph smoothly?

KEY CONCEPTS:
  • Latent space is continuous — points between "dog" and "cat" are valid sounds
  • This is exactly what Stable Diffusion does for image mixing
  • If interpolation sounds glitchy → latent space is not smooth → fix VAE

COURSE REFERENCE:
  • L3-M2 stable_diffusion — latent space arithmetic, concept mixing

EXAMPLES:
  # Smooth interpolation: 0% dog → 100% cat in 10 steps
  for alpha in [0.0, 0.1, 0.2, ..., 1.0]:
      z_mix = (1 - alpha) * z_dog + alpha * z_cat
      audio = decoder(z_mix)

  # Three-way mix
  z_mix = 0.5 * z_dog + 0.3 * z_cat + 0.2 * z_bird
"""
