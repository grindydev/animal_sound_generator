"""
tuning.py — Phase 7b: Optuna Hyperparameter Tuning for the Generator
======================================================================

RE-PRACTICE: Same technique as NSFW Phase 3 (Optuna), now on VAE generator.

WHAT YOU'LL BUILD:
  • Optuna study: search best VAE architecture + hyperparameters
  • FlexibleVAE: variable encoder/decoder depth, filters, latent dim
  • Search space includes VAE-specific params (KL weight β)
  • Train 20 trials × 10 epochs on 50% data
  • Retrain best config on full data

KEY CONCEPTS:
  • β (KL weight) is unique to VAEs — Optuna helps find the sweet spot
  • β too low → sharp but mode-collapsed (all dog sounds identical)
  • β too high → diverse but blurry (sounds like noise)
  • Latent dim controls compression: too small = lost details, too big = wasted

COURSE REFERENCE:
  • NSFW tuning.py — Optuna search space, FlexibleCNN, train_fraction
  • L2-M1 optuna/main.py — objective function, study, trials

SEARCH SPACE:
  latent_dim:     [32, 64, 128, 256]
  encoder_depth:  [2, 3, 4, 5, 6]
  decoder_depth:  [2, 3, 4, 5, 6]
  lr:             1e-5 to 1e-2 (log scale)
  kl_weight (β):  0.001 to 10.0 (log scale)
  dropout:        0.1 to 0.5
"""
