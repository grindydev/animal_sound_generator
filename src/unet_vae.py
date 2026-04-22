"""
unet_vae.py — Phase 7c: U-Net VAE with Skip Connections
==========================================================

RE-PRACTICE: Same concept as NSFW Phase 5 (ResidualTunedCNN), now on VAE.

WHAT YOU'LL BUILD:
  • U-Net architecture: skip connections from encoder → decoder
  • Copy encoder feature maps to decoder at each level
  • Better reconstruction = better generation quality

KEY CONCEPTS:
  • ResNet skip:  output = F(x) + x  (same spatial size, add)
  • U-Net skip:   encoder features → concat → decoder  (different levels, concatenate)
  • Why it helps: pooling destroys fine details → skip preserves them
  • Without skip: decoder must reconstruct details from compressed latent only
  • With skip:    decoder gets fine details directly from encoder

COURSE REFERENCE:
  • NSFW residual_cnn_tuned.py — ResidualBlock, skip connections
  • L3-M1 resnet/main.py — why skip connections help deeper networks

ARCHITECTURE:
  Encoder:  conv1 → pool → conv2 → pool → conv3 → pool → latent
               ↓ skip         ↓ skip         ↓ skip
  Decoder:  up  ← concat ← up  ← concat ← up   ← latent

COMPARE WITH NSFW:
  NSFW: skip connections let you go deeper (8+ layers) without vanishing gradients
  Audio: skip connections preserve frequency details that pooling destroys
"""
