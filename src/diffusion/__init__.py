"""
src/diffusion — Phase 7b: Diffusion Refinement for Spectrograms.

Trains a small U-Net diffusion model that takes a VAE's blurry spectrogram
and denoises it into a sharp, realistic spectrogram before passing to HiFi-GAN.

Pipeline: VAE (generate) → Diffusion (sharpen) → HiFi-GAN (convert) → CRISP AUDIO
"""
