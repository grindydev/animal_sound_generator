"""
diffusion_refine.py — Phase 6c: Diffusion Model for Audio Refinement
======================================================================

WHAT YOU'LL BUILD:
  • Forward diffusion: gradually add Gaussian noise to a spectrogram
  • Reverse diffusion: U-Net learns to predict and remove noise
  • Conditional on animal class (guide denoising toward target animal)
  • Use VAE output as starting point → diffusion cleans it up
  • Same architecture as Stable Diffusion but on spectrograms!

KEY CONCEPTS:
  • Your VAE output is rough → diffusion makes it sharper and more realistic
  • This is EXACTLY what Stable Diffusion does (L3-M2): VAE + Diffusion
  • Forward: clean spectrogram → add noise T times → pure noise
  • Reverse: pure noise → U-Net denoises T steps → clean spectrogram
  • More denoising steps = higher quality but slower

COURSE REFERENCE:
  • L3-M2 stable_diffusion — DDPM pipeline, forward/reverse process
  • L3-M2 stable_diffusion — DDPM bedroom model (pixel-space diffusion)
  • L3-M2 stable_diffusion — guidance_scale, inference steps

PIPELINE:
  VAE generates rough spectrogram
       → add noise (forward diffusion, T steps)
       → U-Net denoise (reverse diffusion, T steps, conditioned on animal class)
       → refined spectrogram
       → Griffin-Lim → audio
"""
