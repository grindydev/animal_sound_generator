"""
grad_cam_audio.py — Phase 7d: Grad-CAM on Spectrograms
==========================================================

RE-PRACTICE: Same technique as NSFW Phase 6 (Grad-CAM), now on audio.

WHAT YOU'LL BUILD:
  • Grad-CAM on spectrograms — same algorithm, overlay on spectrogram
  • For classifier: what frequency/time regions matter for each animal?
  • For VAE decoder: which latent dimensions activate which frequencies?
  • Compare attention maps: dog (low freq) vs bird (high freq) vs frog (narrow band)

KEY CONCEPTS:
  • Grad-CAM works the same on spectrograms as on images
  • Y-axis = frequency → shows WHAT frequencies the model attends to
  • X-axis = time → shows WHEN the model focuses
  • Debug generation: if decoder ignores high frequencies → bird sounds fail

COURSE REFERENCE:
  • NSFW grad_cam.py — GradCAM class, forward/backward hooks
  • L3-M2 saliency_and_class_activation_map — Grad-CAM implementation
  • L3-M2 interpreting — filter + feature map visualization

EXPECTED DISCOVERIES:
  Dog bark:  LOW frequencies + sharp temporal onset
  Cat meow:  MID frequencies + sustained tone
  Bird:      HIGH frequencies + rapid temporal patterns
  Frog:      NARROW frequency band + repetitive pattern

  On VAE decoder:
  If dog generation has weak low frequencies → decoder didn't learn spectral profile
"""
