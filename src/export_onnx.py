"""
export_onnx.py — Phase 8: Export Generator to ONNX
=====================================================

WHAT YOU'LL BUILD:
  • Load best VAE generator checkpoint
  • Export decoder (the generation part) to ONNX format
  • Accept command-line argument for model selection (like NSFW)
  • Auto-detect model type from checkpoint keys
  • Write model metadata (type, input_size) into ONNX file
  • List available models with ← selected marker

KEY CONCEPTS:
  • Only export the DECODER + class embedding (not the encoder)
  • At inference: random z + class embedding → decoder → spectrogram
  • ONNX makes it portable — no PyTorch dependency in production

COURSE REFERENCE:
  • NSFW export_onnx.py — ONNX export, model auto-detection, CLI args
  • L3-M4 ONNX/main.py — torch.onnx.export()

USAGE:
  python export_onnx.py                          # auto-pick best model
  python export_onnx.py best_vae_unet.pth        # specific model
"""
