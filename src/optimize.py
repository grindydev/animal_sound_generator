"""
optimize.py — Phase 7e: Pruning + Quantization on the Generator
==================================================================

RE-PRACTICE: Same technique as NSFW Phase 7b/7c, now on the VAE generator.

WHAT YOU'LL BUILD:
  • L1 unstructured pruning on decoder (the inference component)
  • Fine-tune 3-5 epochs to recover quality after pruning
  • Dynamic quantization: FP32 → INT8
  • ONNX export for cross-platform deployment
  • Benchmark: speed vs quality tradeoff

KEY CONCEPTS:
  • Only prune the DECODER — that's what runs at inference time
  • ConvTranspose2d can also be pruned (same as Conv2d)
  • Same lessons as NSFW: small models don't benefit much from pruning

COURSE REFERENCE:
  • NSFW prune.py — pruning pipeline, fine-tune recovery, quantization
  • L3-M4 pruning/main.py — L1 unstructured pruning
  • L3-M4 quantization/main.py — dynamic quantization
  • L3-M4 metro_city/main.py — full optimization pipeline

MEASURE:
  Original:    ??? ms per sound
  Pruned 30%:  ??? ms  (quality drop: ???%)
  Quantized:   ??? ms  (quality drop: ???%)
  Real-time capable? (< 100ms)

SAME LESSON AS NSFW:
  Small models (your VAE ~2MB) don't benefit much from pruning.
  Large models (ResNet18 43MB) benefit much more.
"""
