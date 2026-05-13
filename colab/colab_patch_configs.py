#!/usr/bin/env python3
"""
colab_patch_configs.py — Patch training configs for Colab T4 GPU (16GB VRAM).

Run this ONCE in Colab after cloning the repo. It bumps batch sizes and
base_channels to take advantage of the larger GPU.

Usage (in Colab):
    python colab_patch_configs.py
"""
import re
import os

PATCHES = {
    "src/vae/train_ae.py": [
        # base_channels 16 → 32 (149M params — fits T4 easily)
        (r'"base_channels":\s*16', '"base_channels": 32'),
        # batch_size 2 → 8 (4× more headroom)
        (r'("train"[^}]*"batch_size":\s*)2', r'\g<1>8'),
        (r'("test"[^}]*"batch_size":\s*)2', r'\g<1>4'),
    ],
    "src/vae/finetune.py": [
        (r'"base_channels":\s*16', '"base_channels": 32'),
        # batch_size 1 → 4, keep grad_accum=2 → effective=8
        (r'("train"[^}]*"batch_size":\s*)1', r'\g<1>4'),
        (r'("test"[^}]*"batch_size":\s*)1', r'\g<1>2'),
    ],
    "src/diffusion/train.py": [
        # batch_size 4 → 16 for diffusion
        (r'("train"[^}]*"batch_size":\s*)4', r'\g<1>16'),
        (r'("test"[^}]*"batch_size":\s*)4', r'\g<1>8'),
    ],
    "src/hifigan/train.py": [
        # batch_size 8 → 16 for HiFi-GAN
        (r'("train"[^}]*"batch_size":\s*)8', r'\g<1>16'),
        (r'("test"[^}]*"batch_size":\s*)4', r'\g<1>8'),
    ],
    "src/train_classifier.py": [
        # Already fine at batch=64, but bump for speed
        (r'("train"[^}]*"batch_size":\s*)64', r'\g<1>128'),
    ],
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # colab/ → project root

for rel_path, replacements in PATCHES.items():
    full_path = os.path.join(ROOT, rel_path)
    if not os.path.exists(full_path):
        print(f"⚠️  Skipping {rel_path} — file not found")
        continue

    with open(full_path, 'r') as f:
        content = f.read()

    original = content
    for pattern, replacement in replacements:
        # Try regex first, then literal
        try:
            content = re.sub(pattern, replacement, content)
        except Exception:
            content = content.replace(pattern, replacement)

    if content != original:
        with open(full_path, 'w') as f:
            f.write(content)
        print(f"✅ Patched: {rel_path}")
    else:
        print(f"ℹ️  No changes needed: {rel_path}")

print("\n🎉 All configs patched for Colab T4!")
print("\nSummary of changes:")
print("  Autoencoder:  base_ch=32 (149M params), batch=8")
print("  VAE finetune: base_ch=32 (223M params), batch=4, grad_accum=2")
print("  Diffusion:    batch=16")
print("  HiFi-GAN:     batch=16")
print("  Classifier:   batch=128")
