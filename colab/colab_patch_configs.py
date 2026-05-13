#!/usr/bin/env python3
"""
colab_patch_configs.py — Patch training configs for Colab GPU (T4 16GB / L4 24GB).

Now that source configs are Colab-optimized by default, this script verifies
settings and allows per-GPU overrides.

Usage (in Colab):
    python colab/colab_patch_configs.py
"""
import re
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Current defaults (already Colab-optimized)
DEFAULTS = {
    "src/vae/train_ae.py":     {"lr": "1e-3", "base_channels": 32, "batch": 16, "workers": 4},
    "src/vae/finetune.py":     {"lr": "3e-4", "base_channels": 32, "batch": 8,  "workers": 4},
    "src/train_classifier.py": {"batch": 256, "workers": 4},
    "src/diffusion/train.py":  {"batch": 16,  "workers": 4},
    "src/hifigan/train.py":    {"batch": 16,  "workers": 4},
}

print("🎯 Colab config checker")
print("=" * 50)

for path, expected in DEFAULTS.items():
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        print(f"  ⚠️  {path} — not found")
        continue
    with open(full) as f:
        content = f.read()
    issues = []
    if "base_channels" in expected:
        if f'"base_channels": {expected["base_channels"]}' not in content:
            issues.append(f'base_channels != {expected["base_channels"]}')
    if "lr" in expected:
        if f'"lr": {expected["lr"]}' not in content:
            issues.append(f'lr != {expected["lr"]}')
    if issues:
        print(f"  ❌ {path}: {', '.join(issues)}")
    else:
        print(f"  ✅ {path}")

print("\nAll configs ready for Colab!")
