"""
helper_utils.py — Shared Utilities
=====================================

Copied from NSFW project — same patterns apply here.

WHAT'S HERE:
  • plot_training_metrics() — plot train/val loss + val accuracy curves
  • set_seed() — reproducible experiments

COURSE REFERENCE:
  • NSFW helper_utils.py — same code
"""

import math
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

# Global plot style
PLOT_STYLE = {
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "font.family": "sans",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 3,
    "lines.markersize": 6,
}

mpl.rcParams.update(PLOT_STYLE)


def set_seed(seed=42):
    """Sets random seed for reproducibility across torch, numpy, python."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_training_metrics(metrics):
    """
    Plot training curves from training_loop() output.

    Args:
        metrics: [train_losses, val_losses, val_accuracies] —
                 three lists, one value per epoch.
    """
    train_losses, val_losses, val_accuracies = metrics
    num_epochs = len(train_losses)
    epochs = range(1, num_epochs + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Loss plot ---
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, color='#085c75', linewidth=2.5, marker='o', markersize=5, label='Training Loss')
    ax1.plot(epochs, val_losses, color='#fa5f64', linewidth=2.5, marker='o', markersize=5, label='Validation Loss')
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Accuracy plot ---
    ax2 = axes[1]
    ax2.plot(epochs, val_accuracies, color='#fa5f64', linewidth=2.5, marker='o', markersize=5, label='Validation Accuracy')
    ax2.set_title('Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Styling
    x_interval = max(1, (num_epochs - 1) // 10 + 1)
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=1, right=num_epochs)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(x_interval))
        ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.show()
