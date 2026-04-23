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


class NestedProgressBar:
    """
    Nested progress bars for training: outer bar = epochs, inner bar = batches.
    Same as NSFW project.

    Usage:
        pbar = NestedProgressBar(total_epochs=10, total_batches=50, mode="train")
        for epoch in range(1, 11):
            for batch in range(1, 51):
                pbar.update_batch(batch)
            pbar.update_epoch(epoch, postfix_dict={"loss": "0.5"})
        pbar.close()
    """
    def __init__(self, total_epochs, total_batches, mode="train"):
        from tqdm.auto import tqdm as tqdm_impl
        self.tqdm = tqdm_impl
        self.mode = mode
        self.total_epochs_raw = total_epochs
        self.total_batches_raw = total_batches

        if self.mode == "train":
            self.epoch_bar = self.tqdm(total=total_epochs, desc="Epoch", position=0, leave=True)
            self.batch_bar = self.tqdm(total=total_batches, desc="Batch", position=1, leave=False)
        elif self.mode == "eval":
            self.epoch_bar = None
            self.batch_bar = self.tqdm(total=total_batches, desc="Evaluating", position=0, leave=False)

        self._last_epoch = -1
        self._last_batch = -1

    def update_epoch(self, epoch, postfix_dict=None):
        step = epoch - 1  # 0-indexed
        if step != self._last_epoch:
            self.epoch_bar.update(1)
            self._last_epoch = step
        if self.mode == "train":
            self.epoch_bar.set_description(f"Epoch {epoch}/{self.total_epochs_raw}")
        if postfix_dict:
            self.epoch_bar.set_postfix(postfix_dict)
        # Reset batch bar for next epoch
        if self.batch_bar:
            self.batch_bar.reset()
            self._last_batch = -1

    def update_batch(self, batch, postfix_dict=None):
        step = batch - 1
        if step != self._last_batch:
            self.batch_bar.update(1)
            self._last_batch = step
        if self.mode == "train":
            self.batch_bar.set_description(f"  Batch {batch}/{self.total_batches_raw}")
        elif self.mode == "eval":
            self.batch_bar.set_description(f"  Eval Batch {batch}/{self.total_batches_raw}")
        if postfix_dict:
            self.batch_bar.set_postfix(postfix_dict)

    def close(self, last_message=None):
        if self.mode == "train" and self.epoch_bar:
            self.epoch_bar.close()
        if self.batch_bar:
            self.batch_bar.close()
        if last_message:
            print(last_message)


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


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Final Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    filename = 'data/confusion_matrix.png'
    plt.savefig(filename, dpi=200)
    plt.show()
    return filename