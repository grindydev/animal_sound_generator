"""
utils.py — Shared utilities for HiFi-GAN.

Functions:
    • init_weights()     — Xavier normal init for Conv, normal for Linear
    • get_padding()      — padding so dilation-causal conv keeps same length
    • AttrDict           — dict that supports attribute access (config.a.b)
    • scan_checkpoints() — find latest epoch in checkpoint dir
"""
import os
import re
import json
import torch
import torch.nn as nn


def init_weights(module: nn.Module, mean: float = 0.0, std: float = 0.01):
    """Kaiming init for Conv1d/ConvTranspose1d — proper for LeakyReLU nets."""
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.kaiming_normal_(module.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """
    Calculate padding so that Conv1d output length matches input.
    
    For stride=1, same-length output needs:
        padding = (kernel_size - 1) * dilation // 2
    """
    return (kernel_size - 1) * dilation // 2


class AttrDict(dict):
    """Dictionary with attribute-style access (d.key instead of d['key'])."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def scan_checkpoints(checkpoint_dir: str, prefix: str = "generator_") -> int:
    """
    Find the highest epoch number in a checkpoint directory.
    
    Expects files named like: generator_000050.pth
    
    Returns:
        Latest epoch number, or 0 if no checkpoints found.
    """
    if not os.path.isdir(checkpoint_dir):
        return 0

    pattern = re.compile(rf"{re.escape(prefix)}(\d+)\.pth$")
    max_epoch = 0

    for fname in os.listdir(checkpoint_dir):
        match = pattern.search(fname)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch

    return max_epoch


def save_checkpoint(
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    epoch: int,
    checkpoint_dir: str,
    config: dict = None,
):
    """Save generator, discriminator, optimizers, and config."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "epoch": epoch,
            "config": config,
        },
        os.path.join(checkpoint_dir, f"checkpoint_{epoch:06d}.pth"),
    )

    # Also save standalone generator for inference
    torch.save(
        {"generator": generator.state_dict(), "config": config},
        os.path.join(checkpoint_dir, f"generator_{epoch:06d}.pth"),
    )


def load_checkpoint(
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    checkpoint_dir: str,
    device: torch.device,
):
    """Load the latest checkpoint. Returns starting epoch + 1."""
    latest_epoch = scan_checkpoints(checkpoint_dir, prefix="checkpoint_")
    if latest_epoch == 0:
        return 0

    path = os.path.join(checkpoint_dir, f"checkpoint_{latest_epoch:06d}.pth")
    ckpt = torch.load(path, map_location=device, weights_only=True)

    generator.load_state_dict(ckpt["generator"])
    discriminator.load_state_dict(ckpt["discriminator"])
    optimizer_g.load_state_dict(ckpt["optimizer_g"])
    optimizer_d.load_state_dict(ckpt["optimizer_d"])

    return ckpt["epoch"] + 1
