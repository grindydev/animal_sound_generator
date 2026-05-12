"""
src/vae/__init__.py — VAE Package

Exports the improved autoencoder and VAE models.
"""
from .autoencoder import ImprovedAutoencoder, DecoderStage
from .model import ImprovedVAE, FiLMDecoderStage
from .blocks import ResEncoderBlock, ResDecoderBlock, SelfAttention1D, FiLM
