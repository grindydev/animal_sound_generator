"""
src/vae/__init__.py — VAE Package

When imported from scripts that have src/ in sys.path, use relative imports.
When used as a regular package, absolute imports work.
"""
from vae.autoencoder import ImprovedAutoencoder, DecoderStage
from vae.model import ImprovedVAE, FiLMDecoderStage
from vae.blocks import ResEncoderBlock, ResDecoderBlock, SelfAttention1D, FiLM
