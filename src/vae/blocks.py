"""
model_blocks.py — Improved Building Blocks for v2 Autoencoder & VAE.

Replaces SimpleEncoderBlock/SimpleDecoderBlock from model.py with:
  - ResEncoderBlock: residual downsampling with learnable skip connection
  - ResDecoderBlock: residual upsampling with FiLM conditioning + encoder skip concat
  - SelfAttention1D: temporal self-attention for bottleneck
  - FiLM: class-conditioned feature modulation (scale + shift)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
#  Residual Encoder Block (downsample with skip)
# ═══════════════════════════════════════════════════════════════

class ResEncoderBlock(nn.Module):
    """
    Residual downsampling block with GroupNorm + SiLU.
    
    Main path:  Conv2d(stride=2) → GN → SiLU → Conv2d → GN
    Skip path:  Conv2d(1×1, stride=2)
    Output:     SiLU(main + skip)
    
    Saves the output as a skip connection for the decoder.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
        )
        # 1x1 conv with stride=2 to match spatial size and channels
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.main(x) + self.skip(x))


# ═══════════════════════════════════════════════════════════════
#  FiLM — Feature-wise Linear Modulation
# ═══════════════════════════════════════════════════════════════

class FiLM(nn.Module):
    """
    Maps class embedding → per-channel scale (γ) and shift (β) parameters.
    
    Used to inject class information into every decoder block:
        h = h * (1 + γ) + β
    
    This gives the class a STRONG influence on every decoder layer, unlike
    the old approach of just concatenating class to z and hoping for the best.
    """
    def __init__(self, cond_dim: int, out_ch: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 2),
            nn.SiLU(),
            nn.Linear(cond_dim * 2, out_ch * 2),
        )

    def forward(self, cond: torch.Tensor, n_spatial_dims: int = 2) -> tuple:
        """
        Args:
            cond: conditioning vector [B, cond_dim] (class embedding)
            n_spatial_dims: 2 for 2D feature maps, 1 for 1D
        Returns:
            (gamma, beta): each [B, out_ch, 1, 1] or [B, out_ch, 1]
        """
        out = self.mlp(cond)  # [B, out_ch * 2]
        gamma, beta = out.chunk(2, dim=1)
        # Reshape for broadcasting
        for _ in range(n_spatial_dims):
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
        return gamma, beta


# ═══════════════════════════════════════════════════════════════
#  Residual Decoder Block (upsample with skip + FiLM)
# ═══════════════════════════════════════════════════════════════

class ResDecoderBlock(nn.Module):
    """
    Residual upsampling block with:
      - Nearest-neighbor 2× upsample
      - FiLM conditioning (class injection)
      - Encoder skip connection concatenation
      - Residual connection
    
    Flow:
      Input h [B, C_in, H, W]
        → Upsample 2× → Conv → GN → FiLM(γ,β) → SiLU → Conv → GN
        → + skip_conv(h_upsampled)
        → SiLU
        → Concat with encoder_skip
        → Conv to C_out
    
    Args:
        in_ch:       channels in from previous layer
        out_ch:      channels out (also channels of encoder skip to concat)
        cond_dim:    dimension of FiLM conditioning input (class embedding dim)
        skip_ch:     channels of encoder skip connection (should equal out_ch)
    """
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, skip_ch: int):
        super().__init__()
        self.out_ch = out_ch
        self.skip_ch = skip_ch

        # Main path after upsample
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.film = FiLM(cond_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(min(32, out_ch), out_ch)

        # Residual connection (after upsampling, before main path)
        self.skip_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        # Projection after concat with encoder skip
        concat_ch = out_ch + skip_ch
        self.proj = nn.Conv2d(concat_ch, out_ch, kernel_size=1)

    def forward(self, h: torch.Tensor, cond: torch.Tensor,
                enc_skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:        [B, in_ch, H, W] — input from previous decoder level
            cond:     [B, cond_dim] — FiLM conditioning (class embedding)
            enc_skip: [B, skip_ch, 2H, 2W] — skip from matching encoder level
        Returns:
            [B, out_ch, 2H, 2W]
        """
        # Upsample
        h = F.interpolate(h, scale_factor=2, mode='nearest')

        # Residual branch
        residual = self.skip_conv(h)

        # Main path
        h = self.conv1(h)
        h = self.gn1(h)
        gamma, beta = self.film(cond, n_spatial_dims=2)
        h = h * (1.0 + gamma) + beta
        h = F.silu(h)
        h = self.conv2(h)
        h = self.gn2(h)

        # Residual connection
        h = F.silu(h + residual)

        # Align encoder skip spatial size (may differ by 1 pixel)
        if h.shape[-2:] != enc_skip.shape[-2:]:
            enc_skip = F.interpolate(enc_skip, size=h.shape[-2:], mode='nearest')

        # Concat and project
        h = torch.cat([h, enc_skip], dim=1)
        h = self.proj(h)

        return h


# ═══════════════════════════════════════════════════════════════
#  Self-Attention for Bottleneck (1D along time axis)
# ═══════════════════════════════════════════════════════════════

class SelfAttention1D(nn.Module):
    """
    Multi-head self-attention applied along the temporal (width) dimension.
    
    For bottleneck features [B, C, H, W]:
      - Reshape to [B*H, W, C] — each frequency row attends across time
      - Apply multi-head attention
      - Reshape back to [B, C, H, W]
    
    This lets the model capture long-range temporal dependencies (e.g., 
    a bark that starts loud then fades, or a rhythmic pattern).
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # [B, C, H, W] → [B*H, W, C] — treat each frequency row independently across time
        x_flat = x.reshape(B * H, C, W).permute(0, 2, 1)  # [B*H, W, C]

        # QKV projection using Linear-like operation on last dim
        qkv_weight = self.qkv.weight.squeeze(-1).squeeze(-1)  # [3C, C]
        qkv_bias = self.qkv.bias  # [3C]
        qkv_out = F.linear(x_flat, qkv_weight, qkv_bias)  # [B*H, W, 3C]
        q, k, v = qkv_out.chunk(3, dim=-1)  # each [B*H, W, C]

        # Reshape for multi-head
        q = q.view(B * H, W, self.num_heads, self.head_dim).transpose(1, 2)  # [B*H, heads, W, head_dim]
        k = k.view(B * H, W, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B * H, W, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # [B*H, heads, W, W]
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # [B*H, heads, W, head_dim]

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B * H, W, C)  # [B*H, W, C]

        # Project
        proj_weight = self.proj.weight.squeeze(-1).squeeze(-1)  # [C, C]
        proj_bias = self.proj.bias  # [C]
        out = F.linear(out, proj_weight, proj_bias)  # [B*H, W, C]

        # Back to original shape
        out = out.permute(0, 2, 1).reshape(B, C, H, W)  # [B, C, H, W]

        return out
