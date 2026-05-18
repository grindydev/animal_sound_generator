"""
GAN Discriminator — Spectral norm + auxiliary classifier

Input:  mel spectrogram [B, 1, 64, 552] (padded to 576)
Output: real/fake logit [B, 1], class logits [B, num_classes]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscBlock(nn.Module):
    """Conv → LeakyReLU → Conv → LeakyReLU + optional downsample"""
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        )
        self.act = nn.LeakyReLU(0.2)
        self.downsample = nn.AvgPool2d(2) if downsample else nn.Identity()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.downsample(x)
        return x


class Discriminator(nn.Module):
    """Projection discriminator with auxiliary classifier.

    Architecture:
        mel [1, 64, 576]
          → DiscBlock (64→128, down)   → [128, 32, 288]
          → DiscBlock (128→256, down)  → [256, 16, 144]
          → DiscBlock (256→512, down)  → [512, 8, 72]
          → DiscBlock (512→512, down)  → [512, 4, 36]
          → DiscBlock (512→512, no ds) → [512, 4, 36]
          → Conv (512→512, 4×4)        → [512, 1, 33]
          → Flatten
          ├→ FC → 1 (real/fake)
          └→ Class embed · features → 1 + FC → 7 (class prediction)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes

        base_ch = config.disc_base_ch
        max_ch = config.disc_max_ch

        chs = [
            min(base_ch * 2, max_ch),
            min(base_ch * 4, max_ch),
            min(base_ch * 8, max_ch),
            min(base_ch * 8, max_ch),
            min(base_ch * 8, max_ch),
        ]
        in_ch = 1
        self.blocks = nn.ModuleList()
        for i, out_ch in enumerate(chs):
            downsample = (i < 4)  # downsample first 4 blocks
            self.blocks.append(DiscBlock(in_ch, out_ch, downsample=downsample))
            in_ch = out_ch

        # Final conv squeeze
        self.final_conv = nn.utils.spectral_norm(
            nn.Conv2d(in_ch, in_ch, kernel_size=4, padding=0)
        )

        # Real/fake head
        self.adv_head = nn.utils.spectral_norm(
            nn.Linear(in_ch, 1)
        )

        # Classifier head (auxiliary)
        self.class_head = nn.utils.spectral_norm(
            nn.Linear(in_ch, config.num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: mel [B, 1, 64, 576]
        Returns:
            adv_logits: [B, 1] — real/fake
            class_logits: [B, num_classes] — class prediction
        """
        for block in self.blocks:
            x = block(x)

        x = self.final_conv(x)              # [B, ch, 1, W]
        x = x.mean(dim=(2, 3))              # [B, ch] — global average pool

        adv_logits = self.adv_head(x)       # [B, 1]
        class_logits = self.class_head(x)   # [B, num_classes]

        return adv_logits, class_logits


def compute_r1_penalty(discriminator, real_images, real_labels, scaler=None):
    """R1 gradient penalty: γ/2 * E[‖∇D(real)‖²]

    Args:
        discriminator: Discriminator model
        real_images: [B, 1, H, W]
        real_labels: [B]
        scaler: GradScaler (for AMP)
    Returns:
        r1: scalar penalty
    """
    real_images = real_images.clone().detach().requires_grad_(True)

    adv_logits, _ = discriminator(real_images)

    # Handle AMP: unscaled loss for gradient computation
    grad = torch.autograd.grad(
        outputs=adv_logits.sum(),
        inputs=real_images,
        create_graph=True,
        retain_graph=True,
    )[0]

    r1 = grad.pow(2).view(real_images.shape[0], -1).sum(1).mean()
    return r1
