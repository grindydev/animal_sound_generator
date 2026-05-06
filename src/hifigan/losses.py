"""
losses.py — HiFi-GAN loss functions.

Three losses combined for the generator:

    1. Mel L1 Loss (λ=45)
       |mel(real_audio) - mel(generated_audio)|
       Forces the generated audio to have the right FREQUENCY content.
       Heavy weight because this is the primary objective.

    2. Feature Matching Loss (λ=2)
       Σ |D_k(real) - D_k(generated)| across all discriminator layers
       Forces the generator to match the discriminator's internal
       representation of "real" audio at every abstraction level.

    3. Hinge GAN Loss (λ=1)
       Generator:  -D(generated)
       Discriminator:  max(0, 1 - D(real)) + max(0, 1 + D(generated))
       Hinge loss is more stable than cross-entropy for audio GANs.

Reference: Kong et al. (2020) Appendix A (loss details)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
#  Mel L1 Loss
# ══════════════════════════════════════════════════════════════

class MelL1Loss(nn.Module):
    """
    L1 distance between mel spectrograms of real and generated audio.
    
    Computes mel on-the-fly (no pre-computed mels needed for comparison).
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 200,
        n_mels: int = 64,
        f_min: float = 0.0,
        f_max: float = 11025.0,
    ):
        super().__init__()
        from torchaudio.transforms import MelSpectrogram

        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=1,  # magnitude (not power)
            normalized=False,
        )

    def forward(self, fake_audio: torch.Tensor, real_audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fake_audio: [B, 1, T] generated waveform
            real_audio: [B, 1, T] ground truth waveform

        Returns:
            scalar L1 loss
        """
        fake_mel = self.mel_transform(fake_audio.squeeze(1))
        real_mel = self.mel_transform(real_audio.squeeze(1))
        return F.l1_loss(fake_mel, real_mel)


# ══════════════════════════════════════════════════════════════
#  Feature Matching Loss
# ══════════════════════════════════════════════════════════════

def feature_matching_loss(
    real_features: list,
    fake_features: list,
) -> torch.Tensor:
    """
    L1 distance between discriminator intermediate features.

    Args:
        real_features: list of lists — [disc_idx][layer_idx] → tensor
        fake_features: same structure

    Returns:
        scalar loss (mean over all layers)
    """
    loss = 0.0
    count = 0

    for real_group, fake_group in zip(real_features, fake_features):
        for r, f in zip(real_group, fake_group):
            loss += F.l1_loss(f, r.detach())
            count += 1

    return loss / max(count, 1)


# ══════════════════════════════════════════════════════════════
#  Generator Loss (all 3 combined)
# ══════════════════════════════════════════════════════════════

def generator_loss(
    fake_audio: torch.Tensor,
    real_audio: torch.Tensor,
    fake_scores: list,
    fake_features: list,
    real_features: list,
    mel_loss_fn: MelL1Loss,
    lambda_mel: float = 45.0,
    lambda_fm: float = 2.0,
    lambda_adv: float = 1.0,
) -> tuple:
    """
    Total generator loss = λ_mel × L_mel + λ_fm × L_fm + λ_adv × L_adv

    Returns:
        total_loss: scalar
        losses: dict with breakdown
    """
    # Mel L1
    loss_mel = mel_loss_fn(fake_audio, real_audio)

    # Feature matching
    loss_fm = feature_matching_loss(real_features, fake_features)

    # Adversarial (generator wants discriminator to say "real")
    loss_adv = 0.0
    count = 0
    for scores in fake_scores:
        # Hinge: -D(fake) — minimize negative score = maximize score
        loss_adv += -scores.mean()
        count += 1
    loss_adv = loss_adv / max(count, 1)

    total = lambda_mel * loss_mel + lambda_fm * loss_fm + lambda_adv * loss_adv

    return total, {
        "g_mel": loss_mel.item(),
        "g_fm": loss_fm.item(),
        "g_adv": loss_adv.item(),
        "g_total": total.item(),
    }


# ══════════════════════════════════════════════════════════════
#  Discriminator Loss
# ══════════════════════════════════════════════════════════════

def discriminator_loss(
    real_scores: list,
    fake_scores: list,
) -> tuple:
    """
    Hinge GAN loss for discriminator.

    L_D = mean(max(0, 1 - D(real))) + mean(max(0, 1 + D(fake)))

    Returns:
        total_loss: scalar
        losses: dict with breakdown
    """
    loss_real = 0.0
    loss_fake = 0.0
    count = 0

    for r_scores, f_scores in zip(real_scores, fake_scores):
        loss_real += F.relu(1.0 - r_scores).mean()
        loss_fake += F.relu(1.0 + f_scores).mean()
        count += 1

    loss_real = loss_real / max(count, 1)
    loss_fake = loss_fake / max(count, 1)

    total = loss_real + loss_fake

    return total, {
        "d_real": loss_real.item(),
        "d_fake": loss_fake.item(),
        "d_total": total.item(),
    }


# ══════════════════════════════════════════════════════════════
#  Quick test
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from .config import config

    # Mel L1 test
    mel_loss = MelL1Loss(n_fft=config.n_fft, hop_length=config.hop_length, n_mels=config.n_mels)
    real = torch.randn(2, 1, config.segment_size)
    fake = real + 0.1 * torch.randn(2, 1, config.segment_size)
    l_mel = mel_loss(fake, real)
    print(f"Mel L1 loss: {l_mel:.4f}")
    print(f"  (fake ≈ real + 0.1×noise, expect small positive)")

    # FM loss test
    r_feats = [[torch.randn(2, 64, 16)], [torch.randn(2, 128, 8)]]
    f_feats = [[r + 0.05 * torch.randn_like(r)] for (r,) in r_feats]
    l_fm = feature_matching_loss(r_feats, f_feats)
    print(f"FM loss: {l_fm:.4f}")

    # Hinge loss test
    r_scores = [torch.ones(2, 1, 4)]
    f_scores = [-torch.ones(2, 1, 4)]
    d_loss, d_dict = discriminator_loss(r_scores, f_scores)
    g_loss, g_dict = generator_loss(
        fake, real, f_scores, f_feats, r_feats,
        mel_loss, lambda_mel=45, lambda_fm=2, lambda_adv=1,
    )
    print(f"D hinge loss: {d_loss:.4f} (perfect: ~0)")
    print(f"G total loss: {g_loss:.4f}")
    print(f"  breakdown: mel={g_dict['g_mel']:.4f} fm={g_dict['g_fm']:.4f} adv={g_dict['g_adv']:.4f}")
