"""
audio_utils.py — Spectrogram → Audio Conversion
===============================================

The VAE generates mel spectrograms in dB scale (same format as the training data).
To hear them, we need to reverse the training transforms:

    MelSpectrogram + AmplitudeToDB + Normalize  →  waveform

Reverse pipeline:
    1. Unnormalize        → raw dB mel spec
    2. DB_to_power        → power mel spec
    3. InverseMelScale    → linear spec
    4. GriffinLim         → waveform
"""

import torch
import torchaudio.functional as F
import torchaudio.transforms as T


def spectrogram_to_waveform(
    spec: torch.Tensor,
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 200,
    n_mels: int = 64,
    top_db: float = 80.0,
    n_iter: int = 200,
) -> torch.Tensor:
    """
    Convert a mel spectrogram (dB scale) back to an audio waveform.

    Args:
        spec:          Generated spectrogram [1, n_mels, time_frames] or [n_mels, time_frames]
        sample_rate:   Audio sample rate (must match training: 44100)
        n_fft:         FFT window size (must match training MelSpectrogram: 1024)
        hop_length:    Hop length between frames (must match training MelSpectrogram default: 200)
        n_mels:        Number of mel bands (must match training: 64)
        top_db:        Top dB for AmplitudeToDB (must match training: 80)
        n_iter:        Griffin-Lim iterations (more = better quality, slower)

    Returns:
        waveform [1, num_samples] — playable audio tensor in [-1, 1]
    """
    if spec.dim() == 4:
        spec = spec.squeeze(0).squeeze(0)  # [1, 1, n_mels, T] → [n_mels, T]
    elif spec.dim() == 3:
        spec = spec.squeeze(0)  # [1, n_mels, T] → [n_mels, T]

    # Step 0: UNNORMALIZE — reverse SimpleNormalize(mean=-30.86, std=21.20)
    #    Training:  raw_dB → (x - (-30.86)) / 21.20  = normalized ~N(0,1)
    #    Inference: normalized → normalized * 21.20 + (-30.86) = raw_dB
    NORM_MEAN = -18.4903
    NORM_STD = 19.8031
    spec = spec * NORM_STD + NORM_MEAN
    # Clamp to valid dB range (prevents extreme values from ruining audio)
    spec = torch.clamp(spec, min=-80.0, max=0.0)

    # Step 1: Undo AmplitudeToDB — dB → power
    #   AmplitudeToDB did:   x_db = 10 * log10(x + eps)  with top_db clamping
    #   We reverse:           x = 10^(x_db / 10)
    spec_power = F.DB_to_amplitude(spec, ref=1.0, power=2.0)

    # Step 2: Undo Mel scale — mel → linear
    #    InverseMelScale is stable with n_fft=1024 (513 freq bins → all 64 mels covered).
    #    We compute the pseudo-inverse manually using SVD (pinv), then
    #    multiply: linear = mel_basis^+ @ mel_spec
    spec_device = spec_power.device
    spec_cpu = spec_power.cpu()  # [n_mels, T]
    n_stft = n_fft // 2 + 1
    # Build the same mel filterbank as torchaudio's MelScale
    import torchaudio.functional as AF
    fb = AF.melscale_fbanks(
        n_freqs=n_stft,
        f_min=0.0,
        f_max=sample_rate / 2.0,
        n_mels=n_mels,
        sample_rate=sample_rate,
    )  # [n_mels, n_stft]
    # melscale_fbanks returns [n_freqs, n_mels]. Some mel bins may be
    # all-zero if n_fft is too small. n_fft=1024 gives 513 freq bins, plenty for 64 mels.
    # Prune those to avoid singular matrix.
    fb_t = fb.T  # [n_mels, n_stft]
    valid_mels = fb_t.sum(dim=1) > 0  # [n_mels] — which mel bins have energy
    fb_valid = fb_t[valid_mels]       # [n_valid_mels, n_stft]
    spec_valid = spec_cpu[valid_mels]  # [n_valid_mels, T]
    
    # Solve fb_valid @ X = spec_valid  →  X = linear [n_stft, T]
    # X contains POWER values (MelSpectrogram uses power=2 internally).
    # GriffinLim expects MAGNITUDE, so take sqrt. Clamp small negatives first.
    result = torch.linalg.lstsq(fb_valid, spec_valid, driver='gelsd')
    linear_power = result.solution.to(spec_device).contiguous()  # [n_stft, T]
    linear_power = torch.clamp(linear_power, min=0.0)  # no negative power
    linear_spec = torch.sqrt(linear_power)  # power → magnitude

    # Step 3: Griffin-Lim — linear spec (magnitude) → waveform
    #    power=1: we pass MAGNITUDES (already sqrt'd above)
    griffin_lim = T.GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=n_iter,
        power=1,
        window_fn=torch.hann_window,
        momentum=0.99,
    )
    waveform = griffin_lim(linear_spec.cpu()).to(spec_device)  # [1, num_samples]

    # Ensure waveform is valid (NaN can come from rank-deficient filterbank)
    waveform = torch.nan_to_num(waveform, nan=0.0)
    waveform = torch.clamp(waveform, min=-1.0, max=1.0)

    return waveform


def save_audio(
    waveform: torch.Tensor,
    path: str,
    sample_rate: int = 22050,
) -> None:
    """Save a waveform tensor as a .wav file."""
    import torchaudio
    # Ensure proper shape: [channels, samples]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(path, waveform.cpu(), sample_rate)
