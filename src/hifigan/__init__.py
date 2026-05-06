"""
HiFi-GAN Neural Vocoder — Phase 7a

Converts mel spectrograms → natural-sounding audio waveforms.
Replaces the lossy GriffinLim mathematical approximation with
a trained neural network (GAN architecture).

Components:
    generator.py      — MRFGenerator (mel → waveform)
    discriminator.py  — MPD + MSD discriminators
    losses.py         — Mel L1, feature matching, hinge GAN
    train.py          — Training loop on (mel, audio) pairs
    inference.py      — mel_to_waveform() single-call API
    config.py         — All hyperparameters
    utils.py          — init_weights, padding, checkpoint I/O
"""
