"""
train.py — Phase 2: Training Pipeline for Audio Classifier
============================================================

WHAT YOU'LL BUILD:
  • CONFIG dict — lr, weight_decay, batch_size, epochs (like NSFW main.py)
  • Training loop with progress bars (from helper_utils)
  • Early stopping + Cosine LR scheduler
  • MLflow logging: params, metrics, best model
  • Device-aware: CUDA / MPS / CPU auto-detect
  • Mixed precision (AMP) for faster training
  • Best model checkpointing

KEY CONCEPTS:
  • Same training pipeline as NSFW — only the data changes (spectrograms vs images)
  • The classifier trained here becomes the EVALUATOR in Phase 5
  • MLflow tracks every run — compare experiments

COURSE REFERENCE:
  • NSFW main.py — CONFIG dict, training loop pattern
  • L2-M1 scheduler/main.py — Cosine LR, early stopping
  • L3-M4 MLflow/main.py — mlflow.log_params(), log_metric(), log_artifact()

CONFIG (adjust these):
  lr = 1e-3
  weight_decay = 0.05
  batch_size = 32
  epochs = 30
  label_smoothing = 0.1
"""
