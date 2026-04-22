"""
transfer_generator.py — Phase 7a: Transfer Learning for the Generator
========================================================================

RE-PRACTICE: Same technique as NSFW Phase 4 (ResNet18), now on audio.

WHAT YOU'LL BUILD:
  • Load pretrained PANNs encoder (trained on AudioSet, 2M clips)
  • Use it as the encoder part of your VAE (replaces your from-scratch encoder)
  • Same 3 strategies as NSFW:
    Strategy 1: Freeze pretrained encoder, train decoder only
    Strategy 2: Fine-tune last encoder layers + full decoder
    Strategy 3: Full retrain everything with small LR
  • Compare: did pretrained encoder improve generation quality?

KEY CONCEPTS:
  • PANNs = "ResNet18 for audio" — pretrained on massive audio dataset
  • Your decoder stays the same — it learns to decode from PANNs features
  • Watch for overfitting! Same lesson as NSFW: big model + small data = danger

COURSE REFERENCE:
  • NSFW transfer_cnn.py / transfer_cnn_finetune.py — freeze/fine-tune/retrain
  • L2-M2 transfer_learning/main.py — 3 strategies pattern
  • L2-M2 pre_processing — matching pretrained model's preprocessing

COMPARE WITH NSFW LESSARNED:
  NSFW: ResNet18 val=87%, test=79% (overfit!)
  Audio: Will PANNs also overfit on 600 clips? Same pattern?
"""
