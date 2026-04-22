"""
evaluate_gen.py — Phase 5: Evaluate Generated Audio Quality
=============================================================

WHAT YOU'LL BUILD:
  • Classification agreement: generate "dog" → does Phase 2 classifier agree?
  • Fréchet Audio Distance (FAD): distribution distance between real & generated
  • Diversity metric: pairwise distance of generated samples in latent space
  • t-SNE visualization: real vs generated clusters
  • Spectrogram comparison: real vs generated side by side

KEY CONCEPTS:
  • Generation evaluation is HARD — no single "correct" answer
  • Multiple metrics needed: quality (sounds real?) + diversity (different each time?)
  • Reuse Phase 2 classifier as an evaluation tool (meta-evaluation!)

EVALUATION STRATEGIES:
  1. Classification agreement:
     Generate "dog" → Phase 2 classifier → should predict "dog"
     Agreement rate = generation accuracy

  2. Turing test (real vs fake):
     Mix real + generated spectrograms → classifier tries to distinguish
     LOW accuracy = generated looks real (good!)
     HIGH accuracy = obvious artifacts (bad!)

  3. Diversity:
     Generate 10 "dog" sounds → pairwise latent distance
     HIGH = diverse (good), LOW = mode collapse (bad)

COURSE REFERENCE:
  • L2-M1 metrics — evaluation methodology
  • L3-M2 interpreting — visualization techniques
  • NSFW evaluate.py — same comparison pattern
"""
