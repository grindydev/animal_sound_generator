"""
sequential_generator.py — Phase 6b: Generate Longer & Sequential Sounds
=========================================================================

WHAT YOU'LL BUILD:
  • Overlap-add stitching: crossfade between generated 2-sec chunks
  • Autoregressive approach: predict next audio chunk from previous
  • Sequence planning: "dog bark → pause → cat meow → cow moo"
  • Temperature control: higher = more random, lower = consistent

KEY CONCEPTS:
  • Your VAE generates 2-second clips — how to make longer sounds?
  • Overlap-add (simple): generate chunks with overlap, crossfade
  • Autoregressive (advanced): like Shakespeare text generation but for audio
  • Causal masking: can only attend to past chunks (not future)

COURSE REFERENCE:
  • L3-M3 decoder_block/main.py — autoregressive generation (Shakespeare)
  • L3-M3 translation/main.py — sequence-to-sequence (English → French)
  • L3-M3 decoder_block — causal masking, temperature

APPROACHES (try all, compare):
  A. Overlap-add (simplest, works well)
     chunk1[0-2s] + chunk2[1.5-3.5s] → crossfade → 3.5s audio

  B. Autoregressive (better quality)
     previous_frames → model → next_frame → append → repeat

  C. Sequence planner (scene composition)
     Input: [dog, pause, cat] → generate each → stitch → scene
"""
