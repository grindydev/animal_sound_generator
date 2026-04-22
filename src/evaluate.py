"""
evaluate.py — Phase 2: Classifier Evaluation
==============================================

WHAT YOU'LL BUILD:
  • Load best model checkpoint, run on test set
  • Confusion matrix (sklearn) — which animal classes get confused?
  • Per-class precision/recall/F1 (classification_report)
  • Plot confusion matrix with class names
  • Print summary: best class, worst class, main confusion pairs

KEY CONCEPTS:
  • Same evaluation as NSFW evaluate.py — only data changes
  • This classifier will be REUSED in Phase 5 as the generation evaluator
  • "Does the generator's output sound like a dog?" → run through this classifier

COURSE REFERENCE:
  • NSFW evaluate.py — confusion matrix, classification_report
  • L3-M4 MLflow/main.py — logging evaluation metrics
"""
