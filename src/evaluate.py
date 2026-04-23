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

Usage:
    python src/evaluate.py
    python src/evaluate.py --model models/best_audio_cnn_train.pth
"""

import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

from model import SimpleAudioCNN
from data_loader import get_dataloaders, get_transformations
from helper_utils import plot_confusion_matrix

CLASS_NAMES = ['Dog', 'Cat', 'Rooster', 'Frog', 'Crow', 'Insect', 'Hen', 'Noise']


def evaluate(model_path=None):
    # ==================== PATH TO YOUR DOWNLOADED MODEL ====================
    if model_path is None:
        model_path = Path(__file__).resolve().parent.parent / 'models' / 'best_audio_cnn_train.pth'

    # ==================== LOAD CHECKPOINT ====================
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)

    print(f"✅ Model loaded from: {model_path.name}")
    print(f"   Classes      : {checkpoint['num_classes']}")
    print(f"   Val Accuracy : {checkpoint['val_accuracy']:.2f}%")
    print(f"   Best epoch   : {checkpoint['epoch']}")

    # ==================== RECREATE MODEL & LOAD WEIGHTS ====================
    num_classes = checkpoint['num_classes']
    model = SimpleAudioCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("   Model in eval mode (ready for inference)")

    # ==================== DEVICE ====================
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model.to(device)
    print(f"   Device: {device}")

    # ==================== TEST SET ====================
    _, _, test_loader, _ = get_dataloaders(
        batch_size=16,
        train_fraction=0.6,
        val_fraction=0.2,
        num_workers=0,  # eval doesn't need multiprocessing
    )

    _, eval_transform = get_transformations()
    eval_transform = eval_transform.to(device)

    # ==================== RUN EVALUATION ====================
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for waveforms, labels in test_loader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            waveforms = eval_transform(waveforms)

            outputs = model(waveforms)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    total = len(all_labels)
    test_accuracy = 100.0 * correct / total

    print(f"\n📊 Test Accuracy: {test_accuracy:.2f}% ({correct}/{total})")

    # ==================== CLASSIFICATION REPORT ====================
    print(f"\n{'='*60}")
    print("Classification Report")
    print(f"{'='*60}")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # ==================== CONFUSION MATRIX ====================
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, CLASS_NAMES)

    return test_accuracy, all_preds, all_labels


if __name__ == '__main__':
    evaluate()
