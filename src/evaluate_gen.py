"""
evaluate_gen.py — Phase 5: VAE Generation Quality Evaluation
=============================================================

Compares two VAE models (from-scratch vs finetune) across 5 metrics:

  1. Reconstruction MSE     — How well does each reconstruct seen data?
  2. Classification Agreement — Does the Phase 2 classifier agree with generated class?
  3. Diversity Score         — Same class, different sound each time?
  4. t-SNE Visualization     — Do real vs generated encodings overlap?
  5. Spectrogram Comparison  — Side-by-side visual inspection

Uses the Phase 2 audio classifier (best_audio_cnn_train.pth) for evaluating
generation quality — exactly what the roadmap describes.
"""

import os
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from data_loader import get_dataloaders, get_transformations
from vae import SimpleAudioVAE
from model import SimpleAudioCNN

warnings.filterwarnings("ignore")


# ==================== CONFIG ====================
CONFIG = {
    "device": "auto",
    "batch_size": 16,
    "num_workers": 2,

    # Model paths
    "classifier_path": "models/best_audio_cnn_train.pth",
    "vae_scratch_path": "models/best_vae_scratch_train.pth",
    "vae_finetune_path": "models/best_vae_finetune_train.pth",

    # Evaluation settings
    "num_generated_per_class": 50,       # Samples for classification agreement
    "num_diversity_samples": 20,         # Samples for diversity metric
    "tsne_samples": 200,                 # Real + generated samples for t-SNE
    "spectrogram_classes": ["Dog", "Cat", "Rooster", "Frog"],  # Classes to visualize
    "output_dir": "evaluation_output",
    "beta": 0.005,                      # β used at validation time (full VAE)
}

CLASS_NAMES = ["Dog", "Cat", "Rooster", "Frog", "Crow", "Insect", "Hen", "Noise"]


# ==================== HELPERS ====================

def get_device():
    if CONFIG["device"] == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def load_vae(checkpoint_path, num_classes, device):
    """Load a trained VAE model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = SimpleAudioVAE(
        latent_dim=ckpt["latent_dim"],
        num_classes=num_classes,
        embed_dim=ckpt.get("embed_dim", 64),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    source = ckpt.get("type", "finetune")
    print(f"   ✅ Loaded VAE ({source}): latent={ckpt['latent_dim']}, "
          f"epoch={ckpt.get('epoch', '?')}, β={ckpt.get('beta', '?')}")
    return model


def load_classifier(checkpoint_path, num_classes, device):
    """Load the Phase 2 audio classifier for evaluation."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = SimpleAudioCNN(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    val_acc = ckpt.get('val_accuracy', '?')
    if isinstance(val_acc, (int, float)) and val_acc > 1:
        val_acc = val_acc / 100.0  # stored as percentage
    print(f"   ✅ Loaded classifier: epoch={ckpt.get('epoch', '?')}, "
          f"val_acc={val_acc:.1%}")
    return model


def reconstruction_mse(vae, loader, device, eval_transform, beta):
    """Compute MSE on the test set."""
    total_mse = 0.0
    count = 0
    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            specs = eval_transform(waveforms)
            recon, _, _ = vae(specs, labels)
            mse = F.mse_loss(recon, specs).item() * specs.size(0)
            total_mse += mse
            count += specs.size(0)
    return total_mse / count


def classification_agreement(vae, classifier, device, eval_transform, num_samples=50):
    """
    Generate sounds for each class, run through classifier.
    Returns % where classifier predicts the intended class.

    High agreement = generated sounds are recognizable by the classifier.
    """
    results = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        correct = 0
        for _ in range(num_samples):
            spec = vae.sample(label=class_idx, num_samples=1, device=device)
            with torch.no_grad():
                logits = classifier(spec)
                pred = logits.argmax(dim=1).item()
            if pred == class_idx:
                correct += 1
        results[class_name] = correct / num_samples
    results["Average"] = sum(results.values()) / len(results)
    return results


def diversity_score(vae, device, num_samples=20):
    """
    Generate multiple samples for each class, measure pairwise distance.
    High distance = diverse (good). Low distance = mode collapse (bad).

    We measure pairwise Euclidean distance in latent space by encoding
    the generated spectrograms, since generated samples don't have an encoding.
    Instead, use the pairwise distance between the spectrograms themselves
    (in pixel space) as a proxy for visual diversity.
    """
    results = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        specs = vae.sample(label=class_idx, num_samples=num_samples, device=device)
        # Flatten to [N, pixels]
        flat = specs.view(num_samples, -1)
        # Pairwise Euclidean distances
        dists = []
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                d = torch.norm(flat[i] - flat[j]).item()
                dists.append(d)
        results[class_name] = np.mean(dists)
    results["Average"] = sum(results.values()) / len(results)
    return results


def compute_tsne(vae, loader, classifier, device, eval_transform, n_samples=200):
    """
    Fit t-SNE on real + generated encodings.
    For real samples, use encoder μ. For generated, use random z + class embedding.

    Returns: tsne_coords (2D), labels (0=real, 1=generated), class_labels
    """
    all_encodings = []
    all_types = []       # 0=real, 1=generated
    all_classes = []

    # Collect real encodings from test set
    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            specs = eval_transform(waveforms)
            mu, _ = vae.encode_to_params(specs)
            # Concatenate class embedding to match 1088-dim latent
            class_emb = vae.class_embed(labels)
            mu_full = torch.cat([mu, class_emb], dim=1)
            all_encodings.append(mu_full.cpu().numpy())
            all_types.extend([0] * len(mu))
            all_classes.extend(labels.cpu().numpy())
            if len(all_types) >= n_samples // 2:
                break

    # Truncate to n_samples/2 for real
    all_encodings_real = np.concatenate(all_encodings, axis=0)[:n_samples // 2]
    all_types_real = all_types[:n_samples // 2]
    all_classes_real = all_classes[:n_samples // 2]

    # Generate samples (equal split across classes)
    gen_per_class = max(1, (n_samples // 2) // len(CLASS_NAMES))
    gen_encodings = []
    gen_classes = []
    for class_idx in range(len(CLASS_NAMES)):
        z = torch.randn(gen_per_class, vae.fc_mu.out_features, device=device)
        class_emb = vae.class_embed(
            torch.full((gen_per_class,), class_idx, dtype=torch.long, device=device)
        )
        # Concatenation: z (1024) + class_emb (64) = 1088-dim latent
        z_cond = torch.cat([z, class_emb], dim=1)
        gen_encodings.append(z_cond.detach().cpu().numpy())
        gen_classes.extend([class_idx] * gen_per_class)

    gen_encodings = np.concatenate(gen_encodings, axis=0)
    gen_types = [1] * len(gen_encodings)

    # Combine
    all_enc = np.concatenate([all_encodings_real, gen_encodings], axis=0)
    all_typ = all_types_real + gen_types
    all_cls = all_classes_real + gen_classes

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_enc) - 1))
    coords = tsne.fit_transform(all_enc)

    return coords, np.array(all_typ), np.array(all_cls)


def plot_tsne(coords, types, classes, model_name, output_dir):
    """Plot t-SNE: blue=real, orange=generated, different markers per class."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Real vs Generated
    ax = axes[0]
    real_mask = types == 0
    gen_mask = types == 1
    ax.scatter(coords[real_mask, 0], coords[real_mask, 1],
               c="steelblue", label="Real", alpha=0.6, s=15)
    ax.scatter(coords[gen_mask, 0], coords[gen_mask, 1],
               c="darkorange", label="Generated", alpha=0.6, s=15)
    ax.set_title(f"{model_name} — Real vs Generated", fontsize=13, weight="bold")
    ax.legend()
    ax.axis("off")

    # Right: Per-class
    colors = plt.cm.tab10(np.linspace(0, 1, len(CLASS_NAMES)))
    ax = axes[1]
    for i, name in enumerate(CLASS_NAMES):
        mask = classes == i
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colors[i]], label=name, alpha=0.6, s=15)
    ax.set_title(f"{model_name} — Per Class", fontsize=13, weight="bold")
    ax.legend(markerscale=2, fontsize=8, loc="upper right")
    ax.axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, f"tsne_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"   📊 t-SNE saved: {path}")


def plot_spectrogram_comparison(vae, loader, device, eval_transform, model_name, output_dir):
    """Side-by-side: real vs reconstructed vs generated for selected classes."""
    classes_to_show = CONFIG["spectrogram_classes"]
    n_classes = len(classes_to_show)
    class_indices = [CLASS_NAMES.index(c) for c in classes_to_show]

    # Collect one real sample per class
    real_specs = {}
    found = set()
    with torch.no_grad():
        for waveforms, labels in loader:
            for i in range(len(labels)):
                cls = labels[i].item()
                if cls in class_indices and cls not in found:
                    wf = waveforms[i].unsqueeze(0).to(device)
                    lb = torch.tensor([cls], device=device)
                    spec = eval_transform(wf)
                    real_specs[cls] = spec.squeeze().cpu().numpy()
                    found.add(cls)
                if len(found) == n_classes:
                    break
            if len(found) == n_classes:
                break

    fig, axes = plt.subplots(n_classes, 3, figsize=(15, 3 * n_classes))
    if n_classes == 1:
        axes = axes.reshape(1, -1)

    for row, (class_idx, class_name) in enumerate(zip(class_indices, classes_to_show)):
        # Real
        if class_idx in real_specs:
            axes[row, 0].imshow(real_specs[class_idx], aspect="auto",
                                origin="lower", cmap="magma")
        axes[row, 0].set_title(f"{class_name} — Real", fontsize=11)
        axes[row, 0].axis("off")

        # Reconstructed
        if class_idx in real_specs:
            spec_t = torch.tensor(real_specs[class_idx]).unsqueeze(0).unsqueeze(0).to(device)
            lb_t = torch.tensor([class_idx], device=device)
            with torch.no_grad():
                recon, _, _ = vae(spec_t, lb_t)
            axes[row, 1].imshow(recon.squeeze().cpu().numpy(), aspect="auto",
                                origin="lower", cmap="magma")
        axes[row, 1].set_title(f"{class_name} — Reconstructed", fontsize=11)
        axes[row, 1].axis("off")

        # Generated
        gen = vae.sample(label=class_idx, num_samples=1, device=device)
        axes[row, 2].imshow(gen.squeeze().cpu().numpy(), aspect="auto",
                            origin="lower", cmap="magma")
        axes[row, 2].set_title(f"{class_name} — Generated", fontsize=11)
        axes[row, 2].axis("off")

    plt.suptitle(f"{model_name} — Spectrogram Comparison", fontsize=14, weight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, f"spectrograms_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   🎨 Spectrograms saved: {path}")


# ==================== MAIN ====================

def evaluate_model(vae, classifier, test_loader, device, eval_transform, name):
    """Run all 5 evaluation metrics for one model."""
    print(f"\n{'='*60}")
    print(f"📋 Evaluating: {name}")
    print(f"{'='*60}")

    results = {}

    # ── 1. Reconstruction MSE ──
    mse = reconstruction_mse(vae, test_loader, device, eval_transform, CONFIG["beta"])
    results["mse"] = mse
    print(f"\n   1️⃣  Reconstruction MSE: {mse:.6f}")

    # ── 2. Classification Agreement ──
    print(f"\n   2️⃣  Classification Agreement (generating {CONFIG['num_generated_per_class']} per class)...")
    agreement = classification_agreement(
        vae, classifier, device, eval_transform,
        num_samples=CONFIG["num_generated_per_class"]
    )
    results["agreement"] = agreement
    for cls, acc in agreement.items():
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"      {cls:<8s}: {acc:.1%} {bar}")

    # ── 3. Diversity Score ──
    print(f"\n   3️⃣  Diversity Score (generating {CONFIG['num_diversity_samples']} per class)...")
    diversity = diversity_score(vae, device, num_samples=CONFIG["num_diversity_samples"])
    results["diversity"] = diversity
    print(f"      Average pairwise distance: {diversity['Average']:.2f}")
    for cls, score in diversity.items():
        if cls != "Average":
            print(f"      {cls:<8s}: {score:.2f}")

    # ── 4. t-SNE Visualization ──
    print(f"\n   4️⃣  t-SNE ({CONFIG['tsne_samples']} samples)...")
    coords, types, classes = compute_tsne(
        vae, test_loader, classifier, device, eval_transform,
        n_samples=CONFIG["tsne_samples"]
    )
    plot_tsne(coords, types, classes, name, CONFIG["output_dir"])

    # ── 5. Spectrogram Comparison ──
    print(f"\n   5️⃣  Spectrogram Comparison...")
    plot_spectrogram_comparison(vae, test_loader, device, eval_transform, name, CONFIG["output_dir"])

    return results


def print_comparison(scratch_results, finetune_results):
    """Print side-by-side comparison table."""
    print(f"\n\n{'='*70}")
    print(f"🏆 MODEL COMPARISON — Scratch vs Finetune")
    print(f"{'='*70}")

    print(f"\n   {'Metric':<35s} {'Scratch':>15s} {'Finetune':>15s}")
    print(f"   {'─'*35} {'─'*15} {'─'*15}")

    # MSE
    print(f"   {'Reconstruction MSE':<35s} {scratch_results['mse']:>15.6f} {finetune_results['mse']:>15.6f}")

    # Classification Agreement (average)
    s_avg = scratch_results["agreement"]["Average"]
    f_avg = finetune_results["agreement"]["Average"]
    print(f"   {'Classification Agreement':<35s} {s_avg:>14.1%} {f_avg:>14.1%}")

    # Diversity (average)
    s_div = scratch_results["diversity"]["Average"]
    f_div = finetune_results["diversity"]["Average"]
    print(f"   {'Diversity Score':<35s} {s_div:>15.2f} {f_div:>15.2f}")

    print(f"\n   {'─'*35} {'─'*15} {'─'*15}")

    # Per-class agreement
    print(f"\n   Per-class Classification Agreement:")
    for cls in CLASS_NAMES:
        s_a = scratch_results["agreement"][cls]
        f_a = finetune_results["agreement"][cls]
        print(f"   {cls:<8s} {'':>26s} {s_a:>14.1%} {f_a:>14.1%}")

    print(f"\n   Visualizations saved to: {CONFIG['output_dir']}/")


# ==================== RUN ====================
if __name__ == "__main__":
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    device = get_device()
    print(f"🔧 Device: {device}")

    # ── Load data ──
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        batch_size=CONFIG["batch_size"],
        train_fraction=0.6,
        val_fraction=0.2,
        num_workers=CONFIG["num_workers"],
    )
    _, eval_transform = get_transformations()
    eval_transform = eval_transform.to(device)
    print(f"✅ Data loaded: {len(test_loader.dataset)} test samples")

    # ── Load classifier ──
    print("\n📦 Loading models...")
    classifier = load_classifier(CONFIG["classifier_path"], num_classes, device)

    # ── Load VAE models ──
    vae_scratch = load_vae(CONFIG["vae_scratch_path"], num_classes, device)
    vae_finetune = load_vae(CONFIG["vae_finetune_path"], num_classes, device)

    # ── Evaluate both ──
    scratch_results = evaluate_model(
        vae_scratch, classifier, test_loader, device, eval_transform, "From-Scratch VAE"
    )
    finetune_results = evaluate_model(
        vae_finetune, classifier, test_loader, device, eval_transform, "Finetune VAE"
    )

    # ── Print comparison ──
    print_comparison(scratch_results, finetune_results)
