"""
server.py — Phase 6: FastAPI Server for Animal Sound Generator
================================================================

Endpoints:
    GET  /                     — Serve frontend
    POST /api/generate         — Generate animal sound, return .wav
    GET  /api/health           — Health check, model info
    GET  /api/models           — List available model checkpoints

Usage:
    python server.py
    # Then open http://localhost:8000
"""

import os
import sys
import time
import torch
import torchaudio
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vae import SimpleAudioVAE
from audio_utils import spectrogram_to_waveform, save_audio

# ─── Config ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
STATIC_DIR = Path(__file__).parent / "frontend"
OUTPUT_DIR = PROJECT_ROOT / "generated_audio"

# Model paths (updated after training completes)
MODEL_PATHS = {
    "finetune": MODELS_DIR / "best_vae_finetune_train.pth",
    "scratch": MODELS_DIR / "best_vae_scratch_train.pth",
}

CLASS_NAMES = ["Dog", "Cat", "Rooster", "Frog", "Crow", "Insect", "Hen", "Noise"]
NUM_CLASSES = len(CLASS_NAMES)

# Spectrogram params (must match training)
SAMPLE_RATE = 22050
N_FFT = 1024       # Must match MelSpectrogram n_fft
HOP_LENGTH = 200  # Must match MelSpectrogram hop_length
N_MELS = 64
LATENT_DIM = 1024
EMBED_DIM = 64

OUTPUT_DIR.mkdir(exist_ok=True)

# ─── FastAPI App ─────────────────────────────────────────────────────

app = FastAPI(title="🐾 Animal Sound Generator", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global model cache ──────────────────────────────────────────────

models = {}  # {"finetune": loaded_model, "scratch": loaded_model}
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def load_model(model_key: str) -> SimpleAudioVAE:
    """Load a VAE model from checkpoint. Cached after first load."""
    if model_key in models:
        return models[model_key]

    path = MODEL_PATHS.get(model_key)
    if not path or not path.exists():
        raise HTTPException(404, f"Model '{model_key}' not found at {path}")

    model = SimpleAudioVAE(
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
    ).to(device)

    ckpt = torch.load(str(path), map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)

    # Handle key mismatch (old checkpoint may have extra/missing keys)
    model_state = model.state_dict()
    # Only load keys that exist in the model
    filtered_state = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered_state, strict=False)

    model.eval()
    models[model_key] = model
    print(f"✅ Loaded {model_key} from {path.name} ({len(filtered_state)}/{len(model_state)} keys)")
    return model


# ─── Request/Response Models ─────────────────────────────────────────

class GenerateRequest(BaseModel):
    class_name: str = "Dog"         # Animal class name
    temperature: float = 0.7        # Sampling temperature (0.5=consistent, 1.5=wild)
    model: str = "finetune"         # "finetune" or "scratch"
    num_samples: int = 1            # Number of sounds to generate (max 5)
    seed: int | None = None        # Random seed (None = random each time)


class GenerateResponse(BaseModel):
    success: bool
    model: str
    class_name: str
    temperature: float
    duration_seconds: float
    generation_time_ms: float


# ─── API Endpoints ───────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Return server status and available models."""
    available = {k: str(v) for k, v in MODEL_PATHS.items() if v.exists()}
    loaded = list(models.keys())
    return {
        "status": "ok",
        "device": str(device),
        "available_models": available,
        "loaded_models": loaded,
        "class_names": CLASS_NAMES,
    }


@app.get("/api/models")
async def list_models():
    """List available model checkpoints."""
    result = {}
    for key, path in MODEL_PATHS.items():
        if path.exists():
            ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
            result[key] = {
                "path": str(path),
                "epoch": ckpt.get("epoch", "?"),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
            }
    return result


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    """Generate an animal sound and return .wav file."""
    if req.class_name not in CLASS_NAMES:
        raise HTTPException(400, f"Unknown class '{req.class_name}'. Options: {CLASS_NAMES}")

    if req.model not in MODEL_PATHS:
        raise HTTPException(400, f"Unknown model '{req.model}'. Options: {list(MODEL_PATHS.keys())}")

    label = CLASS_NAMES.index(req.class_name)

    # Set seed if provided
    if req.seed is not None:
        torch.manual_seed(req.seed)

    try:
        model = load_model(req.model)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {e}")

    # Generate
    t0 = time.time()
    with torch.no_grad():
        # Override temperature via register_buffer trick — we pass it to sample
        spectrograms = model.sample(
            label=label,
            num_samples=req.num_samples,
            device=device,
            temperature=req.temperature,
        )  # [num_samples, 1, n_mels, time_frames]

    # Convert to waveform
    spec = spectrograms[0]  # Take first sample
    waveform = spectrogram_to_waveform(
        spec,
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )

    gen_time = (time.time() - t0) * 1000
    duration = waveform.shape[-1] / SAMPLE_RATE

    # Save to temp file (torchcodec doesn't support BytesIO)
    safe_class = req.class_name.lower()
    timestamp = int(time.time())
    filepath = OUTPUT_DIR / f"{safe_class}_{timestamp}.wav"
    torchaudio.save(str(filepath), waveform.cpu(), SAMPLE_RATE)

    # Read file into buffer for response
    with open(filepath, "rb") as f:
        wav_bytes = f.read()

    # Return .wav file
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Generation-Time-Ms": f"{gen_time:.1f}",
            "X-Duration-Seconds": f"{duration:.2f}",
            "X-Model": req.model,
            "X-Class": req.class_name,
        },
    )


# ─── Static Frontend ─────────────────────────────────────────────────

@app.get("/")
async def index():
    """Serve the frontend HTML."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>Frontend not found</h1>")


# ─── Run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print(f"\n🐾 Animal Sound Generator Server")
    print(f"   Device: {device}")
    print(f"   Models: {[k for k, v in MODEL_PATHS.items() if v.exists()]}")
    print(f"   Classes: {CLASS_NAMES}")
    print(f"   Open: http://localhost:8000\n")

    # Pre-load models at startup
    for key, path in MODEL_PATHS.items():
        if path.exists():
            try:
                load_model(key)
            except Exception as e:
                print(f"   ⚠️  Could not pre-load {key}: {e}")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
