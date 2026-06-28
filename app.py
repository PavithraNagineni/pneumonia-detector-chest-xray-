"""
FastAPI Inference Server — Pneumonia Detection
===============================================
Run: uvicorn app:app --host 0.0.0.0 --port 8001 --reload

Open browser: http://localhost:8001
API docs:     http://localhost:8001/docs
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io, os, time, json, urllib.request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = "./outputs/model_config.json"
MODEL_DIR   = "./outputs"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
LABELS      = ["NORMAL", "PNEUMONIA"]

app = FastAPI(
    title="Pneumonia Detection API",
    description="ResNet-50/18 deep learning model for chest X-ray pneumonia detection",
    version="2.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static folders ────────────────────────────────────────────────────────────
os.makedirs("./outputs", exist_ok=True)
os.makedirs("./samples", exist_ok=True)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/samples", StaticFiles(directory="samples"),  name="samples")

# ── State ─────────────────────────────────────────────────────────────────────
model           = None
model_meta      = {"name": "resnet50", "trained": False, "path": None}
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Model builder ─────────────────────────────────────────────────────────────
def build_model(model_name: str = "resnet50") -> nn.Module:
    if model_name == "resnet50":
        m = models.resnet50(weights=None)
    elif model_name == "resnet18":
        m = models.resnet18(weights=None)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    in_features = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )
    return m


def _resolve_model_path(model_name: str) -> str | None:
    """Return best available .pth path for a given model name, or None."""
    candidates = [
        os.path.join(MODEL_DIR, "best_model.pth"),
        os.path.join(MODEL_DIR, f"best_{model_name}.pth"),
        os.path.join("outputs", "best_model.pth"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _ensure_model_weights() -> None:
    """Download model weights when MODEL_WEIGHTS_URL is set and no checkpoint exists."""
    if _resolve_model_path("resnet50") or _resolve_model_path("resnet18"):
        return

    url = os.environ.get("MODEL_WEIGHTS_URL")
    if not url:
        return

    dest = os.path.join(MODEL_DIR, "best_model.pth")
    logger.info(f"Downloading model weights from {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        logger.info(f"Model weights saved to {dest}")
    except Exception as e:
        logger.error(f"Failed to download model weights ({e})")


# ── Startup: dynamic model loading ────────────────────────────────────────────
@app.on_event("startup")
async def load_model():
    global model, model_meta

    _ensure_model_weights()

    # 1. Read architecture from config if available
    model_name = "resnet50"
    config_data = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                config_data = json.load(f)
            model_name = config_data.get("model_name", "resnet50")
            logger.info(f"Config found → architecture: {model_name}")
        except Exception as e:
            logger.warning(f"Could not parse config ({e}), defaulting to resnet50")

    # 2. Resolve checkpoint path dynamically
    model_path = _resolve_model_path(model_name)

    # 3. Build model
    m = build_model(model_name).to(DEVICE)

    if model_path:
        try:
            state = torch.load(model_path, map_location=DEVICE)
            m.load_state_dict(state)
            m.eval()
            model_meta = {
                "name":       model_name,
                "trained":    True,
                "path":       model_path,
                "epochs":     config_data.get("epochs"),
                "lr":         config_data.get("lr"),
                "trained_at": config_data.get("trained_at"),
            }
            logger.info(f"✅ Trained {model_name} loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load weights ({e}). Running with random weights.")
            m.eval()
            model_meta = {"name": model_name, "trained": False, "path": None}
    else:
        m.eval()
        model_meta = {"name": model_name, "trained": False, "path": None}
        logger.warning("⚠️  No checkpoint found — using untrained weights. Run train.py first.")

    model = m


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    label:             str
    confidence:        float
    probabilities:     dict
    inference_time_ms: float
    recommendation:    str
    model_trained:     bool


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def serve_frontend():
    """Serve the dashboard UI."""
    return FileResponse("index.html")


@app.get("/health")
def health():
    """Server + model status."""
    display_name = "ResNet-50" if model_meta["name"] == "resnet50" else "ResNet-18"
    return {
        "status":        "healthy",
        "device":        DEVICE,
        "model":         display_name,
        "model_trained": model_meta["trained"],
        "model_info": {
            "model":      display_name,
            "path":       model_meta.get("path"),
            "epochs":     model_meta.get("epochs"),
            "lr":         model_meta.get("lr"),
            "trained_at": model_meta.get("trained_at"),
        },
    }


@app.get("/metrics")
def get_metrics():
    """Return saved test metrics (ROC AUC, history) for the dashboard."""
    result = {}

    metrics_path = os.path.join(MODEL_DIR, "test_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            result["test_metrics"] = json.load(f)

    history_path = os.path.join(MODEL_DIR, "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            result["history"] = json.load(f)

    plots = {}
    for name in ("confusion_matrix.png", "roc_curve.png", "training_history.png"):
        p = os.path.join(MODEL_DIR, name)
        if os.path.exists(p):
            plots[name] = f"/outputs/{name}"
    result["plots"] = plots

    if not result:
        return JSONResponse(status_code=404, content={"detail": "No metrics found. Run train.py first."})
    return result


@app.get("/api/samples")
def list_samples():
    """List available sample X-ray images in /samples folder."""
    samples_dir = "./samples"
    files = []
    if os.path.isdir(samples_dir):
        for fn in sorted(os.listdir(samples_dir)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                files.append({"name": fn, "url": f"/samples/{fn}"})
    return {"samples": files, "count": len(files)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file (jpg, png, etc.)")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file.")

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    pred_idx = int(np.argmax(probs))
    label    = LABELS[pred_idx]

    recommendation = (
        "⚠️ Pneumonia indicators detected. Please consult a medical professional immediately for a clinical diagnosis."
        if label == "PNEUMONIA"
        else "✅ No signs of pneumonia detected. Continue regular health check-ups as advised by your doctor."
    )

    return PredictionResponse(
        label=label,
        confidence=float(probs[pred_idx]),
        probabilities={LABELS[i]: float(p) for i, p in enumerate(probs)},
        inference_time_ms=elapsed_ms,
        recommendation=recommendation,
        model_trained=model_meta["trained"],
    )
