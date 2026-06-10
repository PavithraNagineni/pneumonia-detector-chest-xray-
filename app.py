"""
FastAPI Inference Server — Pneumonia Detection
==============================================
Accepts chest X-ray images and returns pneumonia prediction.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8001 --reload
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io, os, time, json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "./outputs/pneumonia/best_model.pth")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
LABELS     = ["NORMAL", "PNEUMONIA"]

app = FastAPI(
    title="Pneumonia Detection API",
    description="ResNet-50 model for chest X-ray pneumonia detection",
    version="1.0.0",
)

model     = None
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model():
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(m.fc.in_features, 256),
        nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2),
    )
    return m


@app.on_event("startup")
async def load_model():
    global model
    try:
        model = build_model()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.warning(f"Model not found ({e}). Using untrained model for demo.")
        model = build_model().to(DEVICE)
        model.eval()


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict
    inference_time_ms: float
    recommendation: str


@app.get("/health")
def health():
    return {"status": "healthy", "device": DEVICE, "model": "ResNet-50"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image    = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor   = transform(image).unsqueeze(0).to(DEVICE)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    elapsed = (time.perf_counter() - start) * 1000

    pred_idx = int(np.argmax(probs))
    label    = LABELS[pred_idx]

    recommendation = (
        "⚠️ Pneumonia detected. Please consult a medical professional immediately."
        if label == "PNEUMONIA"
        else "✅ No signs of pneumonia detected. Regular check-ups recommended."
    )

    return PredictionResponse(
        label=label,
        confidence=float(probs[pred_idx]),
        probabilities={LABELS[i]: float(p) for i, p in enumerate(probs)},
        inference_time_ms=round(elapsed, 2),
        recommendation=recommendation,
    )
