"""
quick_train.py — Lightweight model weight generator (no scipy/sklearn)
=======================================================================
Trains ResNet-18 on real or synthetic data and saves best_model.pth.
Run this if train.py fails due to scipy DLL errors on Windows.

Usage:
    python quick_train.py
"""

import os, json, time, copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms, datasets
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = "./outputs"
DATA_DIR   = "./data"
SEED       = 42
torch.manual_seed(SEED)


def build_model():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )
    return m


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Try loading real dataset first
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    try:
        train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), train_tf)
        val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   val_tf)
        logger.info(f"Real dataset loaded — Train: {len(train_ds)} Val: {len(val_ds)}")
        use_real = True
    except FileNotFoundError:
        logger.warning("Real dataset not found — using synthetic data for demonstration.")
        N = 32
        X = torch.randn(N, 3, 64, 64)   # tiny images to save RAM
        y = torch.randint(0, 2, (N,))
        train_ds = val_ds = TensorDataset(X, y)
        use_real = False

    # num_workers=0 to avoid Windows spawn issues; batch_size=4 for low RAM
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=0)

    model     = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    best_wts  = copy.deepcopy(model.state_dict())
    best_acc  = 0.0
    history   = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    EPOCHS    = 3  # fewer epochs for speed

    for epoch in range(EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{EPOCHS}")
        for phase, loader in [("train", train_loader), ("val", val_loader)]:
            model.train() if phase == "train" else model.eval()
            running_loss, running_correct, total = 0.0, 0, 0

            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                    preds   = outputs.argmax(dim=1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss    += loss.item() * inputs.size(0)
                running_correct += (preds == labels).sum().item()
                total           += inputs.size(0)

            epoch_loss = running_loss / total
            epoch_acc  = running_correct / total
            logger.info(f"  {phase.upper():5s} — Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            if phase == "train":
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc)
                scheduler.step()
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pth")
                    logger.info(f"  ✅ New best model saved (acc={best_acc:.4f})")

    model.load_state_dict(best_wts)

    # Save history
    with open(f"{OUTPUT_DIR}/history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save config
    config = {
        "model_name": "resnet18",
        "epochs": EPOCHS,
        "batch_size": 4,
        "lr": 1e-4,
        "device": str(device),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "synthetic_data": not use_real,
    }
    with open(f"{OUTPUT_DIR}/model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Simple accuracy on val set
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds   = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += inputs.size(0)
    val_accuracy = correct / total if total > 0 else 0.0
    logger.info(f"\nFinal Val Accuracy: {val_accuracy:.4f}")

    # Save test metrics
    with open(f"{OUTPUT_DIR}/test_metrics.json", "w") as f:
        json.dump({"test_accuracy": round(val_accuracy, 4), "roc_auc": None, "note": "Run train.py for full metrics"}, f, indent=2)

    logger.info(f"\n✅ Done. Model saved to {OUTPUT_DIR}/best_model.pth")
    logger.info(f"   Start the API: uvicorn app:app --host 0.0.0.0 --port 8001 --reload")


if __name__ == "__main__":
    main()
