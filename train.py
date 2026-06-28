"""
Pneumonia Detection using Deep Learning (ResNet Transfer Learning)
==================================================================
Uses ResNet-50 pretrained on ImageNet, fine-tuned on Chest X-Ray dataset.
Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Structure expected:
    data/
      train/  NORMAL/  PNEUMONIA/
      val/    NORMAL/  PNEUMONIA/
      test/   NORMAL/  PNEUMONIA/

Usage:
    python train.py --epochs 10 --batch_size 32 --lr 1e-4 --model resnet50
"""

import os
import argparse
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABELS     = ["NORMAL", "PNEUMONIA"]
DATA_DIR   = "./data"
OUTPUT_DIR = "./outputs"
SEED       = 42
torch.manual_seed(SEED)


# ── Data transforms ──────────────────────────────────────────────────────────
def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_transforms, val_transforms


# ── Model builder ─────────────────────────────────────────────────────────────
def build_model(model_name: str = "resnet50", freeze_backbone: bool = False):
    try:
        if model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        logger.info(f"Loaded {model_name} with ImageNet weights.")
    except Exception as e:
        logger.warning(f"Could not download pretrained weights ({e}). Using random init.")
        if model_name == "resnet50":
            model = models.resnet50(weights=None)
        elif model_name == "resnet18":
            model = models.resnet18(weights=None)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )
    return model


# ── Training loop ─────────────────────────────────────────────────────────────
def train_model(model, dataloaders, criterion, optimizer, scheduler,
                device, num_epochs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc       = 0.0
    history        = {"train_loss": [], "val_loss": [],
                      "train_acc":  [], "val_acc":  []}

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs} {'─'*40}")
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss, running_correct, total = 0.0, 0, 0

            for inputs, labels in dataloaders[phase]:
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

            if phase == "train":
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc)
                scheduler.step()
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc)
                # ── Save best checkpoint by accuracy (renamed from best_f1) ──
                if epoch_acc > best_acc:
                    best_acc       = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(),
                               os.path.join(output_dir, "best_model.pth"))
                    logger.info(f"  ✅ New best_acc={best_acc:.4f} — checkpoint saved")

            logger.info(f"  {phase.upper():5s} — Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

    model.load_state_dict(best_model_wts)
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"\nTraining complete. Best val acc: {best_acc:.4f}")
    return model, history


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, dataloader, device, output_dir):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs  = inputs.to(device)
            outputs = model(inputs)
            probs   = torch.softmax(outputs, dim=1).cpu().numpy()
            preds   = np.argmax(probs, axis=1)
            all_preds.extend(preds.tolist())
            all_probs.extend(probs[:, 1].tolist())
            all_labels.extend(labels.numpy().tolist())

    preds  = np.array(all_preds)
    probs  = np.array(all_probs)
    labels = np.array(all_labels)

    report  = classification_report(labels, preds, target_names=LABELS)
    roc_auc = roc_auc_score(labels, probs)
    logger.info(f"\nTest Classification Report:\n{report}")
    logger.info(f"ROC AUC Score: {roc_auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title("Confusion Matrix — Pneumonia Detection")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Pneumonia Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close()

    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump({"roc_auc": float(roc_auc),
                   "classification_report": report}, f, indent=2)
    return {"roc_auc": float(roc_auc), "classification_report": report}


# ── Plot history ──────────────────────────────────────────────────────────────
def plot_history(history, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   "r-s", label="Val Loss")
    ax1.set_title("Loss");  ax1.set_xlabel("Epoch")
    ax1.legend();           ax1.grid(True)

    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   "r-s", label="Val Acc")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch")
    ax2.legend();              ax2.grid(True)

    plt.suptitle("Training History — Pneumonia Detection")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=150)
    plt.close()
    logger.info(f"Training history plot saved.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_tf, val_tf = get_transforms()

    # Try real dataset, fall back to synthetic demo
    try:
        train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), train_tf)
        val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   val_tf)
        test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  val_tf)
        logger.info(f"Dataset — Train:{len(train_ds)} Val:{len(val_ds)} Test:{len(test_ds)}")
    except FileNotFoundError:
        logger.warning("Real dataset not found → using synthetic demo (64 random samples).")
        from torch.utils.data import TensorDataset
        N = 64
        X = torch.randn(N, 3, 224, 224)
        y = torch.randint(0, 2, (N,))
        train_ds = val_ds = test_ds = TensorDataset(X, y)

    # Class weights for imbalanced dataset
    if hasattr(train_ds, "targets"):
        class_counts  = np.bincount(train_ds.targets)
        class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
    else:
        class_weights = None

    dataloaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size,
                            shuffle=True,  num_workers=0),
        "val":   DataLoader(val_ds,   batch_size=args.batch_size,
                            shuffle=False, num_workers=0),
        "test":  DataLoader(test_ds,  batch_size=args.batch_size,
                            shuffle=False, num_workers=0),
    }

    model     = build_model(args.model, freeze_backbone=args.freeze).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model, history = train_model(
        model, dataloaders, criterion, optimizer, scheduler,
        device, args.epochs, OUTPUT_DIR,
    )
    plot_history(history, OUTPUT_DIR)
    evaluate(model, dataloaders["test"], device, OUTPUT_DIR)

    # ── Save model_config.json ────────────────────────────────────────────────
    config = {
        "model_name": args.model,
        "epochs":     args.epochs,
        "batch_size": args.batch_size,
        "lr":         args.lr,
        "device":     str(device),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(OUTPUT_DIR, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"model_config.json saved.")
    logger.info(f"✅ All outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--model",      type=str,   default="resnet50",
                        choices=["resnet50", "resnet18"])
    parser.add_argument("--freeze",     action="store_true",
                        help="Freeze backbone, train only classifier head")
    args = parser.parse_args()
    main(args)
