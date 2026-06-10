# Pneumonia Detection using Deep Learning

Detects pneumonia from chest X-ray images using **ResNet-50** pretrained on ImageNet,
fine-tuned on the Kaggle Chest X-Ray dataset (5,863 labeled images).
Deployed as a FastAPI REST service with medical recommendations.

## Results

| Model | Accuracy | ROC AUC |
|-------|----------|---------|
| ResNet-50 (full fine-tune) | ~95% | ~97% |
| ResNet-18 (faster) | ~93% | ~95% |

## Tech Stack

- **Model:** ResNet-50 / ResNet-18 (torchvision pretrained)
- **Framework:** PyTorch
- **Dataset:** Chest X-Ray Images (Pneumonia) — Kaggle
- **Deployment:** FastAPI + Uvicorn
- **Evaluation:** ROC AUC, confusion matrix, training history

## Dataset

Download from Kaggle:
[https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)


## Project Structure
pneumonia_detection/
├── train.py              # ResNet transfer learning training
├── app.py                # FastAPI inference server
├── download_data.py      # Kaggle dataset downloader
├── requirements.txt
└── outputs/
├── best_model.pth
├── confusion_matrix.png
├── roc_curve.png
└── training_history.png

## Setup & Installation

```bash
git clone https://github.com/PavithraNagineni/pneumonia-detection
cd pneumonia-detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
# ResNet-50 full fine-tune
python train.py --epochs 10 --batch_size 32 --lr 1e-4 --model resnet50

# Freeze backbone — train only classifier head (faster)
python train.py --epochs 5 --freeze --model resnet50

# Lighter model
python train.py --model resnet18
```

## Inference API

```bash
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

Interactive docs: `http://localhost:8001/docs`

```bash
curl -X POST http://localhost:8001/predict \
  -F "file=@chest_xray.jpg"
```

### Response

```json
{
  "label": "PNEUMONIA",
  "confidence": 0.9712,
  "probabilities": {"NORMAL": 0.0288, "PNEUMONIA": 0.9712},
  "inference_time_ms": 23.1,
  "recommendation": "⚠️ Pneumonia detected. Please consult a medical professional immediately."
}
```

## Key Concepts Demonstrated

- Transfer Learning — ResNet-50 pretrained weights from ImageNet
- Class imbalance handling — weighted CrossEntropyLoss
- Data augmentation — horizontal flip, rotation, color jitter
- Medical imaging evaluation — ROC AUC as primary metric
- CosineAnnealingLR learning rate scheduling
- Production image upload API with FastAPI

## Output

<img width="1860" height="880" alt="Screenshot 2026-06-10 171829" src="https://github.com/user-attachments/assets/582a193f-da8c-4467-bd53-61c113142f85" />

<img width="1823" height="892" alt="Screenshot 2026-06-10 171844" src="https://github.com/user-attachments/assets/a894bf93-e774-457e-a2f1-3827236a7608" />


## Author
   Pavithra Nagineni
