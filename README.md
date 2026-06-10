# Pneumonia Detection using Deep Learning (ResNet Transfer Learning)

Detects pneumonia from chest X-ray images using ResNet-50 pretrained on ImageNet, fine-tuned on the Kaggle Chest X-Ray dataset (5,863 images).

## Results

| Model | Accuracy | ROC AUC | Notes |
|-------|----------|---------|-------|
| ResNet-50 | ~95% | ~97% | Full fine-tune |
| ResNet-18 | ~93% | ~95% | Faster training |

## Project Structure

```
pneumonia_detection/
├── train.py           # Training script with ResNet transfer learning
├── app.py             # FastAPI inference server
├── download_data.py   # Kaggle dataset downloader
├── requirements.txt
├── Dockerfile
└── outputs/
    ├── best_model.pth
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── training_history.png
```

## Setup

```bash
pip install -r requirements.txt
python download_data.py   # Requires Kaggle API key
```

## Training

```bash
# ResNet-50 full fine-tune
python train.py --epochs 10 --batch_size 32 --lr 1e-4 --model resnet50

# Freeze backbone, train only classifier head
python train.py --epochs 5 --freeze --model resnet50
```

## Inference API

```bash
uvicorn app:app --host 0.0.0.0 --port 8001
```

```bash
curl -X POST http://localhost:8001/predict \
  -F "file=@chest_xray.jpg"
```

## Key Concepts Demonstrated

- **Transfer Learning** — ResNet-50 pretrained on ImageNet
- **Class imbalance handling** — weighted CrossEntropyLoss
- **Data augmentation** — flips, rotations, color jitter
- **Medical imaging** — domain-specific model evaluation
- **ROC AUC** as primary metric (more meaningful than accuracy for medical tasks)
"# pneumonia-detector-chest-xray-" 
