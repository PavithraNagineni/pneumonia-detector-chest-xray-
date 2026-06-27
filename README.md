# 🫁 Pneumonia Detection — Deep Learning (ResNet-50)

Detects pneumonia from chest X-ray images using ResNet-50 transfer learning.
Includes a drag-and-drop web UI and FastAPI REST backend.

## Results (after training on real dataset)
- Accuracy: ~95% | ROC AUC: ~97%

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset (Kaggle)
python download_data.py

# 3. Train model
python train.py --epochs 10 --batch_size 32

# 4. Start API + UI
uvicorn app:app --host 0.0.0.0 --port 8001 --reload

# 5. Open browser → http://localhost:8001
```

## Tech Stack
Python · PyTorch · ResNet-50 · FastAPI · Uvicorn · scikit-learn · HTML/CSS/JS
