# 🫁 Pneumonia Detector using Chest X-ray

A deep learning-powered web application that detects **Pneumonia** from chest X-ray images using **ResNet Transfer Learning**. The application provides real-time predictions through a **FastAPI** backend and an interactive web interface.

---

## 📌 Features

- Upload chest X-ray images
- Detect Pneumonia using a trained ResNet model
- FastAPI REST API for inference
- User-friendly web interface
- Confidence score for predictions
- Model evaluation with multiple performance metrics
- Easy deployment using Render or Docker

---

## 🛠️ Tech Stack

### Frontend
- HTML
- CSS
- JavaScript

### Backend
- FastAPI
- Uvicorn

### Deep Learning
- PyTorch
- TorchVision
- ResNet Transfer Learning

### Data Processing
- NumPy
- Pillow
- OpenCV

### Deployment
- Render
- Docker
- GitHub

---

## 📂 Project Structure

```
.
├── app.py
├── train.py
├── quick_train.py
├── download_data.py
├── requirements.txt
├── Dockerfile
├── render.yaml
├── outputs/
│   ├── best_model.pth
│   ├── history.json
│   ├── metrics.json
│   ├── model_config.json
│   └── test_metrics.json
├── index.html
└── README.md
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/PavithraNagineni/pneumonia-detector-chest-xray-.git

cd pneumonia-detector-chest-xray-
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the FastAPI server

```bash
uvicorn app:app --reload
```

Open your browser

```
http://127.0.0.1:8000
```

## 🌐 Live Demo
 
---

## 📊 Model

The project uses **ResNet Transfer Learning** for binary image classification.

Classes:

- Normal
- Pneumonia

Training includes:

- Data Augmentation
- Early Stopping
- Model Checkpointing
- Learning Rate Scheduling

---

## 📈 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- Confusion Matrix

Training artifacts include:

- Training History
- ROC Curve
- Confusion Matrix
- Model Configuration
- Test Metrics

---

## 🌐 API Endpoint

### Predict

```
POST /predict
```

Upload an X-ray image and receive:

```json
{
  "prediction": "Pneumonia",
  "confidence": 98.63
}
```

---

## 📷 Sample Workflow

1. Upload Chest X-ray
2. Image preprocessing
3. Deep Learning prediction
4. Display diagnosis
5. Show confidence score

---

## 📸 Screenshots

Add screenshots here:

- Home Page
- Upload Interface
- Prediction Result
- ROC Curve
- Confusion Matrix

---

## 🧠 Future Improvements

- Multi-class disease detection
- Explainable AI using Grad-CAM
- Patient history integration
- Cloud storage support
- Mobile application
- Doctor dashboard
- PDF medical report generation

---

## 🎯 Applications

- Hospitals
- Diagnostic Centers
- Medical Research
- Healthcare AI
- Remote Healthcare
- Clinical Decision Support Systems

---

## 📚 Dataset

Chest X-ray Pneumonia Dataset

Source:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push the branch
5. Create a Pull Request

---

## Output

<img width="797" height="533" alt="Screenshot 2026-06-28 011027" src="https://github.com/user-attachments/assets/f2623195-4d7e-4ade-bee5-f396327f5491" />

<img width="772" height="868" alt="Screenshot 2026-06-28 011108" src="https://github.com/user-attachments/assets/055b8a1c-b5c8-470f-adab-1ab10434aed1" />


## 👩‍💻 Author

**Pavithra Nagineni**


---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.
