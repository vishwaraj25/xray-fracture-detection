# BoneAI — X-Ray Fracture Detection

Classical computer vision pipeline for bone fracture detection from X-ray images.
Full web UI with live pipeline animation. Runs locally, no GPU needed.

## Pipeline

```
X-ray → CLAHE → Gaussian Blur → Canny Edges
      → HOG + LBP + Hu Moments → StandardScaler → SVM (RBF)
      → If FRACTURED: contour filtering → overlay on original
      → Web UI result with confidence + metrics
```

## Dataset

**Bone Fracture Multi-Region X-ray Data** (~10,500 images)
https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data

## Setup & Run

```bash
# 1. Clone
git clone https://github.com/vishwaraj25/xray-fracture-detection.git
cd xray-fracture-detection

# 2. Virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Download dataset → place at data/fracture_dataset/

# 4. Build feature dataset (~20-30 min for 10k images)
cd src
python build_dataset.py

# 5. Train model (~5-10 min)
python train_svm.py

# 6. Launch web UI
python app.py
# Open http://localhost:5000
```

## Folder structure

```
xray-fracture-detection/
├── data/
│   ├── fracture_dataset/       ← local only, not in Git
│   ├── features.npy            ← generated
│   └── labels.npy              ← generated
├── models/
│   ├── svm.pkl                 ← generated
│   ├── scaler.pkl              ← generated
│   └── metrics.json            ← generated
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── contour_analysis.py
│   ├── build_dataset.py
│   ├── train_svm.py
│   ├── predict.py
│   └── app.py
├── ui/
│   └── index.html
├── requirements.txt
├── .gitignore
└── README.md
```

## Team ownership

| File | Owner | What they explain |
|---|---|---|
| `preprocessing.py` | Member 2 | CLAHE, Gaussian blur, Canny |
| `feature_extraction.py` | Member 3 | HOG, LBP, Hu Moments |
| `contour_analysis.py` | Member 4 | Contour filtering, circularity |
| `build_dataset.py` + `train_svm.py` | Lead | Full ML pipeline |
| `app.py` + `ui/index.html` | Member 5 | Flask backend + UI |
