"""Shared paths and settings for the whole pipeline."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Data
RAW_DIR = BASE_DIR / "data" / "raw" / "FracAtlas"
# External test set from another hospital, never used for training (external_validation.py)
EXTERNAL_DIR = BASE_DIR / "data" / "raw" / "GRAZPEDWRI-DX"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"
METADATA_PATH = PROCESSED_DIR / "metadata.csv"
FEATURES_PATH = PROCESSED_DIR / "features" / "hog_lbp.npy"

# The two models
MODELS_DIR = BASE_DIR / "models"
SVM_PATH = MODELS_DIR / "svm.joblib"
CNN_PATH = MODELS_DIR / "cnn.pt"

# Reports
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"

# Images are downscaled once (longer side) so every later step is fast
CACHE_SIZE = 384
# Input size of each model, in pixels
SVM_IMAGE_SIZE = 224
CNN_IMAGE_SIZE = 384

# Decision rule: be 95% confident (on validation) that at least 85% of fractures are caught
TARGET_SENSITIVITY = 0.85
THRESHOLD_CONFIDENCE = 0.95

REGIONS = ["hand", "leg", "hip", "shoulder", "mixed"]
SEED = 42
