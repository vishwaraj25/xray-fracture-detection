"""Shared paths and constants for the project."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw" / "FracAtlas"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"
METADATA_PATH = PROCESSED_DIR / "metadata.csv"
FEATURES_DIR = PROCESSED_DIR / "features"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"

# Images are downscaled once so every later step is fast
CACHE_SIZE = 384

REGIONS = ["hand", "leg", "hip", "shoulder", "mixed"]
SPLITS = ("train", "val", "test")
SEED = 42
