"""
preprocessing.py
Converts a raw X-ray image into a cleaned edge image ready for feature extraction.

Pipeline:
    grayscale → CLAHE → Gaussian blur → Canny edges

CLAHE (Contrast Limited Adaptive Histogram Equalization) is the key upgrade
over basic preprocessing — it enhances local contrast so faint fracture lines
become visible without over-amplifying noise in flat regions.
"""

import cv2
import numpy as np


def preprocess(image_path: str) -> np.ndarray | None:
    """
    Returns a 224x224 uint8 edge image, or None if image can't be read.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized  = cv2.resize(gray, (224, 224))
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    blurred  = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges    = cv2.Canny(blurred, 50, 150)
    return edges


def preprocess_stages(image_path: str) -> dict | None:
    """
    Returns every intermediate stage as a named dict of arrays.
    Used by the UI to animate the pipeline step by step.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized  = cv2.resize(gray, (224, 224))
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    blurred  = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges    = cv2.Canny(blurred, 50, 150)

    return {
        "original":  cv2.resize(img, (224, 224)),   # BGR
        "gray":      resized,
        "clahe":     enhanced,
        "blur":      blurred,
        "edges":     edges,
    }
