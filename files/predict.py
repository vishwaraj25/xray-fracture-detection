"""
predict.py - core prediction engine
Paths are relative to this file's location so it works both locally and on Render.
"""

import os, base64
import numpy as np
import joblib, cv2

from preprocessing import preprocess_stages
from feature_extraction import extract_features
from contour_analysis import draw_contour_overlay

BASE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(BASE)
MODEL_DIR = os.path.join(ROOT, "models")

_pipe = None

def _load():
    global _pipe
    if _pipe is None:
        mp = os.path.join(MODEL_DIR, "svm.pkl")
        if not os.path.exists(mp):
            raise FileNotFoundError("Model not found — run train_svm.py first.")
        _pipe = joblib.load(mp)

def _b64(arr):
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    _, buf = cv2.imencode(".png", arr)
    return base64.b64encode(buf).decode()

def predict(image_path: str) -> dict:
    _load()

    stages = preprocess_stages(image_path)
    if stages is None:
        return {"error": "Cannot read image."}

    feat = extract_features(image_path)
    if feat is None:
        return {"error": "Feature extraction failed."}

    label  = int(_pipe.predict(feat.reshape(1, -1))[0])
    prob   = float(_pipe.predict_proba(feat.reshape(1, -1))[0][label])

    if label == 1:
        overlay = draw_contour_overlay(stages["original"], stages["edges"])
    else:
        overlay = stages["original"]

    return {
        "prediction": "FRACTURED" if label == 1 else "NOT FRACTURED",
        "label":      label,
        "confidence": round(prob, 4),
        "stages": {
            "original": _b64(stages["original"]),
            "gray":     _b64(stages["gray"]),
            "clahe":    _b64(stages["clahe"]),
            "blur":     _b64(stages["blur"]),
            "edges":    _b64(stages["edges"]),
            "overlay":  _b64(overlay),
        }
    }
