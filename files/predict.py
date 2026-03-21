"""
predict.py
Core prediction engine — used by app.py (web) and run_pipeline.py (CLI).
"""

import os, base64
import numpy as np
import joblib, cv2

from preprocessing import preprocess_stages
from feature_extraction import extract_features
from contour_analysis import draw_contour_overlay

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

_model = _scaler = None

def _load():
    global _model, _scaler
    if _model is None:
        mp = os.path.join(MODEL_DIR, "svm.pkl")
        sp = os.path.join(MODEL_DIR, "scaler.pkl")
        if not os.path.exists(mp):
            raise FileNotFoundError("Model not found — run train_svm.py first.")
        _model  = joblib.load(mp)
        _scaler = joblib.load(sp)

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

    feat_s = _scaler.transform(feat.reshape(1, -1))
    label  = int(_model.predict(feat_s)[0])
    prob   = float(_model.predict_proba(feat_s)[0][label])

    # Contour overlay — only when fractured
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
