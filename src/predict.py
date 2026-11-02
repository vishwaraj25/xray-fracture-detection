"""
Predict whether X-ray images show a fracture.

Usage:
    python src/predict.py path/to/xray.jpg [more.jpg ...]
    python src/predict.py --model svm_rbf path/to/xray.jpg

--model is a CNN architecture (resnet18, ...) or a classical model name
(logistic_regression, svm_rbf, random_forest, gradient_boosting).
The decision threshold is the one chosen on the validation split by evaluate.py.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import MODELS_DIR, PREDICTIONS_DIR
from evaluate import threshold_for_sensitivity
from features import extract_features, load_square


def validation_threshold(prediction_name):
    preds = pd.read_csv(PREDICTIONS_DIR / f"{prediction_name}.csv")
    val = preds[preds["split"] == "val"]
    return threshold_for_sensitivity(val["y_true"].to_numpy(), val["score"].to_numpy())


def cnn_scorer(arch):
    import torch

    from train_cnn import EVAL_TRANSFORM, LOAD_SIZE, build_model, get_device, to_tensor

    device = get_device()
    model = build_model(arch, pretrained=False)
    model.load_state_dict(torch.load(MODELS_DIR / f"cnn_{arch}.pt", map_location="cpu")["state_dict"])
    model.to(device).eval()

    @torch.no_grad()
    def score(path):
        image = EVAL_TRANSFORM(to_tensor(load_square(path, LOAD_SIZE))).unsqueeze(0).to(device)
        logits = (model(image) + model(torch.flip(image, dims=[3]))) / 2
        return torch.sigmoid(logits).item()

    return score


def classical_scorer(name):
    model = joblib.load(MODELS_DIR / f"{name}.joblib")

    def score(path):
        features = extract_features(path).reshape(1, -1)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(features)[0, 1]
        return model.decision_function(features)[0]

    return score


def default_model():
    best = MODELS_DIR / "best_model.json"
    return json.loads(best.read_text())["name"] if best.exists() else "resnet18"


def main():
    parser = argparse.ArgumentParser(description="Predict fractures in X-ray images")
    parser.add_argument("images", nargs="+", help="X-ray image file(s)")
    parser.add_argument("--model", default=None, help="model name (default: best from evaluate.py)")
    args = parser.parse_args()

    name = args.model or default_model()
    is_cnn = (MODELS_DIR / f"cnn_{name}.pt").exists()
    prediction_name = f"cnn_{name}" if is_cnn else name
    score = cnn_scorer(name) if is_cnn else classical_scorer(name)
    threshold = validation_threshold(prediction_name)
    print(f"Model: {prediction_name} (threshold {threshold:.3f})")

    for image_path in args.images:
        try:
            s = score(image_path)
        except ValueError as err:
            print(f"{image_path}: {err}")
            continue
        label = "Fractured" if s >= threshold else "Not fractured"
        print(f"{Path(image_path).name}: {label} (score {s:.3f})")


if __name__ == "__main__":
    main()
