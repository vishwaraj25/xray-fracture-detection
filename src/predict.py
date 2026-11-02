"""
Predict whether X-ray images show a fracture.

Usage:
    python src/predict.py path/to/xray.jpg [more.jpg ...]
    python src/predict.py --model svm_rbf path/to/xray.jpg

--model is "ensemble", a CNN run (cnn_resnet18, cnn_resnet18_384, ...) or a classical
model (logistic_regression, svm_rbf, random_forest, gradient_boosting).
The default is the model evaluate.py picked on the validation split.
The decision threshold is the one chosen on the validation split.
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from config import MODELS_DIR, PREDICTIONS_DIR
from evaluate import threshold_for_sensitivity
from features import extract_features


def validation_threshold(name):
    preds = pd.read_csv(PREDICTIONS_DIR / f"{name}.csv")
    val = preds[preds["split"] == "val"]
    return threshold_for_sensitivity(val["y_true"].to_numpy(), val["score"].to_numpy())


def classical_scorer(name):
    model = joblib.load(MODELS_DIR / f"{name}.joblib")

    def score(path):
        features = extract_features(path).reshape(1, -1)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(features)[0, 1]
        return model.decision_function(features)[0]

    return score


def cnn_scorer(name):
    from inference import FractureAnalyzer

    analyzer = FractureAnalyzer(name)
    return lambda path: analyzer.score(Path(path).read_bytes())


def main():
    from inference import default_model_name

    parser = argparse.ArgumentParser(description="Predict fractures in X-ray images")
    parser.add_argument("images", nargs="+", help="X-ray image file(s)")
    parser.add_argument("--model", default=None, help="model name (default: best on validation)")
    args = parser.parse_args()

    name = args.model or default_model_name()
    if not name.startswith("cnn_") and name != "ensemble" and (MODELS_DIR / f"cnn_{name}.pt").exists():
        name = f"cnn_{name}"  # allow "--model resnet18"
    is_cnn = name == "ensemble" or (MODELS_DIR / f"{name}.pt").exists()
    score = cnn_scorer(name) if is_cnn else classical_scorer(name)
    threshold = validation_threshold(name)
    print(f"Model: {name} (threshold {threshold:.3f})")

    for image_path in args.images:
        try:
            s = score(image_path)
        except (ValueError, FileNotFoundError) as err:
            print(f"{image_path}: {err}")
            continue
        label = "Fractured" if s >= threshold else "Not fractured"
        print(f"{Path(image_path).name}: {label} (score {s:.3f})")


if __name__ == "__main__":
    main()
