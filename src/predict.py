"""
Predict whether X-ray images show a fracture.

Usage:
    python src/predict.py path/to/xray.jpg [more.jpg ...]
"""
import argparse
from pathlib import Path

import joblib

from features import extract_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "fracture_svm.joblib"


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("images", nargs="+", help="X-ray image file(s)")
    parser.add_argument("--model", default=MODEL_PATH, help="trained model file")
    args = parser.parse_args()

    model = joblib.load(args.model)

    for image_path in args.images:
        try:
            features = extract_features(image_path).reshape(1, -1)
        except ValueError as err:
            print(f"{image_path}: {err}")
            continue

        pred = model.predict(features)[0]
        # Distance from the SVM boundary; larger magnitude = more confident
        margin = model.decision_function(features)[0]
        label = "Fractured" if pred == 1 else "Not fractured"
        print(f"{Path(image_path).name}: {label} (margin {margin:+.2f})")


if __name__ == "__main__":
    main()
