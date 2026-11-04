"""
Predict whether X-ray images show a fracture.

    python src/predict.py path/to/xray.jpg [more.jpg ...]
    python src/predict.py --model svm path/to/xray.jpg

The CNN is the default; --model svm uses the baseline. The decision threshold is the one chosen
on the validation split (see evaluate.py). The CNN score is the average of the image and its
mirror image, as in evaluation.

The web demo (docs/, built by export_web.py) repeats the CNN steps in JavaScript. LEVELS and
level_of define the verdicts it shows.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from config import PREDICTIONS_DIR, SVM_PATH
from evaluate import MODELS, validation_threshold
from preprocessing import decode_image, letterbox
from train_cnn import LOAD_SIZE, eval_transform, get_device, load_model, to_tensor

LIKELY_SCORE = 0.6  # at or above this the model is confident: about 8 in 10 such test X-rays were fractured

LEVELS = {
    "unlikely": ("Fracture unlikely",
                 "The model's score is below its cut-off, so it does not suggest a fracture. "
                 "This does not rule one out: a radiologist should still read the image."),
    "possible": ("Possible fracture",
                 "The score is above the cut-off but not high, so the model is uncertain. "
                 "A radiologist should review the image."),
    "likely": ("Likely fracture",
               "The score is high, which suggests a fracture is likely. "
               "This is not a diagnosis: a radiologist should confirm it."),
}


def level_of(score, threshold):
    if score < threshold:
        return "unlikely"
    return "possible" if score < LIKELY_SCORE else "likely"


class CnnScorer:
    """The trained CNN: fracture probability for the bytes of an image file."""

    def __init__(self):
        self.device = get_device()
        self.model = load_model(self.device)
        self.transform = eval_transform()

    def score(self, data):
        gray, _ = decode_image(data)
        square, _ = letterbox(gray, LOAD_SIZE)
        tensor = self.transform(to_tensor(square)).to(self.device)
        batch = torch.stack([tensor, torch.flip(tensor, dims=[2])])
        with torch.no_grad():
            return torch.sigmoid(self.model(batch)).mean().item()


def svm_scorer():
    import joblib

    from feature_extraction import extract_features

    model = joblib.load(SVM_PATH)
    return lambda path: model.decision_function(extract_features(path).reshape(1, -1))[0]


def main():
    parser = argparse.ArgumentParser(description="Predict fractures in X-ray images")
    parser.add_argument("images", nargs="+", help="X-ray image file(s)")
    parser.add_argument("--model", choices=["cnn", "svm"], default="cnn")
    args = parser.parse_args()

    if args.model == "cnn":
        scorer = CnnScorer()
        score = lambda path: scorer.score(Path(path).read_bytes())  # noqa: E731
    else:
        score = svm_scorer()
    threshold = validation_threshold(pd.read_csv(PREDICTIONS_DIR / f"{args.model}.csv"))
    print(f"Model: {MODELS[args.model][0]} (threshold {threshold:.3f})")

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
