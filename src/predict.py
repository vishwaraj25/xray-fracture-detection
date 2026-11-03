"""
Predict whether X-ray images show a fracture.

    python src/predict.py path/to/xray.jpg [more.jpg ...]
    python src/predict.py --model svm path/to/xray.jpg

The CNN is the default; --model svm uses the baseline. The decision threshold
is the one chosen on the validation split (see evaluate.py).

FractureAnalyzer is also what the web app uses. Its analyze() runs the same steps
as training and evaluation and returns every intermediate result for the UI:
  1. decode     - read JPEG/PNG/DICOM, convert to grayscale
  2. preprocess - letterbox to a square at the CNN's input size
  3. classify   - CNN score, averaged with a horizontally flipped copy
  4. explain    - Grad-CAM heatmap mapped back onto the original image
  5. decide     - compare the score with the validation-chosen threshold
"""
import argparse
import base64
import io
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from config import CNN_IMAGE_SIZE, PREDICTIONS_DIR, SVM_PATH, TARGET_SENSITIVITY
from evaluate import MODELS, threshold_metrics, validation_threshold
from explain import CAM_CELL, GradCAM
from preprocessing import decode_image, letterbox
from train_cnn import LOAD_SIZE, eval_transform, get_device, load_model, to_tensor

DISPLAY_MAX = 1024


def png_base64(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def jpeg_base64(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def confidence_note(score, threshold):
    """What the result means, in plain words, with how sure the model is."""
    if abs(score - threshold) < 0.1:
        return "This one is borderline, so the model is not sure. It needs a careful look by a doctor."
    if score >= threshold:
        strength = "strongly" if score > 0.8 else "fairly clearly"
        return f"The model sees signs that {strength} look like a fracture. A doctor should check this X-ray."
    strength = "clearly" if score < threshold / 3 else "fairly clearly"
    return (f"The model sees nothing that {strength} looks like a fracture. "
            "This does not rule one out: it misses about 1 in 10 fractures.")


class FractureAnalyzer:
    """The trained CNN with its Grad-CAM hook and validation-chosen threshold."""

    def __init__(self):
        self.device = get_device()
        self.model = load_model(self.device)
        self.transform = eval_transform()
        self.cam = GradCAM(self.model)
        # Grad-CAM keeps per-call state on the hooks, so one analysis at a time
        self.lock = threading.Lock()
        self.info = self._model_info()

    def _model_info(self):
        preds = pd.read_csv(PREDICTIONS_DIR / "cnn.csv")
        test = preds[preds["split"] == "test"]
        threshold = validation_threshold(preds)
        m = threshold_metrics(test["y_true"].to_numpy(), test["score"].to_numpy(), threshold)
        return {
            "model": MODELS["cnn"][0].removeprefix("CNN: "),
            "dataset": "FracAtlas + GRAZPEDWRI-DX",
            "threshold": float(threshold),
            "target_sensitivity": TARGET_SENSITIVITY,
            "test": {
                "images": int(len(test)),
                "roc_auc": float(roc_auc_score(test["y_true"], test["score"])),
                "sensitivity": float(m["sensitivity"]),
                "specificity": float(m["specificity"]),
            },
            # Size of one Grad-CAM cell as a fraction of the square input, for the UI's pointing check
            "cam_cell": CAM_CELL / CNN_IMAGE_SIZE,
            "device": str(self.device),
        }

    def score(self, data):
        """Probability only, for command-line use."""
        return self.analyze(data)["score"]

    def analyze(self, data):
        steps = []

        def step(key, label, start, detail):
            steps.append({"key": key, "label": label, "ms": round((time.perf_counter() - start) * 1000),
                          "detail": detail})

        t = time.perf_counter()
        gray, fmt = decode_image(data)
        h, w = gray.shape
        step("decode", "Image loaded", t, f"{fmt} · {w}×{h} px")

        t = time.perf_counter()
        square, (top, left, new_h, new_w) = letterbox(gray, LOAD_SIZE)
        tensor = self.transform(to_tensor(square)).to(self.device)
        model_input = cv2.resize(square, (CNN_IMAGE_SIZE, CNN_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        step("preprocess", "Preprocessed", t, f"made square and sharper · {CNN_IMAGE_SIZE}×{CNN_IMAGE_SIZE} px")

        t = time.perf_counter()
        with self.lock:
            with torch.no_grad():
                flipped = torch.sigmoid(self.model(torch.flip(tensor, dims=[2]).unsqueeze(0))).item()
            heatmap, original = self.cam(tensor)
        score = (original + flipped) / 2
        step("classify", "Model inference", t, "AI model scores the X-ray")

        t = time.perf_counter()
        display_scale = min(1.0, DISPLAY_MAX / max(h, w))
        dw, dh = round(w * display_scale), round(h * display_scale)
        display = cv2.resize(gray, (dw, dh), interpolation=cv2.INTER_AREA)
        # Crop the padding off the heatmap and stretch it onto the displayed image
        scale = CNN_IMAGE_SIZE / LOAD_SIZE
        crop = heatmap[round(top * scale):round((top + new_h) * scale),
                       round(left * scale):round((left + new_w) * scale)]
        cam = np.clip(cv2.resize(crop, (dw, dh), interpolation=cv2.INTER_CUBIC), 0, 1)
        cam = cam / (cam.max() + 1e-8)
        peak_y, peak_x = np.unravel_index(cam.argmax(), cam.shape)
        # Grad-CAM is normalised per image, so on a negative image it would still look
        # "hot". Fade it by how close the score is to the threshold to avoid implying a finding.
        threshold = self.info["threshold"]
        positive = score >= threshold
        strength = 1.0 if positive else 0.25 + 0.5 * score / threshold
        # White-on-transparent overlay keeps the UI monochrome
        overlay = np.zeros((dh, dw, 4), dtype=np.uint8)
        overlay[..., :3] = 255
        overlay[..., 3] = (np.clip((cam - 0.35) / 0.65, 0, 1) ** 1.5 * 200 * strength).astype(np.uint8)
        step("explain", "Attention map", t, "heat map of where it looked")

        t = time.perf_counter()
        step("decide", "Decision", t, f"cut-off {threshold:.2f}")

        return {
            "score": float(score),
            "threshold": threshold,
            "positive": bool(positive),
            "finding": "Possible fracture" if positive else "No fracture found",
            "confidence_note": confidence_note(score, threshold),
            "peak": {"x": float(peak_x / dw), "y": float(peak_y / dh)},
            "images": {
                "original": jpeg_base64(display),
                "model_input": png_base64(model_input),
                "attention": png_base64(overlay),
            },
            "size": {"width": int(w), "height": int(h)},
            "steps": steps,
        }


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
        analyzer = FractureAnalyzer()
        score = lambda path: analyzer.score(Path(path).read_bytes())  # noqa: E731
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
