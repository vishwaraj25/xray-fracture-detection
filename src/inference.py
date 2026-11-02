"""
Single-image analysis used by the web app.

FractureAnalyzer.analyze() runs the same steps as training/evaluation and
returns every intermediate result, so the UI can show what happened:
  1. decode     - read JPEG/PNG/DICOM, convert to grayscale
  2. preprocess - letterbox to a square and resize to the network input
  3. classify   - CNN score, averaged with a horizontally flipped copy
  4. explain    - Grad-CAM heatmap mapped back onto the original image
  5. decide     - compare the score with the validation-chosen threshold
"""
import base64
import io
import threading
import time

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

from config import MODELS_DIR, PREDICTIONS_DIR
from evaluate import threshold_for_sensitivity, threshold_metrics, TARGET_SENSITIVITY
from explain import CAM_CELL, TARGET_LAYERS, GradCAM
from sklearn.metrics import roc_auc_score
from train_cnn import EVAL_TRANSFORM, INPUT_SIZE, LOAD_SIZE, build_model, get_device, to_tensor

DISPLAY_MAX = 1024


def png_base64(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def jpeg_base64(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def decode_image(data):
    """Bytes of a JPEG/PNG/DICOM file -> uint8 grayscale array and a format label."""
    if len(data) > 132 and data[128:132] == b"DICM":
        import pydicom

        ds = pydicom.dcmread(io.BytesIO(data))
        pixels = ds.pixel_array.astype(np.float32)
        lo, hi = np.percentile(pixels, [0.5, 99.5])
        pixels = np.clip((pixels - lo) / max(hi - lo, 1e-6), 0, 1)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixels = 1 - pixels  # MONOCHROME1 stores bone as dark
        return (pixels * 255).astype(np.uint8), "DICOM"

    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format or "image"
            gray = ImageOps.exif_transpose(im).convert("L")
            return np.asarray(gray), fmt
    except Exception as err:
        raise ValueError("Unsupported or corrupted image file") from err


def letterbox(gray, size):
    """Same square padding as training (features.load_square), from an array."""
    h, w = gray.shape
    scale = size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, (top, left, new_h, new_w)


class FractureAnalyzer:
    def __init__(self, arch="resnet18"):
        self.arch = arch
        self.device = get_device()
        self.model = build_model(arch, pretrained=False)
        state = torch.load(MODELS_DIR / f"cnn_{arch}.pt", map_location="cpu")["state_dict"]
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.cam = GradCAM(self.model, TARGET_LAYERS[arch](self.model))
        # Grad-CAM keeps per-call state on the hooks, so one analysis at a time
        self.lock = threading.Lock()
        self.info = self._model_info()

    def _model_info(self):
        preds = pd.read_csv(PREDICTIONS_DIR / f"cnn_{self.arch}.csv")
        val, test = preds[preds["split"] == "val"], preds[preds["split"] == "test"]
        threshold = threshold_for_sensitivity(val["y_true"].to_numpy(), val["score"].to_numpy())
        m = threshold_metrics(test["y_true"].to_numpy(), test["score"].to_numpy(), threshold)
        return {
            "model": {"resnet18": "ResNet-18", "resnet50": "ResNet-50", "densenet121": "DenseNet-121",
                      "efficientnet_b0": "EfficientNet-B0"}.get(self.arch, self.arch),
            "dataset": "FracAtlas",
            "threshold": float(threshold),
            "target_sensitivity": TARGET_SENSITIVITY,
            "test": {
                "images": int(len(test)),
                "roc_auc": float(roc_auc_score(test["y_true"], test["score"])),
                "sensitivity": float(m["sensitivity"]),
                "specificity": float(m["specificity"]),
            },
            "device": str(self.device),
        }

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
        tensor = EVAL_TRANSFORM(to_tensor(square)).to(self.device)
        model_input = cv2.resize(square, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        step("preprocess", "Preprocessed", t, f"padded to square · {INPUT_SIZE}×{INPUT_SIZE} input")

        t = time.perf_counter()
        with self.lock:
            with torch.no_grad():
                flipped = torch.sigmoid(self.model(torch.flip(tensor, dims=[2]).unsqueeze(0))).item()
            heatmap, score = self.cam(tensor)
        score = (score + flipped) / 2
        step("classify", "Model inference", t, f"{self.info['model']} · flip-averaged")

        t = time.perf_counter()
        # Map the CAM from the padded square back onto the original image
        scale = INPUT_SIZE / LOAD_SIZE
        crop = heatmap[round(top * scale):round((top + new_h) * scale),
                       round(left * scale):round((left + new_w) * scale)]
        display_scale = min(1.0, DISPLAY_MAX / max(h, w))
        dw, dh = round(w * display_scale), round(h * display_scale)
        display = cv2.resize(gray, (dw, dh), interpolation=cv2.INTER_AREA)
        cam_full = cv2.resize(crop, (dw, dh), interpolation=cv2.INTER_CUBIC)
        cam_full = np.clip(cam_full, 0, 1)
        peak_y, peak_x = np.unravel_index(cam_full.argmax(), cam_full.shape)
        # Grad-CAM is normalised per image, so on a negative image it would still look
        # "hot". Fade it by how close the score is to the threshold to avoid implying a finding.
        threshold = self.info["threshold"]
        positive = score >= threshold
        strength = 1.0 if positive else 0.25 + 0.5 * score / threshold
        # White-on-transparent overlay keeps the UI monochrome
        overlay = np.zeros((dh, dw, 4), dtype=np.uint8)
        overlay[..., :3] = 255
        overlay[..., 3] = (np.clip((cam_full - 0.35) / 0.65, 0, 1) ** 1.5 * 200 * strength).astype(np.uint8)
        step("explain", "Attention map", t, f"Grad-CAM · {CAM_CELL} px resolution")

        t = time.perf_counter()
        step("decide", "Decision", t, f"threshold {threshold:.2f}")

        return {
            "score": score,
            "threshold": threshold,
            "positive": bool(positive),
            "finding": "Fracture suspected" if positive else "No fracture detected",
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


def confidence_note(score, threshold):
    """Plain-language distance from the decision boundary."""
    margin = abs(score - threshold)
    if margin < 0.1:
        return "Borderline: close to the decision threshold. Review carefully."
    if score >= threshold:
        return "Well above the threshold." if score > 0.8 else "Above the threshold."
    return "Well below the threshold." if score < threshold / 3 else "Below the threshold."
