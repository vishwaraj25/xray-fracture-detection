"""
Single-image analysis used by the web app and predict.py.

FractureAnalyzer.analyze() runs the same steps as training/evaluation and
returns every intermediate result, so the UI can show what happened:
  1. decode     - read JPEG/PNG/DICOM, convert to grayscale
  2. preprocess - letterbox to a square and resize to each network's input
  3. classify   - CNN score(s), averaged with a horizontally flipped copy;
                  an ensemble averages its members' probabilities
  4. explain    - Grad-CAM heatmap(s) mapped back onto the original image
  5. decide     - compare the score with the validation-chosen threshold
"""
import base64
import io
import json
import threading
import time

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score

from config import MODELS_DIR, PREDICTIONS_DIR
from evaluate import TARGET_SENSITIVITY, display_name, threshold_for_sensitivity, threshold_metrics
from explain import CAM_CELL, TARGET_LAYERS, GradCAM
from train_cnn import eval_transform, get_device, load_checkpoint, load_size, to_tensor

DISPLAY_MAX = 1024
DEFAULT_MODEL = "cnn_resnet18"


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


def default_model_name():
    """The model evaluate.py picked on validation, if it can run here."""
    best = MODELS_DIR / "best_model.json"
    name = json.loads(best.read_text())["name"] if best.exists() else DEFAULT_MODEL
    if name == "ensemble" or (MODELS_DIR / f"{name}.pt").exists():
        return name
    return DEFAULT_MODEL


class CNNMember:
    """One trained CNN with its own input size and Grad-CAM hook."""

    def __init__(self, name, device):
        self.name = name
        self.device = device
        self.model, self.arch, self.input_size = load_checkpoint(name, device)
        self.load_size = load_size(self.input_size)
        self.transform = eval_transform(self.input_size)
        self.cam = GradCAM(self.model, TARGET_LAYERS[self.arch](self.model))

    def prepare(self, gray):
        square, placement = letterbox(gray, self.load_size)
        return square, placement, self.transform(to_tensor(square)).to(self.device)

    def run(self, tensor, placement, display_shape):
        """Flip-averaged probability and a CAM resized onto the displayed image."""
        with torch.no_grad():
            flipped = torch.sigmoid(self.model(torch.flip(tensor, dims=[2]).unsqueeze(0))).item()
        heatmap, score = self.cam(tensor)
        top, left, new_h, new_w = placement
        scale = self.input_size / self.load_size
        crop = heatmap[round(top * scale):round((top + new_h) * scale),
                       round(left * scale):round((left + new_w) * scale)]
        dh, dw = display_shape
        return (score + flipped) / 2, np.clip(cv2.resize(crop, (dw, dh), interpolation=cv2.INTER_CUBIC), 0, 1)


class FractureAnalyzer:
    def __init__(self, name=None):
        self.name = name or default_model_name()
        self.device = get_device()
        if self.name == "ensemble":
            members = json.loads((MODELS_DIR / "ensemble.json").read_text())["members"]
        else:
            members = [self.name]
        self.members = [CNNMember(m, self.device) for m in members]
        # Grad-CAM keeps per-call state on the hooks, so one analysis at a time
        self.lock = threading.Lock()
        self.info = self._model_info()

    def _model_info(self):
        preds = pd.read_csv(PREDICTIONS_DIR / f"{self.name}.csv")
        val, test = preds[preds["split"] == "val"], preds[preds["split"] == "test"]
        threshold = threshold_for_sensitivity(val["y_true"].to_numpy(), val["score"].to_numpy())
        m = threshold_metrics(test["y_true"].to_numpy(), test["score"].to_numpy(), threshold)
        return {
            "model": "Ensemble" if self.name == "ensemble" else display_name(self.name).removeprefix("CNN: "),
            "members": [display_name(m.name).removeprefix("CNN: ") for m in self.members],
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
        prepared = [m.prepare(gray) for m in self.members]
        # Preview what the highest-resolution member sees
        biggest = max(range(len(self.members)), key=lambda i: self.members[i].input_size)
        size = self.members[biggest].input_size
        model_input = cv2.resize(prepared[biggest][0], (size, size), interpolation=cv2.INTER_AREA)
        sizes = sorted({m.input_size for m in self.members})
        step("preprocess", "Preprocessed", t,
             f"padded to square · {' / '.join(f'{s}×{s}' for s in sizes)} input")

        t = time.perf_counter()
        display_scale = min(1.0, DISPLAY_MAX / max(h, w))
        dw, dh = round(w * display_scale), round(h * display_scale)
        scores, cams = [], []
        with self.lock:
            for member, (_, placement, tensor) in zip(self.members, prepared):
                s, c = member.run(tensor, placement, (dh, dw))
                scores.append(s)
                cams.append(c)
        score = float(np.mean(scores))
        detail = (f"{len(self.members)} CNNs averaged · flip-averaged" if len(self.members) > 1
                  else f"{self.info['model']} · flip-averaged")
        step("classify", "Model inference", t, detail)

        t = time.perf_counter()
        display = cv2.resize(gray, (dw, dh), interpolation=cv2.INTER_AREA)
        cam_full = np.mean(cams, axis=0)
        cam_full = cam_full / (cam_full.max() + 1e-8)
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
            "member_scores": [float(s) for s in scores],
            "threshold": threshold,
            "positive": bool(positive),
            "finding": "Fracture suspected" if positive else "No fracture detected",
            "confidence_note": confidence_note(score, threshold, scores),
            "peak": {"x": float(peak_x / dw), "y": float(peak_y / dh)},
            "images": {
                "original": jpeg_base64(display),
                "model_input": png_base64(model_input),
                "attention": png_base64(overlay),
            },
            "size": {"width": int(w), "height": int(h)},
            "steps": steps,
        }


def confidence_note(score, threshold, member_scores=()):
    """Plain-language distance from the decision boundary, plus ensemble disagreement."""
    if len(member_scores) > 1:
        votes = sum(s >= threshold for s in member_scores)
        if 0 < votes < len(member_scores):
            return f"Models disagree ({votes} of {len(member_scores)} flag a fracture). Review carefully."
    margin = abs(score - threshold)
    if margin < 0.1:
        return "Borderline: close to the decision threshold. Review carefully."
    if score >= threshold:
        return "Well above the threshold." if score > 0.8 else "Above the threshold."
    return "Well below the threshold." if score < threshold / 3 else "Below the threshold."
