import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from preprocessing import decode_image, letterbox  # noqa: E402

MODEL_READY = (SRC.parent / "models" / "cnn.pt").exists()


def png_bytes(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def test_decode_image_returns_grayscale():
    rgb = np.random.default_rng(0).integers(0, 255, (40, 30, 3), dtype=np.uint8)
    gray, fmt = decode_image(png_bytes(rgb))
    assert gray.shape == (40, 30) and fmt == "PNG"


def test_decode_image_rejects_garbage():
    with pytest.raises(ValueError):
        decode_image(b"not an image at all")


def test_letterbox_reports_placement():
    canvas, (top, left, h, w) = letterbox(np.full((100, 50), 255, dtype=np.uint8), 64)
    assert canvas.shape == (64, 64)
    assert (h, w) == (64, 32) and (top, left) == (0, 16)
    assert canvas[:, :left].max() == 0 and canvas[:, left:left + w].min() == 255


@pytest.mark.skipif(not MODEL_READY, reason="trained CNN not available")
def test_cnn_score_is_a_probability():
    from predict import CnnScorer

    image = np.random.default_rng(1).integers(0, 255, (300, 200), dtype=np.uint8)
    assert 0 <= CnnScorer().score(png_bytes(image)) <= 1


@pytest.mark.skipif(not MODEL_READY, reason="trained CNN not available")
def test_onnx_export_matches_pytorch(tmp_path):
    import onnxruntime as ort
    import torch

    from export_web import export_model

    web = export_model(tmp_path / "model.onnx")
    batch = np.random.default_rng(2).integers(0, 255, (2, 1, 384, 384)).astype(np.float32)
    logit, cam = ort.InferenceSession(str(tmp_path / "model.onnx")).run(None, {"gray": batch})
    with torch.no_grad():
        expected_logit, expected_cam = web(torch.from_numpy(batch))
    assert cam.shape == (2, 12, 12)
    assert np.abs(logit - expected_logit.numpy()).max() < 5e-3  # half-precision weights
    assert np.abs(cam - expected_cam.numpy()).max() < 5e-2


def test_verdict_levels():
    from predict import LIKELY_SCORE, level_of

    assert level_of(0.05, 0.1) == "unlikely"
    assert level_of(0.1, 0.1) == "possible"
    assert level_of(LIKELY_SCORE, 0.1) == "likely"


def test_annotated_boxes_are_normalised(tmp_path, monkeypatch):
    import explain

    (tmp_path / "Annotations" / "YOLO").mkdir(parents=True)
    (tmp_path / "Annotations" / "YOLO" / "IMG1.txt").write_text("0 0.5 0.5 0.2 0.4\n")
    monkeypatch.setattr(explain, "RAW_DIR", tmp_path)
    (x, y, w, h), = explain.annotated_boxes("IMG1.jpg")
    assert (round(x, 6), round(y, 6), w, h) == (0.4, 0.3, 0.2, 0.4)
    assert explain.annotated_boxes("missing.jpg") == []
