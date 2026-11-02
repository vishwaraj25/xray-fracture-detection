import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from inference import decode_image, letterbox  # noqa: E402

MODEL_READY = (SRC.parent / "models" / "cnn_resnet18.pt").exists()


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
def test_analyze_endpoint_end_to_end():
    from fastapi.testclient import TestClient

    from app import app

    client = TestClient(app)
    image = np.random.default_rng(1).integers(0, 255, (300, 200), dtype=np.uint8)
    res = client.post("/api/analyze", files={"file": ("x.png", png_bytes(image), "image/png")})
    assert res.status_code == 200
    body = res.json()
    assert 0 <= body["score"] <= 1
    assert [s["key"] for s in body["steps"]] == ["decode", "preprocess", "classify", "explain", "decide"]
    assert body["positive"] == (body["score"] >= body["threshold"])

    bad = client.post("/api/analyze", files={"file": ("x.txt", b"hello", "text/plain")})
    assert bad.status_code == 422
