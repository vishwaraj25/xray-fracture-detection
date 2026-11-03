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


PREDICTIONS_READY = (SRC.parent / "reports" / "predictions" / "cnn.csv").exists()


@pytest.mark.skipif(not (MODEL_READY and PREDICTIONS_READY), reason="trained CNN or predictions not available")
def test_performance_endpoint_matches_test_split():
    from fastapi.testclient import TestClient

    from app import app

    body = TestClient(app).get("/api/performance").json()
    assert len(body["fractured"]) + len(body["normal"]) == 627
    caught = sum(s >= body["threshold"] for s in body["fractured"])
    assert 0 < caught <= len(body["fractured"])
    assert 0.5 < body["roc_auc"] <= 1


def test_annotated_boxes_are_normalised(tmp_path, monkeypatch):
    import explain

    (tmp_path / "Annotations" / "YOLO").mkdir(parents=True)
    (tmp_path / "Annotations" / "YOLO" / "IMG1.txt").write_text("0 0.5 0.5 0.2 0.4\n")
    monkeypatch.setattr(explain, "RAW_DIR", tmp_path)
    (x, y, w, h), = explain.annotated_boxes("IMG1.jpg")
    assert (round(x, 6), round(y, 6), w, h) == (0.4, 0.3, 0.2, 0.4)
    assert explain.annotated_boxes("missing.jpg") == []
