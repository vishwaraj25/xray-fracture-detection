import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluate import sensitivity_lower_bound, threshold_for_sensitivity, threshold_metrics  # noqa: E402
from build_dataset import assign_splits, body_region, duplicate_groups  # noqa: E402
from feature_extraction import extract_features  # noqa: E402
from preprocessing import load_square  # noqa: E402

IMAGE_SIZE = 224


@pytest.fixture
def tall_image(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "xray.png"
    cv2.imwrite(str(path), rng.integers(0, 255, (300, 150), dtype=np.uint8))
    return path


def test_load_square_letterboxes_without_stretching(tall_image):
    square = load_square(tall_image, IMAGE_SIZE)
    assert square.shape == (IMAGE_SIZE, IMAGE_SIZE)
    # A 2:1 tall image fills the middle half of the width; the sides are padding
    assert square[:, :IMAGE_SIZE // 5].max() == 0
    assert square[:, IMAGE_SIZE // 2].max() > 0


def test_features_are_fixed_length_and_finite(tall_image):
    a = extract_features(tall_image)
    b = extract_features(tall_image)
    assert a.ndim == 1 and np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)


def test_unreadable_image_raises(tmp_path):
    with pytest.raises(ValueError):
        extract_features(tmp_path / "missing.png")


def test_body_region_marks_multi_region_as_mixed():
    row = {"hand": 1, "leg": 1, "hip": 0, "shoulder": 0, "mixed": 0}
    assert body_region(row) == "mixed"
    assert body_region({**row, "leg": 0}) == "hand"


def test_duplicate_groups_join_close_hashes_only():
    hashes = ["0000000000000000", "0000000000000001", "ffffffffffffffff"]
    groups = duplicate_groups(hashes)
    assert groups[0] == groups[1] != groups[2]


def test_splits_keep_duplicate_groups_together():
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({
        "fractured": rng.integers(0, 2, n),
        "region": rng.choice(["hand", "leg"], n),
        "dup_group": rng.integers(0, 150, n),
    })
    df["split"] = assign_splits(df)
    assert (df.groupby("dup_group")["split"].nunique() == 1).all()
    assert set(df["split"]) == {"train", "val", "test"}


def test_threshold_reaches_target_sensitivity():
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.5, 0.7, 0.2, 0.6, 0.8, 0.9])
    threshold = threshold_for_sensitivity(y, scores, target=0.75, confidence=None)
    metrics = threshold_metrics(y, scores, threshold)
    assert metrics["sensitivity"] >= 0.75
    assert metrics["tp"] + metrics["fn"] == 4


def test_confident_threshold_is_more_conservative():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(500), np.ones(110)].astype(int)
    scores = np.r_[rng.normal(0, 1, 500), rng.normal(1.5, 1, 110)]
    point = threshold_for_sensitivity(y, scores, target=0.85, confidence=None)
    confident = threshold_for_sensitivity(y, scores, target=0.85, confidence=0.95)
    assert confident < point
    caught = (scores[y == 1] >= confident).sum()
    assert sensitivity_lower_bound(caught, 110, 0.95) >= 0.85
    assert sensitivity_lower_bound(caught - 1, 110, 0.95) < 0.85
