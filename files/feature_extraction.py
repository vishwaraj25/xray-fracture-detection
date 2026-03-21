"""
feature_extraction.py
Extracts a combined HOG + LBP + Hu Moments feature vector from one X-ray image.

Why three feature types?
  HOG   — captures bone structure and shape (gradient direction histograms)
  LBP   — captures local texture (how each pixel relates to its neighbours)
  Hu    — 7 rotation-invariant shape moments from contours; robust to
          how the X-ray was taken (angle, flip)

The three are concatenated into a single 1-D float64 vector for the SVM.
"""

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

from preprocessing import preprocess


def extract_features(image_path: str) -> np.ndarray | None:
    edges = preprocess(image_path)
    if edges is None:
        return None

    # ── HOG ─────────────────────────────────────────────────────────────
    hog_feat = hog(
        edges,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )

    # ── LBP ─────────────────────────────────────────────────────────────
    lbp = local_binary_pattern(edges, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59), range=(0, 58))
    lbp_hist = lbp_hist.astype("float64")
    lbp_hist /= lbp_hist.sum() + 1e-6

    # ── Hu Moments ──────────────────────────────────────────────────────
    # Computed on the edge image; log-transform to reduce dynamic range
    moments = cv2.moments(edges)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    return np.concatenate([hog_feat, lbp_hist, hu])
