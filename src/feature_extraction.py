"""Hand-crafted image features for the SVM."""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

from config import SVM_IMAGE_SIZE
from preprocessing import enhance_contrast, load_square

LBP_SCALES = [(8, 1), (16, 2), (24, 3)]  # (points, radius)
INTENSITY_BINS = 16


def extract_features(image_path):
    """
    1D feature vector for one image:
      - HOG: bone outlines and edge orientations (shape)
      - multi-scale uniform LBP histograms: fine and coarse texture
      - intensity histogram and edge density: global appearance
    """
    gray = enhance_contrast(load_square(image_path, SVM_IMAGE_SIZE))

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )

    lbp_histograms = []
    for points, radius in LBP_SCALES:
        lbp = local_binary_pattern(gray, P=points, R=radius, method="uniform")
        # "uniform" LBP only produces values 0..P+1, so P+2 bins
        hist, _ = np.histogram(lbp.ravel(), bins=points + 2, range=(0, points + 2))
        lbp_histograms.append(hist / (hist.sum() + 1e-6))

    intensity, _ = np.histogram(gray, bins=INTENSITY_BINS, range=(0, 256))
    intensity = intensity / intensity.sum()

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edge_density = np.array([edges.mean() / 255])

    return np.concatenate([hog_features, *lbp_histograms, intensity, edge_density]).astype(np.float32)
