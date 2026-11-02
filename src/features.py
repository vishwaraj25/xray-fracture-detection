"""Hand-crafted image features for the classical (non-deep-learning) models."""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

IMAGE_SIZE = 224
LBP_SCALES = [(8, 1), (16, 2), (24, 3)]  # (points, radius)
INTENSITY_BINS = 16


def load_square(image_path, size=IMAGE_SIZE):
    """Load as grayscale and letterbox to a square so bones are not stretched."""
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image at {image_path}")

    h, w = gray.shape
    scale = size / max(h, w)
    gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    top, left = (size - gray.shape[0]) // 2, (size - gray.shape[1]) // 2
    canvas[top:top + gray.shape[0], left:left + gray.shape[1]] = gray
    return canvas


def preprocess(image_path):
    """Square grayscale image with CLAHE contrast enhancement, which suits X-rays."""
    gray = load_square(image_path)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def extract_features(image_path):
    """
    1D feature vector:
      - HOG: bone outlines and edge orientations (shape)
      - multi-scale uniform LBP histograms: fine and coarse texture
      - intensity histogram and edge density: global appearance
    """
    gray = preprocess(image_path)

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
