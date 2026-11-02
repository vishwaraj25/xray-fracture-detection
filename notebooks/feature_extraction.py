import cv2
import numpy as np

from skimage.feature import hog
from skimage.feature import local_binary_pattern

def extract_features(image_path):
    """
    Takes an image file path and returns a 1D feature vector
    combining HOG and LBP features.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize
    resized = cv2.resize(gray, (224, 224))

    # Noise reduction
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # ---- HOG FEATURES ----
    hog_features = hog(
        edges,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    # ---- LBP FEATURES ----
    lbp = local_binary_pattern(
        edges,
        P=8,
        R=1,
        method="uniform"
    )

    # Histogram of LBP
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 59),
        range=(0, 58)
    )

    # Normalize LBP histogram
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    # ---- COMBINE FEATURES ----
    features = np.concatenate((hog_features, lbp_hist))

    return features
