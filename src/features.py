import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

IMAGE_SIZE = (128, 128)
LBP_POINTS = 8
LBP_RADIUS = 1


def preprocess(image_path):
    """Load an X-ray as grayscale, resize it and boost local contrast."""
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image at {image_path}")

    gray = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # CLAHE brings out bone edges better than global equalisation on X-rays
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return cv2.GaussianBlur(gray, (3, 3), 0)


def extract_features(image_path):
    """
    Return a 1D feature vector combining HOG (shape) and a
    uniform LBP histogram (texture), both computed on the grayscale image.
    """
    gray = preprocess(image_path)

    # ---- HOG FEATURES ----
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )

    # ---- LBP FEATURES ----
    # "uniform" LBP only produces values 0..P+1, so P+2 bins
    lbp = local_binary_pattern(gray, P=LBP_POINTS, R=LBP_RADIUS, method="uniform")
    n_bins = LBP_POINTS + 2
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= lbp_hist.sum() + 1e-6

    return np.concatenate((hog_features, lbp_hist))
