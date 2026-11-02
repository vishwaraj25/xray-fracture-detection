"""
Image loading and preprocessing shared by the SVM, the CNN and the web app.

Every X-ray is converted to grayscale and padded to a square (letterboxed)
so bones are never stretched.
"""
import io

import cv2
import numpy as np
from PIL import Image, ImageOps


def letterbox(gray, size):
    """Resize the longer side to `size` and pad to a square.

    Returns the square image and where the original sits in it (top, left, height, width).
    """
    h, w = gray.shape
    scale = size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, (top, left, new_h, new_w)


def load_square(image_path, size):
    """Read an image file as grayscale and letterbox it to `size` x `size`."""
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image at {image_path}")
    return letterbox(gray, size)[0]


def enhance_contrast(gray):
    """CLAHE: local contrast enhancement that makes bone edges clearer on X-rays."""
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


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
