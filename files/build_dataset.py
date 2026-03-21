"""
build_dataset.py
Walk the dataset folder, extract features from every image,
save features.npy + labels.npy.

Supports both folder layouts automatically:
  Flat:          data/fracture_dataset/fractured/  +  data/fracture_dataset/not fractured/
  Train/val/test: data/fracture_dataset/train/fractured/  etc.

Usage:
    python build_dataset.py
"""

import os
import numpy as np
from tqdm import tqdm
from feature_extraction import extract_features

BASE_PATH  = os.path.join("..", "data", "fracture_dataset")
OUTPUT_DIR = os.path.join("..", "data")
VALID_EXT  = {".jpg", ".jpeg", ".png", ".bmp"}
LABEL_MAP  = {"fractured": 1, "not fractured": 0, "not_fractured": 0}


def find_class_folders(base):
    results = []

    def _check(folder):
        name = os.path.basename(folder).lower().strip()
        for key, label in LABEL_MAP.items():
            if name.startswith(key):
                results.append((folder, label))
                return True
        return False

    for entry in os.listdir(base):
        full = os.path.join(base, entry)
        if not os.path.isdir(full):
            continue
        if not _check(full):
            for sub in os.listdir(full):
                _check(os.path.join(full, sub))

    return results


def build():
    pairs = find_class_folders(BASE_PATH)
    if not pairs:
        print("[ERROR] No class folders found. Check BASE_PATH.")
        return

    print(f"Found {len(pairs)} class folder(s):")
    for p, l in pairs:
        print(f"  label={l}  {os.path.relpath(p)}")

    features_list, labels_list = [], []

    for folder, label in pairs:
        files = [f for f in os.listdir(folder)
                 if os.path.splitext(f)[1].lower() in VALID_EXT]
        name = "Fractured" if label == 1 else "Not fractured"
        for f in tqdm(files, desc=f"  {name} ({os.path.basename(folder)})"):
            feat = extract_features(os.path.join(folder, f))
            if feat is not None:
                features_list.append(feat)
                labels_list.append(label)

    if not features_list:
        print("[ERROR] No images processed.")
        return

    X = np.array(features_list)
    y = np.array(labels_list)

    print(f"\nTotal     : {len(X)} samples")
    print(f"Features  : {X.shape[1]} dimensions (HOG + LBP + Hu)")
    print(f"Fractured : {(y==1).sum()}")
    print(f"Not frac  : {(y==0).sum()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "features.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"),   y)
    print(f"\nSaved → {OUTPUT_DIR}/features.npy + labels.npy")


if __name__ == "__main__":
    build()
