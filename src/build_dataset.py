"""
Extract features from every image in data/fracture_dataset and save them.

Outputs (in data/features/):
  features.npy : feature matrix, one row per image
  labels.npy   : 1 = fractured, 0 = not fractured
  groups.npy   : source X-ray id, so rotated copies stay in the same fold
  paths.npy    : image path for each row (for error analysis)
"""
import hashlib
from pathlib import Path

import numpy as np

from features import extract_features

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "fracture_dataset"
OUT_DIR = BASE_DIR / "data" / "features"

CLASSES = {"fractured": 1, "not_fractured": 0}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def source_id(class_name, path):
    # Files are named like "11-rotated1-rotated2.jpg": "11" is the original X-ray
    return f"{class_name}/{path.stem.split('-')[0]}"


def main():
    features, labels, groups, paths = [], [], [], []
    seen_hashes = set()
    skipped_duplicates = 0

    for class_name, label in CLASSES.items():
        class_dir = DATA_DIR / class_name
        image_paths = sorted(
            p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        print(f"Processing {len(image_paths)} images in {class_name}/")

        for path in image_paths:
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            if digest in seen_hashes:
                skipped_duplicates += 1
                continue
            seen_hashes.add(digest)

            try:
                features.append(extract_features(path))
            except ValueError as err:
                print(f"  Skipping {path.name}: {err}")
                continue
            labels.append(label)
            groups.append(source_id(class_name, path))
            paths.append(str(path.relative_to(BASE_DIR)))

    X = np.array(features)
    y = np.array(labels)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "features.npy", X)
    np.save(OUT_DIR / "labels.npy", y)
    np.save(OUT_DIR / "groups.npy", np.array(groups))
    np.save(OUT_DIR / "paths.npy", np.array(paths))

    print(f"Skipped {skipped_duplicates} exact duplicate images")
    print("Feature shape:", X.shape)
    print("Fractured:", int(y.sum()), "| Not fractured:", int((y == 0).sum()))
    print("Original X-rays (groups):", sorted(set(groups)))
    print(f"Saved to {OUT_DIR.relative_to(BASE_DIR)}/")


if __name__ == "__main__":
    main()
