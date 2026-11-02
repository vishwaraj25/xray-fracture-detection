"""
Build the dataset: turn the raw FracAtlas download into clean, split metadata.

Steps:
  1. Read dataset.csv and attach image paths, body region and view.
  2. Downscale every image once (longer side = CACHE_SIZE) into data/processed/images.
  3. Record image statistics (original size, brightness, contrast) for EDA.
  4. Group near-duplicate images with a perceptual hash. FracAtlas has no patient
     IDs, so this is the best available guard against the same scan appearing
     in both train and test.
  5. Make a stratified, group-aware 70/15/15 train/val/test split.

Output: data/processed/metadata.csv (one row per image).

Download the dataset first (CC BY 4.0, ~340 MB):
  https://figshare.com/articles/dataset/The_dataset/22363012
and extract it so that data/raw/FracAtlas/dataset.csv exists.
"""
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import imagehash
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from config import CACHE_SIZE, IMAGES_DIR, METADATA_PATH, RAW_DIR, REGIONS, SEED

# A few FracAtlas JPEGs are missing their last bytes; the pixels are still usable
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Hashes this close (out of 64 bits) are treated as the same scan
DUPLICATE_HASH_DISTANCE = 4


def body_region(row):
    regions = [r for r in REGIONS if row[r] == 1]
    if row["mixed"] == 1 or len(regions) != 1:
        return "mixed"
    return regions[0]


def view(row):
    views = [v for v in ("frontal", "lateral", "oblique") if row[v] == 1]
    return views[0] if len(views) == 1 else "multiple"


def cache_image(src, dst):
    """Downscale to CACHE_SIZE on the longer side and return stats from the original."""
    with Image.open(src) as im:
        gray = im.convert("L")
        arr = np.asarray(gray, dtype=np.float32)
        stats = {
            "orig_width": gray.width,
            "orig_height": gray.height,
            "mean_intensity": arr.mean(),
            "std_intensity": arr.std(),
        }
        gray.thumbnail((CACHE_SIZE, CACHE_SIZE), Image.Resampling.LANCZOS)
        gray.save(dst)
        stats["phash"] = str(imagehash.phash(gray))
    return stats


def duplicate_groups(hashes):
    """Union-find over images whose perceptual hashes are within the threshold."""
    bits = np.array(
        [np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8)) for h in hashes]
    )
    parent = list(range(len(hashes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(bits)):
        distances = (bits[i + 1:] != bits[i]).sum(axis=1)
        for j in np.nonzero(distances <= DUPLICATE_HASH_DISTANCE)[0] + i + 1:
            parent[find(j)] = find(i)

    return np.array([find(i) for i in range(len(hashes))])


def assign_splits(df):
    """70/15/15 split, stratified by label+region, keeping duplicate groups together."""
    strata = df["fractured"].astype(str) + "_" + df["region"]
    split = np.full(len(df), "train", dtype=object)

    # 20 folds: 3 folds (15%) -> test, 3 folds -> val, rest -> train
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    for fold, (_, idx) in enumerate(sgkf.split(df, strata, df["dup_group"])):
        if fold < 3:
            split[idx] = "test"
        elif fold < 6:
            split[idx] = "val"
    return split


def main():
    df = pd.read_csv(RAW_DIR / "dataset.csv")
    folder_of = {p.name: p.parent.name for p in (RAW_DIR / "images").rglob("*.jpg")}
    df["folder"] = df["image_id"].map(folder_of)
    missing = df["folder"].isna().sum()
    if missing:
        raise FileNotFoundError(f"{missing} images listed in dataset.csv were not found")

    df["region"] = df.apply(body_region, axis=1)
    df["view"] = df.apply(view, axis=1)
    # dataset.csv is the label source; a couple of images sit in the other folder
    df["label_conflict"] = (df["folder"] == "Fractured").astype(int) != df["fractured"]

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    stats = []
    for image_id, folder in tqdm(zip(df["image_id"], df["folder"]), total=len(df), desc="Caching"):
        dst = IMAGES_DIR / image_id.replace(".jpg", ".png")
        stats.append(cache_image(RAW_DIR / "images" / folder / image_id, dst))
    df = pd.concat([df, pd.DataFrame(stats)], axis=1)
    df["path"] = df["image_id"].str.replace(".jpg", ".png", regex=False)

    df["dup_group"] = duplicate_groups(df["phash"].tolist())
    df["split"] = assign_splits(df)

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(METADATA_PATH, index=False)

    group_sizes = df["dup_group"].value_counts()
    print(f"Images: {len(df)} | fractured: {df['fractured'].sum()}")
    print(f"Label conflicts (folder vs CSV): {df['label_conflict'].sum()}")
    print(f"Near-duplicate groups with >1 image: {(group_sizes > 1).sum()} "
          f"covering {group_sizes[group_sizes > 1].sum()} images")
    print(pd.crosstab(df["split"], df["fractured"], margins=True))
    print(f"Saved {METADATA_PATH}")


if __name__ == "__main__":
    main()
