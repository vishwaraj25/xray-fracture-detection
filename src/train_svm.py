"""
Train the SVM baseline on hand-crafted HOG + LBP features.

Run build_dataset.py first. The script:
  1. extracts features for every image (cached after the first run);
  2. tunes the SVM with 5-fold cross-validation on the training split, grouped by
     near-duplicate cluster and scored by PR-AUC, which suits the 18% fracture rate;
  3. saves the model to models/svm.joblib and its validation/test scores to
     reports/predictions/svm.csv for evaluate.py.
"""
import time

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from config import FEATURES_PATH, IMAGES_DIR, METADATA_PATH, PREDICTIONS_DIR, SEED, SVM_PATH
from feature_extraction import extract_features

PARAM_GRID = {"pca__n_components": [50, 200], "svm__C": [1, 10], "svm__gamma": ["scale", 0.001]}


def load_features(df):
    """Extract features once and cache them; extraction takes a few minutes."""
    if FEATURES_PATH.exists():
        X = np.load(FEATURES_PATH)
        if len(X) == len(df):
            return X
    paths = [IMAGES_DIR / p for p in df["path"]]
    X = np.stack(Parallel(n_jobs=-1)(delayed(extract_features)(p) for p in tqdm(paths, desc="Features")))
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(FEATURES_PATH, X)
    return X


def main():
    df = pd.read_csv(METADATA_PATH)
    X = load_features(df)
    y = df["fractured"].to_numpy()
    print("Feature matrix:", X.shape)

    train = (df["split"] == "train").to_numpy()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=SEED)),
        ("svm", SVC(kernel="rbf", class_weight="balanced")),
    ])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipeline, PARAM_GRID, cv=cv, scoring="average_precision", n_jobs=-1)

    start = time.time()
    search.fit(X[train], y[train], groups=df.loc[train, "dup_group"])
    print(f"CV PR-AUC {search.best_score_:.3f}  {search.best_params_}  ({time.time() - start:.0f}s)")

    held_out = ~train
    preds = df.loc[held_out, ["image_id", "split", "region", "hardware"]].copy()
    preds["y_true"] = y[held_out]
    # The SVM's signed distance from the boundary; evaluate.py only needs a ranking
    preds["score"] = search.best_estimator_.decision_function(X[held_out])
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(PREDICTIONS_DIR / "svm.csv", index=False)

    SVM_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, SVM_PATH)
    print(f"Saved {SVM_PATH.name}")


if __name__ == "__main__":
    main()
