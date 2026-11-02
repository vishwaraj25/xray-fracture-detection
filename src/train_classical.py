"""
Train and compare classical ML models on hand-crafted features.

Run prepare_data.py first. For each model this script:
  1. tunes hyperparameters with 5-fold cross-validation on the training split,
     grouped by near-duplicate cluster and scored by average precision (PR-AUC),
     which suits the 18% positive rate better than accuracy;
  2. refits the best settings on the full training split;
  3. writes validation and test scores to reports/predictions/<model>.csv.
Final metrics, thresholds and plots are produced by evaluate.py.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from config import FEATURES_DIR, IMAGES_DIR, METADATA_PATH, MODELS_DIR, PREDICTIONS_DIR, SEED
from features import extract_features

FEATURES_PATH = FEATURES_DIR / "classical.npy"

MODELS = {
    "logistic_regression": (
        LogisticRegression(class_weight="balanced", max_iter=5000),
        {"pca__n_components": [50, 200], "model__C": [0.01, 0.1, 1]},
    ),
    "svm_rbf": (
        SVC(kernel="rbf", class_weight="balanced"),
        {"pca__n_components": [50, 200], "model__C": [1, 10], "model__gamma": ["scale", 0.001]},
    ),
    "random_forest": (
        RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                               random_state=SEED, n_jobs=-1),
        {"pca__n_components": [50, 200], "model__min_samples_leaf": [1, 5]},
    ),
    "gradient_boosting": (
        HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
        {"pca__n_components": [50, 200], "model__learning_rate": [0.05, 0.1],
         "model__max_leaf_nodes": [15, 31]},
    ),
}


def load_features(df):
    """Extract features once and cache them; extraction takes a few minutes."""
    if FEATURES_PATH.exists():
        X = np.load(FEATURES_PATH)
        if len(X) == len(df):
            return X
    paths = [IMAGES_DIR / p for p in df["path"]]
    rows = Parallel(n_jobs=-1)(delayed(extract_features)(p) for p in tqdm(paths, desc="Features"))
    X = np.stack(rows)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FEATURES_PATH, X)
    return X


def scores_for(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def main():
    df = pd.read_csv(METADATA_PATH)
    X = load_features(df)
    y = df["fractured"].to_numpy()
    print("Feature matrix:", X.shape)

    train = (df["split"] == "train").to_numpy()
    held_out = ~train
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for name, (estimator, grid) in MODELS.items():
        start = time.time()
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=SEED)),
            ("model", estimator),
        ])
        search = GridSearchCV(pipeline, grid, cv=cv, scoring="average_precision", n_jobs=-1)
        search.fit(X[train], y[train], groups=df.loc[train, "dup_group"])

        preds = df.loc[held_out, ["image_id", "split", "region", "hardware"]].copy()
        preds["y_true"] = y[held_out]
        preds["score"] = scores_for(search.best_estimator_, X[held_out])
        preds.to_csv(PREDICTIONS_DIR / f"{name}.csv", index=False)
        joblib.dump(search.best_estimator_, MODELS_DIR / f"{name}.joblib")

        summary[name] = {
            "cv_average_precision": round(search.best_score_, 4),
            "best_params": search.best_params_,
            "seconds": round(time.time() - start, 1),
        }
        print(f"{name:<20} CV PR-AUC {search.best_score_:.3f}  "
              f"{search.best_params_}  ({time.time() - start:.0f}s)")

    (MODELS_DIR / "classical_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
