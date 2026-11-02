"""
Train and evaluate the SVM fracture classifier.

Run build_dataset.py first. This script:
  1. Evaluates with grouped cross-validation: every fold holds out ALL images
     that come from one original X-ray, so rotated copies never appear in
     both train and test. Hyperparameters are tuned inside each fold.
  2. Repeats the evaluation with a plain random split, only to show how much
     leakage between rotated copies inflates the score.
  3. Fits the final model on all data and saves it to models/.
"""
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "data" / "features"
MODEL_PATH = BASE_DIR / "models" / "fracture_svm.joblib"

PARAM_GRID = [
    {
        "pca__n_components": [20, 50],
        "svc__kernel": ["linear"],
        "svc__C": [0.01, 0.1, 1],
    },
    {
        "pca__n_components": [20, 50],
        "svc__kernel": ["rbf"],
        "svc__C": [1, 10, 100],
        "svc__gamma": ["scale"],
    },
]


def make_search(cv):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=42)),
        ("svc", SVC(class_weight="balanced")),
    ])
    return GridSearchCV(pipeline, PARAM_GRID, cv=cv, scoring="accuracy", n_jobs=-1)


def report(title, y, y_pred, scores=None):
    print(f"\n=== {title} ===")
    print("Accuracy:", round(accuracy_score(y, y_pred), 3))
    if scores is not None:
        print("ROC-AUC:", round(roc_auc_score(y, scores), 3))
    print("Confusion matrix (rows = true, cols = predicted) [not fractured, fractured]:")
    print(confusion_matrix(y, y_pred, labels=[0, 1]))
    print(classification_report(
        y, y_pred, labels=[0, 1], target_names=["not fractured", "fractured"],
        zero_division=0,
    ))


def grouped_evaluation(X, y, groups):
    """Nested leave-one-X-ray-out CV. Returns out-of-fold predictions and scores."""
    y_pred = np.empty_like(y)
    scores = np.empty(len(y), dtype=float)

    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        search = make_search(LeaveOneGroupOut())
        search.fit(X[train_idx], y[train_idx], groups=groups[train_idx])
        y_pred[test_idx] = search.predict(X[test_idx])
        scores[test_idx] = search.decision_function(X[test_idx])

        held_out = groups[test_idx][0]
        acc = accuracy_score(y[test_idx], y_pred[test_idx])
        print(f"  held out {held_out:<18} n={len(test_idx):<3} acc={acc:.2f}  best={search.best_params_}")

    return y_pred, scores


def main():
    X = np.load(FEATURES_DIR / "features.npy")
    y = np.load(FEATURES_DIR / "labels.npy")
    groups = np.load(FEATURES_DIR / "groups.npy")
    paths = np.load(FEATURES_DIR / "paths.npy")

    print("Samples:", X.shape[0], "| Features:", X.shape[1])
    print("Original X-rays:", len(set(groups)))

    print("\nGrouped CV (one original X-ray held out per fold):")
    y_pred, scores = grouped_evaluation(X, y, groups)
    report("Grouped CV - honest estimate", y, y_pred, scores)

    wrong_by_group = defaultdict(int)
    for path, g, true, pred in zip(paths, groups, y, y_pred):
        if true != pred:
            wrong_by_group[str(g)] += 1
    print("Misclassified images per original X-ray:", dict(wrong_by_group) or "none")

    random_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    leaky_pred = cross_val_predict(make_search(random_cv), X, y, cv=random_cv)
    report("Random 5-fold CV - leaky, for comparison only", y, leaky_pred)

    final = make_search(LeaveOneGroupOut())
    final.fit(X, y, groups=groups)
    print("\nFinal model params:", final.best_params_)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final.best_estimator_, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
