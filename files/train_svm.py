"""
train_svm.py
Train an SVM classifier on the extracted features.
Saves model, scaler, and a metrics.json used by the web UI.

Usage:
    python train_svm.py
"""

import os, json
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

FEATURES  = os.path.join("..", "data", "features.npy")
LABELS    = os.path.join("..", "data", "labels.npy")
MODEL_DIR = os.path.join("..", "models")


def train():
    X = np.load(FEATURES)
    y = np.load(LABELS)
    print(f"Loaded {len(X)} samples, {X.shape[1]} features each")
    print(f"  Fractured: {(y==1).sum()}  |  Not fractured: {(y==0).sum()}")

    # Split — stratify keeps class balance in both halves
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale — fit ONLY on training data to prevent leakage
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    print("\nTraining SVM (RBF, C=10, gamma=scale) ...")
    model = SVC(kernel="rbf", C=10, gamma="scale",
                probability=True, random_state=42)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    acc    = accuracy_score(y_te, y_pred)
    auc    = roc_auc_score(y_te, y_prob)
    cm     = confusion_matrix(y_te, y_pred)

    # 5-fold cross-validation for a more honest accuracy estimate
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_sc  = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy")

    print(f"\nTest accuracy : {acc:.4f} ({acc*100:.1f}%)")
    print(f"ROC-AUC       : {auc:.4f}")
    print(f"CV 5-fold     : {cv_sc.mean():.4f} ± {cv_sc.std():.4f}")
    print(f"\nConfusion matrix:\n  TN={cm[0][0]} FP={cm[0][1]}\n  FN={cm[1][0]} TP={cm[1][1]}")
    print("\n" + classification_report(y_te, y_pred,
          target_names=["Not fractured", "Fractured"]))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,  os.path.join(MODEL_DIR, "svm.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    report = classification_report(y_te, y_pred,
                 target_names=["Not fractured", "Fractured"], output_dict=True)
    metrics = {
        "accuracy":        round(float(acc), 4),
        "roc_auc":         round(float(auc), 4),
        "cv_mean":         round(float(cv_sc.mean()), 4),
        "cv_std":          round(float(cv_sc.std()),  4),
        "train_samples":   int(X_tr.shape[0]),
        "test_samples":    int(X_te.shape[0]),
        "confusion_matrix": cm.tolist(),
        "report":          report,
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved → models/svm.pkl, scaler.pkl, metrics.json")


if __name__ == "__main__":
    train()
