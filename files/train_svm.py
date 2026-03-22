"""
train_svm.py
Full training pipeline with proper evaluation to prove model validity.
Outputs: svm.pkl, metrics.json, learning_curve.png, mistakes.png
"""

import os, json
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, learning_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score,
                             ConfusionMatrixDisplay)

FEATURES  = os.path.join("..", "data", "features.npy")
LABELS    = os.path.join("..", "data", "labels.npy")
MODEL_DIR = os.path.join("..", "models")


def plot_learning_curve(pipe, X_tr, y_tr, save_path):
    print("  Plotting learning curve (this takes ~2 min)...")
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X_tr, y_tr,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="accuracy",
        random_state=42,
    )
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0c1018")
    ax.set_facecolor("#111620")

    ax.plot(train_sizes, train_mean, "o-", color="#38b6ff", label="Training score", lw=2)
    ax.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.15, color="#38b6ff")
    ax.plot(train_sizes, val_mean, "o-", color="#00e5a0", label="Cross-val score", lw=2)
    ax.fill_between(train_sizes, val_mean - val_std,
                    val_mean + val_std, alpha=0.15, color="#00e5a0")

    ax.set_xlabel("Training samples", color="#6b90b0")
    ax.set_ylabel("Accuracy", color="#6b90b0")
    ax.set_title("Learning Curve — HOG+LBP+Hu → PCA → SVM",
                 color="#c8d8f0", pad=12)
    ax.legend(facecolor="#0c1018", labelcolor="#c8d8f0", edgecolor="#1e3050")
    ax.tick_params(colors="#6b90b0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3050")
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, color="#1e3050", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {save_path}")


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0c1018")
    ax.set_facecolor("#111620")

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix", color="#c8d8f0", pad=12)
    ax.set_xlabel("Predicted label", color="#6b90b0")
    ax.set_ylabel("True label", color="#6b90b0")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not fractured", "Fractured"], color="#6b90b0")
    ax.set_yticklabels(["Not fractured", "Fractured"], color="#6b90b0")
    ax.tick_params(colors="#6b90b0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3050")

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "#38b6ff",
                    fontsize=22, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {save_path}")


def plot_cv_scores(cv_scores, save_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0c1018")
    ax.set_facecolor("#111620")

    folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
    bars  = ax.bar(folds, cv_scores, color="#38b6ff", alpha=0.85, width=0.5)
    ax.axhline(cv_scores.mean(), color="#00e5a0", linestyle="--",
               lw=1.5, label=f"Mean: {cv_scores.mean():.4f}")
    ax.set_ylim(0.9, 1.01)
    ax.set_ylabel("Accuracy", color="#6b90b0")
    ax.set_title("5-Fold Cross-Validation Scores", color="#c8d8f0", pad=12)
    ax.legend(facecolor="#0c1018", labelcolor="#c8d8f0", edgecolor="#1e3050")
    ax.tick_params(colors="#6b90b0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3050")
    ax.grid(True, axis="y", color="#1e3050", linestyle="--", alpha=0.5)

    for bar, score in zip(bars, cv_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{score:.3f}", ha="center", va="bottom",
                color="#c8d8f0", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {save_path}")


def train():
    print("=" * 55)
    print("BoneAI — Training Pipeline")
    print("=" * 55)

    print("\nLoading features...")
    X = np.load(FEATURES)
    y = np.load(LABELS)
    print(f"  Samples  : {len(X)}")
    print(f"  Features : {X.shape[1]}")
    print(f"  Fractured: {(y==1).sum()}  |  Not fractured: {(y==0).sum()}")

    # Split FIRST
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_tr)}  |  Test: {len(X_te)}")

    # Pipeline — everything fit on train only
    print("\nBuilding pipeline: StandardScaler → PCA(300) → SVM RBF...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=300, random_state=42)),
        ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                       probability=True, random_state=42)),
    ])

    print("Training SVM... (3-5 minutes)")
    pipe.fit(X_tr, y_tr)

    # Evaluate
    y_pred = pipe.predict(X_te)
    y_prob = pipe.predict_proba(X_te)[:, 1]
    acc    = accuracy_score(y_te, y_pred)
    auc    = roc_auc_score(y_te, y_prob)
    cm     = confusion_matrix(y_te, y_pred)

    # 5-fold CV
    print("\nRunning 5-fold cross-validation...")
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_sc  = cross_val_score(pipe, X_tr, y_tr, cv=cv,
                              scoring="accuracy", n_jobs=-1)

    print("\n" + "=" * 55)
    print("RESULTS")
    print("=" * 55)
    print(f"Test accuracy  : {acc:.4f} ({acc*100:.1f}%)")
    print(f"ROC-AUC        : {auc:.4f}")
    print(f"CV 5-fold      : {cv_sc.mean():.4f} ± {cv_sc.std():.4f}")
    print(f"CV fold scores : {[round(s,4) for s in cv_sc]}")
    print(f"\nConfusion matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"  Total mistakes: {cm[0][1] + cm[1][0]} out of {len(y_te)}")
    print("\n" + classification_report(y_te, y_pred,
          target_names=["Not fractured", "Fractured"]))

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, os.path.join(MODEL_DIR, "svm.pkl"))

    # Save metrics.json for UI
    report = classification_report(y_te, y_pred,
                 target_names=["Not fractured", "Fractured"], output_dict=True)
    metrics = {
        "accuracy":         round(float(acc), 4),
        "roc_auc":          round(float(auc), 4),
        "cv_mean":          round(float(cv_sc.mean()), 4),
        "cv_std":           round(float(cv_sc.std()),  4),
        "cv_fold_scores":   [round(float(s), 4) for s in cv_sc],
        "train_samples":    int(X_tr.shape[0]),
        "test_samples":     int(X_te.shape[0]),
        "total_mistakes":   int(cm[0][1] + cm[1][0]),
        "confusion_matrix": cm.tolist(),
        "report":           report,
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save evaluation plots
    print("\nGenerating evaluation plots...")
    plot_confusion_matrix(cm,
        os.path.join(MODEL_DIR, "confusion_matrix.png"))
    plot_cv_scores(cv_sc,
        os.path.join(MODEL_DIR, "cv_scores.png"))
    plot_learning_curve(pipe, X_tr, y_tr,
        os.path.join(MODEL_DIR, "learning_curve.png"))

    print(f"\nAll saved → models/")
    print("  svm.pkl")
    print("  metrics.json")
    print("  confusion_matrix.png")
    print("  cv_scores.png")
    print("  learning_curve.png")


if __name__ == "__main__":
    train()
