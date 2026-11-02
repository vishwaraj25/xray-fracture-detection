"""
Compare the SVM and the CNN on the held-out test set.

Reads reports/predictions/{svm,cnn}.csv (written by the train_* scripts) and produces:
  - reports/results.md: metrics with 95% bootstrap confidence intervals
  - reports/figures/results_*.png: ROC and PR curves, confusion matrix, per-region AUC

A "metadata only" reference is added: logistic regression on body region, view,
implant flag and resolution, never looking at pixels. It shows how far shortcuts
alone get, so a real image model has to clearly beat it.

The decision threshold is chosen on the VALIDATION split, then applied unchanged
to the test split. With ~110 validation fractures a point estimate is noisy, so the
threshold must reach the target sensitivity at the lower end of a one-sided
Clopper-Pearson confidence bound, not just on average.
"""
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from config import (FIGURES_DIR, METADATA_PATH, PREDICTIONS_DIR, REGIONS, REPORTS_DIR, SEED,
                    TARGET_SENSITIVITY, THRESHOLD_CONFIDENCE)
from plotting import FRACTURED, NOT_FRACTURED, TEXT_MUTED, plt, save

N_BOOTSTRAP = 1000

# Key -> (display name, colour, line style)
MODELS = {
    "metadata": ("Metadata only (reference, no image)", TEXT_MUTED, "--"),
    "svm": ("SVM on HOG + LBP features", NOT_FRACTURED, "-"),
    "cnn": ("CNN: ResNet-18, 384 px", FRACTURED, "-"),
}


def sensitivity_lower_bound(caught, total, confidence=THRESHOLD_CONFIDENCE):
    """One-sided Clopper-Pearson lower bound on sensitivity after catching `caught` of `total`."""
    return beta.ppf(1 - confidence, caught, total - caught + 1) if caught else 0.0


def threshold_for_sensitivity(y, scores, target=TARGET_SENSITIVITY, confidence=THRESHOLD_CONFIDENCE):
    """Highest threshold at which we are `confidence` sure the sensitivity reaches the target.

    confidence=None gives the plain point estimate (sensitivity on (y, scores) >= target).
    """
    positives = np.sort(scores[y == 1])[::-1]
    n = len(positives)
    for caught in range(1, n + 1):
        reached = caught / n if confidence is None else sensitivity_lower_bound(caught, n, confidence)
        if reached >= target:
            return positives[caught - 1]
    return positives[-1]


def validation_threshold(preds):
    val = preds[preds["split"] == "val"]
    return threshold_for_sensitivity(val["y_true"].to_numpy(), val["score"].to_numpy())


def threshold_metrics(y, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "sensitivity": sensitivity,
        "specificity": tn / (tn + fp),
        "precision": precision,
        "f1": 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def bootstrap_ci(y, scores, metric, rng):
    values = []
    n = len(y)
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        values.append(metric(y[idx], scores[idx]))
    return np.percentile(values, [2.5, 97.5])


def metadata_predictions():
    """Shortcut reference: logistic regression on metadata only, trained on the train split."""
    df = pd.read_csv(METADATA_PATH)
    X = pd.get_dummies(df[["region", "view"]].astype(str))
    X["hardware"] = df["hardware"]
    X["multiscan"] = df["multiscan"]
    X["high_res"] = (df[["orig_width", "orig_height"]].max(axis=1) > 1000).astype(int)
    X = X.astype(float)

    train = df["split"] == "train"
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X[train], df.loc[train, "fractured"])
    preds = df.loc[~train, ["image_id", "split", "region", "hardware"]].copy()
    preds["y_true"] = df.loc[~train, "fractured"]
    preds["score"] = model.predict_proba(X[~train])[:, 1]
    return preds


def evaluate_model(preds, rng):
    test = preds[preds["split"] == "test"]
    y, s = test["y_true"].to_numpy(), test["score"].to_numpy()
    threshold = validation_threshold(preds)

    result = {
        "roc_auc": roc_auc_score(y, s),
        "roc_auc_ci": bootstrap_ci(y, s, roc_auc_score, rng),
        "pr_auc": average_precision_score(y, s),
        "pr_auc_ci": bootstrap_ci(y, s, average_precision_score, rng),
        "threshold": threshold,
        **threshold_metrics(y, s, threshold),
    }

    no_implant = test["hardware"] == 0
    result["roc_auc_no_implant"] = roc_auc_score(y[no_implant], s[no_implant])

    result["region_auc"] = {}
    for region in REGIONS:
        part = test[test["region"] == region]
        if part["y_true"].nunique() == 2:
            result["region_auc"][region] = (roc_auc_score(part["y_true"], part["score"]),
                                            int(part["y_true"].sum()), len(part))
    return result


def plot_curves(preds, results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for key, df in preds.items():
        label, color, style = MODELS[key]
        test = df[df["split"] == "test"]
        fpr, tpr, _ = roc_curve(test["y_true"], test["score"])
        prec, rec, _ = precision_recall_curve(test["y_true"], test["score"])
        axes[0].plot(fpr, tpr, style, color=color, label=f"{label} ({results[key]['roc_auc']:.2f})")
        axes[1].plot(rec, prec, style, color=color, label=f"{label} ({results[key]['pr_auc']:.2f})")

    prevalence = preds["cnn"].query("split == 'test'")["y_true"].mean()
    axes[0].plot([0, 1], [0, 1], ":", color=TEXT_MUTED, linewidth=1)
    axes[1].axhline(prevalence, linestyle=":", color=TEXT_MUTED, linewidth=1)
    axes[1].text(0.02, prevalence - 0.05, f"chance = {prevalence:.2f}", ha="left",
                 color=TEXT_MUTED, fontsize=9)
    axes[0].set(xlabel="False positive rate (1 − specificity)", ylabel="Sensitivity",
                title="ROC curve (test set)")
    axes[1].set(xlabel="Recall (sensitivity)", ylabel="Precision", title="Precision–recall curve (test set)",
                ylim=(0, 1.02))
    axes[0].legend(title="ROC-AUC", loc="lower right", fontsize=8, title_fontsize=8)
    axes[1].legend(title="PR-AUC", loc="upper right", fontsize=8, title_fontsize=8)
    save(fig, FIGURES_DIR / "results_curves.png")


def plot_confusion(result):
    matrix = np.array([[result["tn"], result["fp"]], [result["fn"], result["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(matrix, cmap="Blues")
    ax.grid(False)
    for (r, c), v in np.ndenumerate(matrix):
        ax.text(c, r, str(v), ha="center", va="center", fontsize=14,
                color="white" if v > matrix.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Not fractured", "Fractured"])
    ax.set_yticks([0, 1], ["Not fractured", "Fractured"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{MODELS['cnn'][0]}\nthreshold set for {TARGET_SENSITIVITY:.0%} sensitivity "
                 f"({THRESHOLD_CONFIDENCE:.0%} lower bound) on validation", fontsize=10)
    save(fig, FIGURES_DIR / "results_confusion_matrix.png")


def plot_region_auc(results):
    fig, ax = plt.subplots(figsize=(8, 4))
    keys = list(results)
    width = 0.8 / len(keys)
    regions = list(results["cnn"]["region_auc"])
    for i, key in enumerate(keys):
        label, color, style = MODELS[key]
        aucs = [results[key]["region_auc"][r][0] for r in regions]
        x = np.arange(len(regions)) + (i - (len(keys) - 1) / 2) * width
        ax.bar(x, aucs, width=width * 0.9, color=color, label=label,
               hatch="//" if style == "--" else None, alpha=0.5 if style == "--" else 1)
    counts = [results["cnn"]["region_auc"][r] for r in regions]
    ax.set_xticks(np.arange(len(regions)), [f"{r}\n{k} of {n}\nfractured" for r, (_, k, n) in zip(regions, counts)])
    ax.axhline(0.5, linestyle=":", color=TEXT_MUTED, linewidth=1)
    ax.set_ylim(0.4, 1)
    ax.set_ylabel("ROC-AUC (test)")
    ax.set_title("Performance by body region (hip and shoulder have very few fractures)", pad=28)
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(keys))
    ax.grid(axis="x", visible=False)
    save(fig, FIGURES_DIR / "results_by_region.png")


def write_report(results):
    lines = [
        "# Results",
        "",
        "Generated by `src/evaluate.py` on the held-out **test split** (627 images, 111 fractured).",
        f"The threshold is chosen on the validation split so that, at {THRESHOLD_CONFIDENCE:.0%} confidence "
        f"(one-sided Clopper-Pearson), sensitivity is at least **{TARGET_SENSITIVITY:.0%}**; it is then applied "
        "to the test split unchanged. The brackets show 95% bootstrap confidence intervals.",
        "",
        "| Model | ROC-AUC | PR-AUC | Sensitivity | Specificity | Precision | F1 | ROC-AUC, no implants |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, r in results.items():
        name = MODELS[key][0]
        if key == "cnn":
            name = f"**{name}**"
        lines.append(
            f"| {name} "
            f"| {r['roc_auc']:.3f} [{r['roc_auc_ci'][0]:.2f}–{r['roc_auc_ci'][1]:.2f}] "
            f"| {r['pr_auc']:.3f} [{r['pr_auc_ci'][0]:.2f}–{r['pr_auc_ci'][1]:.2f}] "
            f"| {r['sensitivity']:.2f} | {r['specificity']:.2f} | {r['precision']:.2f} | {r['f1']:.2f} "
            f"| {r['roc_auc_no_implant']:.3f} |"
        )
    c = results["cnn"]
    lines += [
        "",
        f"**The CNN** finds {c['tp']} of {c['tp'] + c['fn']} fractures in the test set and raises "
        f"{c['fp']} false alarms out of {c['fp'] + c['tn']} normal images.",
        "",
        "The chance level for PR-AUC is the fracture rate, about 0.18. The metadata reference never looks at the "
        "image; it shows how far shortcuts such as body region and metal implants get on their own. "
        "The \"no implants\" column removes the implant shortcut.",
        "",
        "![ROC and PR curves](figures/results_curves.png)",
        "",
        "![Confusion matrix](figures/results_confusion_matrix.png)",
        "",
        "![By region](figures/results_by_region.png)",
        "",
    ]
    (REPORTS_DIR / "results.md").write_text("\n".join(lines))
    print("\n".join(lines[:len(results) + 9]))


def main():
    missing = [k for k in ("svm", "cnn") if not (PREDICTIONS_DIR / f"{k}.csv").exists()]
    if missing:
        raise SystemExit(f"Missing predictions for {missing}; run train_svm.py and train_cnn.py first")

    rng = np.random.default_rng(SEED)
    preds = {"metadata": metadata_predictions(),
             "svm": pd.read_csv(PREDICTIONS_DIR / "svm.csv"),
             "cnn": pd.read_csv(PREDICTIONS_DIR / "cnn.csv")}
    results = {key: evaluate_model(df, rng) for key, df in preds.items()}

    plot_curves(preds, results)
    plot_confusion(results["cnn"])
    plot_region_auc(results)
    write_report(results)


if __name__ == "__main__":
    main()
