"""
Compare every trained model on the held-out test set.

Reads reports/predictions/*.csv (written by the train_* scripts) and produces:
  - reports/results.md: metrics table with 95% bootstrap confidence intervals
  - reports/figures/results_*.png: ROC and PR curves, confusion matrix, per-region AUC

The decision threshold is chosen on the VALIDATION split to reach a target
sensitivity (missing a fracture is worse than a false alarm), then applied
unchanged to the test split. With ~110 validation fractures a point estimate
is noisy, so the threshold must reach the target at the lower end of a
one-sided Clopper-Pearson confidence bound, not just on average.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from config import FIGURES_DIR, MODELS_DIR, PREDICTIONS_DIR, REGIONS, REPORTS_DIR, SEED
from plotting import FRACTURED, NOT_FRACTURED, SERIES, TEXT_MUTED, plt, save

TARGET_SENSITIVITY = 0.85
# One-sided confidence that the true sensitivity is at least the target
THRESHOLD_CONFIDENCE = 0.95
N_BOOTSTRAP = 1000

DISPLAY_NAMES = {
    "metadata_baseline": "Metadata only (no pixels)",
    "logistic_regression": "Logistic regression",
    "svm_rbf": "SVM (RBF)",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "cnn_resnet18": "CNN: ResNet-18",
    "cnn_resnet18_384": "CNN: ResNet-18, 384 px",
    "cnn_resnet50": "CNN: ResNet-50",
    "cnn_densenet121": "CNN: DenseNet-121",
    "cnn_efficientnet_b0": "CNN: EfficientNet-B0",
    "ensemble": "Ensemble of CNNs",
}


def display_name(name):
    if name in DISPLAY_NAMES:
        return DISPLAY_NAMES[name]
    if name.startswith("cnn_"):
        arch, _, size = name.removeprefix("cnn_").rpartition("_")
        return f"CNN: {arch}, {size} px" if size.isdigit() else f"CNN: {name.removeprefix('cnn_')}"
    return name


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


def load_predictions():
    preds = {}
    for path in sorted(PREDICTIONS_DIR.glob("*.csv")):
        preds[path.stem] = pd.read_csv(path)
    order = list(DISPLAY_NAMES)
    return dict(sorted(preds.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99))


def evaluate_model(df, rng):
    val, test = df[df["split"] == "val"], df[df["split"] == "test"]
    y, s = test["y_true"].to_numpy(), test["score"].to_numpy()
    threshold = threshold_for_sensitivity(val["y_true"].to_numpy(), val["score"].to_numpy())

    result = {
        "val_pr_auc": average_precision_score(val["y_true"], val["score"]),
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
    for i, (name, df) in enumerate(preds.items()):
        test = df[df["split"] == "test"]
        color = SERIES[i % len(SERIES)]
        dashed = "--" if name == "metadata_baseline" else "-"
        fpr, tpr, _ = roc_curve(test["y_true"], test["score"])
        prec, rec, _ = precision_recall_curve(test["y_true"], test["score"])
        label = display_name(name)
        axes[0].plot(fpr, tpr, dashed, color=color,
                     label=f"{label} ({results[name]['roc_auc']:.2f})")
        axes[1].plot(rec, prec, dashed, color=color,
                     label=f"{label} ({results[name]['pr_auc']:.2f})")

    prevalence = next(iter(preds.values())).query("split == 'test'")["y_true"].mean()
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


def plot_confusion(name, result):
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
    ax.set_title(f"{display_name(name)}\nthreshold set for {TARGET_SENSITIVITY:.0%} sensitivity "
                 f"({THRESHOLD_CONFIDENCE:.0%} lower bound) on validation", fontsize=10)
    save(fig, FIGURES_DIR / "results_confusion_matrix.png")


def plot_region_auc(names, results):
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        regions = list(results[name]["region_auc"])
        aucs = [results[name]["region_auc"][r][0] for r in regions]
        x = np.arange(len(regions)) + (i - (len(names) - 1) / 2) * width
        ax.bar(x, aucs, width=width * 0.9, color=SERIES[i % len(SERIES)],
               label=display_name(name))
    regions = list(results[names[0]]["region_auc"])
    counts = [results[names[0]]["region_auc"][r] for r in regions]
    ax.set_xticks(np.arange(len(regions)), [f"{r}\n{k} of {n}\nfractured" for r, (_, k, n) in zip(regions, counts)])
    ax.axhline(0.5, linestyle=":", color=TEXT_MUTED, linewidth=1)
    ax.set_ylim(0.4, 1)
    ax.set_ylabel("ROC-AUC (test)")
    ax.set_title("Performance by body region (hip and shoulder have very few fractures)", pad=28)
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(names))
    ax.grid(axis="x", visible=False)
    save(fig, FIGURES_DIR / "results_by_region.png")


def write_report(results, best):
    lines = [
        "# Results",
        "",
        "Generated by `src/evaluate.py` on the held-out **test split** (627 images, 111 fractured).",
        f"The threshold is chosen on the validation split so that, at {THRESHOLD_CONFIDENCE:.0%} confidence "
        f"(one-sided Clopper-Pearson), sensitivity is at least **{TARGET_SENSITIVITY:.0%}**; it is then applied "
        "to the test split unchanged. The brackets show 95% bootstrap confidence intervals.",
        "",
        "The best model is picked by **validation** PR-AUC (first column), so choosing it does not use the test set.",
        "",
        "| Model | Val PR-AUC | ROC-AUC | PR-AUC | Sensitivity | Specificity | Precision | F1 | ROC-AUC, no implants |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        bold = "**" if name == best else ""
        lines.append(
            f"| {bold}{display_name(name)}{bold} "
            f"| {r['val_pr_auc']:.3f} "
            f"| {r['roc_auc']:.3f} [{r['roc_auc_ci'][0]:.2f}–{r['roc_auc_ci'][1]:.2f}] "
            f"| {r['pr_auc']:.3f} [{r['pr_auc_ci'][0]:.2f}–{r['pr_auc_ci'][1]:.2f}] "
            f"| {r['sensitivity']:.2f} | {r['specificity']:.2f} | {r['precision']:.2f} | {r['f1']:.2f} "
            f"| {r['roc_auc_no_implant']:.3f} |"
        )
    b = results[best]
    lines += [
        "",
        f"**Best model: {display_name(best)}.** At the validation-chosen threshold it finds "
        f"{b['tp']} of {b['tp'] + b['fn']} fractures in the test set and raises {b['fp']} false alarms "
        f"out of {b['fp'] + b['tn']} normal images.",
        "",
        "The chance level for PR-AUC is the fracture rate, about 0.18. The \"no implants\" column removes "
        "the metal-implant shortcut found during the data analysis.",
        "",
        "![ROC and PR curves](figures/results_curves.png)",
        "",
        "![Confusion matrix](figures/results_confusion_matrix.png)",
        "",
        "![By region](figures/results_by_region.png)",
        "",
    ]
    (REPORTS_DIR / "results.md").write_text("\n".join(lines))
    print("\n".join(lines[:len(results) + 10]))


def main():
    rng = np.random.default_rng(SEED)
    preds = load_predictions()
    if not preds:
        raise SystemExit("No predictions found; run the train_* scripts first")

    results = {name: evaluate_model(df, rng) for name, df in preds.items()}
    # Pick on validation, never on test. The metadata baseline is a reference, not an image model.
    image_models = [n for n in results if n != "metadata_baseline"]
    best = max(image_models, key=lambda n: results[n]["val_pr_auc"])
    classical = [n for n in image_models if not n.startswith("cnn") and n != "ensemble"]
    best_classical = max(classical, key=lambda n: results[n]["val_pr_auc"], default=None)

    # Curves: baseline, best classical model and every CNN/ensemble (max 8 series)
    plotted = [n for n in preds if n in ("metadata_baseline", best_classical)
               or n.startswith("cnn") or n == "ensemble"][:len(SERIES)]
    plot_curves({n: preds[n] for n in plotted}, results)
    plot_confusion(best, results[best])
    compared = [n for n in dict.fromkeys(["metadata_baseline", best_classical, best]) if n]
    plot_region_auc(compared, results)
    write_report(results, best)

    # predict.py and the web app use this as their default model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "best_model.json").write_text(json.dumps({"name": best}))


if __name__ == "__main__":
    main()
