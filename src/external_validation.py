"""
External validation: score the trained CNN on X-rays from a different hospital.

GRAZPEDWRI-DX (Nagy et al., Scientific Data, 2022) holds paediatric wrist X-rays
from Graz, Austria: another country, other scanners and a younger population than
FracAtlas. Nothing from it was used for training, model choice or the threshold,
so it shows how the model behaves on data it was never tuned for.

    python src/external_validation.py           # scores every image found, then reports
    python src/external_validation.py --report  # report only, from saved scores

Reads data/raw/GRAZPEDWRI-DX/dataset.csv and the PNGs inside the images_part*.zip
files in that folder. The zips are read directly, not extracted, to save disk space;
any subset of the four parts works. Once scored, the zips can be deleted. Writes:
  - reports/predictions/cnn_external.csv: score per image
  - reports/external.md and reports/figures/external_curves.png

The threshold is the one chosen on FracAtlas validation, applied unchanged.
Confidence intervals resample patients, because one patient has several images.
"""
import argparse
import zipfile
from pathlib import PurePosixPath

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

from config import EXTERNAL_DIR, FIGURES_DIR, PREDICTIONS_DIR, REPORTS_DIR, SEED
from evaluate import threshold_metrics, validation_threshold
from plotting import FRACTURED, NOT_FRACTURED, TEXT_MUTED, plt, save
from preprocessing import letterbox
from train_cnn import LOAD_SIZE, eval_transform, get_device, load_model, predict, to_tensor

SCORES_PATH = PREDICTIONS_DIR / "cnn_external.csv"
N_BOOTSTRAP = 1000


def decode_png(data, name):
    """Graz images are 16-bit PNGs; window them to 8 bits like a viewer would (0.5-99.5th percentile)."""
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image {name}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = image.astype(np.float32)
    lo, hi = np.percentile(pixels, [0.5, 99.5])
    return (np.clip((pixels - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


class ExternalDataset(Dataset):
    """Reads images on the fly from the zips: the external set is too large to hold in memory."""

    def __init__(self, members):
        self.members = members  # (zip path, member name) pairs
        self.transform = eval_transform()
        self.zips = {}  # opened lazily, so each DataLoader worker has its own handles

    def __len__(self):
        return len(self.members)

    def __getitem__(self, idx):
        zip_path, name = self.members[idx]
        if zip_path not in self.zips:
            self.zips[zip_path] = zipfile.ZipFile(zip_path)
        gray = decode_png(self.zips[zip_path].read(name), name)
        return self.transform(to_tensor(letterbox(gray, LOAD_SIZE)[0])), 0


def load_labels():
    """One row per image with the label and the subgroup flags used in the report."""
    df = pd.read_csv(EXTERNAL_DIR / "dataset.csv", encoding="utf-8-sig")
    flags = ["fracture_visible", "cast", "metal", "diagnosis_uncertain", "initial_exam"]
    df[flags] = df[flags].fillna(0).astype(int)
    members = {}
    for zip_path in sorted(EXTERNAL_DIR.glob("images_part*.zip")):
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.endswith(".png"):
                    members[PurePosixPath(name).stem] = (str(zip_path), name)
    df["member"] = df["filestem"].map(members)
    return df.dropna(subset=["member"]).rename(columns={"fracture_visible": "y_true"})


def score_images(df, batch_size=32):
    device = get_device()
    model = load_model(device)
    loader = DataLoader(ExternalDataset(df["member"].tolist()), batch_size=batch_size, num_workers=4)
    out = df[["filestem", "patient_id", "y_true", "cast", "metal", "diagnosis_uncertain", "initial_exam"]].copy()
    out["score"] = predict(model, loader, device)
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SCORES_PATH, index=False)
    return out


def patient_bootstrap_ci(df, metric, rng):
    """95% interval from resampling whole patients."""
    groups = [g for _, g in df.groupby("patient_id")]
    values = []
    for _ in range(N_BOOTSTRAP):
        sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if sample["y_true"].nunique() == 2:
            values.append(metric(sample["y_true"], sample["score"]))
    return np.percentile(values, [2.5, 97.5])


def summarise(name, df, threshold, rng, with_ci=False):
    y, s = df["y_true"].to_numpy(), df["score"].to_numpy()
    m = threshold_metrics(y, s, threshold)
    row = {"set": name, "images": len(df), "patients": df["patient_id"].nunique(), "fractured": int(y.sum()),
           "roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s), **m}
    if with_ci:
        row["roc_ci"] = patient_bootstrap_ci(df, roc_auc_score, rng)
        row["pr_ci"] = patient_bootstrap_ci(df, average_precision_score, rng)
    return row


def fracatlas_reference(threshold):
    """The same model on FracAtlas test X-rays, all regions and hand only (closest to wrists)."""
    preds = pd.read_csv(PREDICTIONS_DIR / "cnn.csv")
    test = preds[preds["split"] == "test"]
    rows = []
    for name, part in [("FracAtlas test, all regions", test), ("FracAtlas test, hand", test[test["region"] == "hand"])]:
        y, s = part["y_true"].to_numpy(), part["score"].to_numpy()
        rows.append({"set": name, "images": len(part), "fractured": int(y.sum()), "roc_auc": roc_auc_score(y, s),
                     "pr_auc": average_precision_score(y, s), **threshold_metrics(y, s, threshold)})
    return rows, test


def plot_curves(main, test):
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    for label, df, color in [("FracAtlas test (same hospitals)", test, NOT_FRACTURED),
                             ("Graz, new hospital (main set)", main, FRACTURED)]:
        fpr, tpr, _ = roc_curve(df["y_true"], df["score"])
        ax.plot(fpr, tpr, color=color, label=f"{label}: {roc_auc_score(df['y_true'], df['score']):.2f}")
    ax.plot([0, 1], [0, 1], ":", color=TEXT_MUTED, linewidth=1)
    ax.set(xlabel="False positive rate (1 − specificity)", ylabel="Sensitivity", title="ROC curve: CNN")
    ax.legend(title="ROC-AUC", loc="lower right", fontsize=8, title_fontsize=8)
    save(fig, FIGURES_DIR / "external_curves.png")


def write_report(rows, reference, threshold):
    def fmt(r):
        ci = (f" [{r['roc_ci'][0]:.2f}–{r['roc_ci'][1]:.2f}]" if "roc_ci" in r else "")
        pr_ci = (f" [{r['pr_ci'][0]:.2f}–{r['pr_ci'][1]:.2f}]" if "pr_ci" in r else "")
        return (f"| {r['set']} | {r['images']} | {r['fractured']} | {r['roc_auc']:.3f}{ci} | {r['pr_auc']:.3f}{pr_ci} "
                f"| {r['sensitivity']:.2f} ({r['tp']}/{r['tp'] + r['fn']}) | {r['specificity']:.2f} |")

    header = ["| set | images | fractured | ROC-AUC [95% CI] | PR-AUC [95% CI] | sensitivity | specificity |",
              "|---|---|---|---|---|---|---|"]
    text = (
        "# External validation: GRAZPEDWRI-DX\n\n"
        "The FracAtlas-trained CNN scored on paediatric wrist X-rays from Graz, Austria, with no retraining. "
        f"The decision threshold ({threshold:.3f}) is the one chosen on FracAtlas validation.\n\n"
        "**Main set:** first examinations without a cast or metal, where the radiologist was certain. "
        "Casts and implants are excluded because they reveal the answer without reading the bone. "
        "Confidence intervals resample whole patients.\n\n"
        + "\n".join(header + [fmt(r) for r in rows]) + "\n\n"
        "**Same model on FracAtlas** (for comparison):\n\n"
        + "\n".join(header + [fmt(r) for r in reference]) + "\n\n"
        "![ROC curves](figures/external_curves.png)\n"
    )
    (REPORTS_DIR / "external.md").write_text(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description="External validation on GRAZPEDWRI-DX")
    parser.add_argument("--report", action="store_true", help="skip scoring, use saved scores")
    args = parser.parse_args()

    if args.report:
        df = pd.read_csv(SCORES_PATH)
    else:
        labels = load_labels()
        print(f"Scoring {len(labels)} Graz images")
        df = score_images(labels)

    threshold = validation_threshold(pd.read_csv(PREDICTIONS_DIR / "cnn.csv"))
    rng = np.random.default_rng(SEED)
    clean = (df["cast"] == 0) & (df["metal"] == 0) & (df["diagnosis_uncertain"] == 0)
    main_set = df[clean & (df["initial_exam"] == 1)]
    rows = [
        summarise("Graz, main set", main_set, threshold, rng, with_ci=True),
        summarise("Graz, all images", df, threshold, rng),
        summarise("Graz, with cast", df[df["cast"] == 1], threshold, rng),
        summarise("Graz, follow-up exams", df[clean & (df["initial_exam"] == 0)], threshold, rng),
    ]
    reference, test = fracatlas_reference(threshold)
    plot_curves(main_set, test)
    write_report(rows, reference, threshold)


if __name__ == "__main__":
    main()
