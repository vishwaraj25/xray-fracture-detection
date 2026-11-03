"""
Second dataset: GRAZPEDWRI-DX, paediatric wrist X-rays from Graz, Austria.

GRAZPEDWRI-DX (Nagy et al., Scientific Data, 2022) comes from another country,
other scanners and a younger population than FracAtlas. The FracAtlas-only CNN
scored ROC-AUC 0.67 on it (vs 0.84 on FracAtlas hands), so part of it is used for
training (train_cnn.py --with-graz) and the rest to test.

    python src/graz.py --prepare   # cache images from the zips and split by patient
    python src/graz.py             # score the Graz images with the CNN, then report
    python src/graz.py --report    # report only, from saved scores

--prepare reads data/raw/GRAZPEDWRI-DX/dataset.csv and the PNGs inside the
images_part*.zip files (any subset of the four parts) and writes:
  - data/processed/graz/*.png: 8-bit images, letterboxed to the CNN's load size
  - data/processed/graz_metadata.csv: labels, flags and a 60/20/20 patient split
After that the zips are no longer needed and can be deleted.

Scoring and the report write reports/predictions/cnn_graz.csv, reports/graz.md and
reports/figures/graz_curves.png. Only test patients are reported. The threshold is
the one chosen on FracAtlas validation. Confidence intervals resample patients,
because one patient has several images.
"""
import argparse
import zipfile
from pathlib import PurePosixPath

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (FIGURES_DIR, GRAZ_DIR, GRAZ_IMAGES_DIR, GRAZ_METADATA_PATH, PREDICTIONS_DIR,
                    REPORTS_DIR, SEED)
from evaluate import threshold_metrics, validation_threshold
from plotting import FRACTURED, NOT_FRACTURED, TEXT_MUTED, plt, save
from preprocessing import letterbox
from train_cnn import LOAD_SIZE, XrayDataset, eval_transform, get_device, load_model, predict

SCORES_PATH = PREDICTIONS_DIR / "cnn_graz.csv"
FLAGS = ["cast", "metal", "diagnosis_uncertain", "initial_exam"]
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


def patient_splits(df):
    """60/20/20 train/val/test split that keeps every patient's images together."""
    split = np.full(len(df), "train", dtype=object)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (_, idx) in enumerate(sgkf.split(df, df["fractured"], df["patient_id"])):
        if fold == 0:
            split[idx] = "test"
        elif fold == 1:
            split[idx] = "val"
    return split


def prepare():
    df = pd.read_csv(GRAZ_DIR / "dataset.csv", encoding="utf-8-sig")
    df[FLAGS + ["fracture_visible"]] = df[FLAGS + ["fracture_visible"]].fillna(0).astype(int)
    df = df.rename(columns={"filestem": "image_id", "fracture_visible": "fractured"})

    GRAZ_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    found = set()
    for zip_path in sorted(GRAZ_DIR.glob("images_part*.zip")):
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if n.endswith(".png")]
            for name in tqdm(names, desc=zip_path.name):
                stem = PurePosixPath(name).stem
                square = letterbox(decode_png(zf.read(name), name), LOAD_SIZE)[0]
                cv2.imwrite(str(GRAZ_IMAGES_DIR / f"{stem}.png"), square)
                found.add(stem)

    df = df[df["image_id"].isin(found)].reset_index(drop=True)
    df["split"] = patient_splits(df)
    df["image_file"] = [str(GRAZ_IMAGES_DIR / f"{i}.png") for i in df["image_id"]]
    df[["image_id", "patient_id", "fractured", *FLAGS, "split", "image_file"]].to_csv(GRAZ_METADATA_PATH, index=False)
    print(df.groupby("split").agg(images=("image_id", "size"), patients=("patient_id", "nunique"),
                                  fractured=("fractured", "mean")).round(2))


def is_clean(df):
    """No cast, no metal and a certain diagnosis: casts and implants give the answer away."""
    return (df["cast"] == 0) & (df["metal"] == 0) & (df["diagnosis_uncertain"] == 0)


def training_frames():
    """Clean Graz train and validation images for train_cnn.py --with-graz."""
    df = pd.read_csv(GRAZ_METADATA_PATH)
    df = df[is_clean(df)]
    return {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val")}


def score_images():
    df = pd.read_csv(GRAZ_METADATA_PATH)
    print(f"Scoring {len(df)} Graz images")
    device = get_device()
    loader = DataLoader(XrayDataset(df, eval_transform()), batch_size=32)
    out = df[["image_id", "patient_id", "split", "fractured", *FLAGS]].rename(columns={"fractured": "y_true"})
    out["score"] = predict(load_model(device), loader, device)
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


def summarise(name, df, threshold, rng=None):
    y, s = df["y_true"].to_numpy(), df["score"].to_numpy()
    row = {"set": name, "images": len(df), "fractured": int(y.sum()), "roc_auc": roc_auc_score(y, s),
           "pr_auc": average_precision_score(y, s), **threshold_metrics(y, s, threshold)}
    if rng is not None:
        row["roc_ci"] = patient_bootstrap_ci(df, roc_auc_score, rng)
        row["pr_ci"] = patient_bootstrap_ci(df, average_precision_score, rng)
    return row


def plot_curves(main_set, fracatlas_test):
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    for label, df, color in [("FracAtlas test", fracatlas_test, NOT_FRACTURED),
                             ("Graz test patients (main set)", main_set, FRACTURED)]:
        fpr, tpr, _ = roc_curve(df["y_true"], df["score"])
        ax.plot(fpr, tpr, color=color, label=f"{label}: {roc_auc_score(df['y_true'], df['score']):.2f}")
    ax.plot([0, 1], [0, 1], ":", color=TEXT_MUTED, linewidth=1)
    ax.set(xlabel="False positive rate (1 − specificity)", ylabel="Sensitivity", title="ROC curve: CNN")
    ax.legend(title="ROC-AUC", loc="lower right", fontsize=8, title_fontsize=8)
    save(fig, FIGURES_DIR / "graz_curves.png")


def write_report(rows, threshold):
    def fmt(r):
        ci = f" [{r['roc_ci'][0]:.2f}–{r['roc_ci'][1]:.2f}]" if "roc_ci" in r else ""
        pr_ci = f" [{r['pr_ci'][0]:.2f}–{r['pr_ci'][1]:.2f}]" if "pr_ci" in r else ""
        return (f"| {r['set']} | {r['images']} | {r['fractured']} | {r['roc_auc']:.3f}{ci} | {r['pr_auc']:.3f}{pr_ci} "
                f"| {r['sensitivity']:.2f} ({r['tp']}/{r['tp'] + r['fn']}) | {r['specificity']:.2f} |")

    text = (
        "# GRAZPEDWRI-DX: paediatric wrists from another hospital\n\n"
        "The CNN on Graz test patients, whose images were never used for training or model choice. "
        f"The decision threshold ({threshold:.3f}) is the one chosen on FracAtlas validation.\n\n"
        "**Main set:** first examinations without a cast or metal, where the radiologist was certain. "
        "Casts and implants are reported separately because they reveal the answer without reading the bone. "
        "Confidence intervals resample whole patients. FracAtlas rows are for comparison.\n\n"
        "| set | images | fractured | ROC-AUC [95% CI] | PR-AUC [95% CI] | sensitivity | specificity |\n"
        "|---|---|---|---|---|---|---|\n"
        + "\n".join(fmt(r) for r in rows) + "\n\n"
        "![ROC curves](figures/graz_curves.png)\n"
    )
    (REPORTS_DIR / "graz.md").write_text(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description="GRAZPEDWRI-DX: prepare, score and report")
    parser.add_argument("--prepare", action="store_true", help="cache images from the zips and split patients")
    parser.add_argument("--report", action="store_true", help="skip scoring, use saved scores")
    args = parser.parse_args()

    if args.prepare:
        prepare()
        return
    df = pd.read_csv(SCORES_PATH) if args.report else score_images()

    preds = pd.read_csv(PREDICTIONS_DIR / "cnn.csv")
    threshold = validation_threshold(preds)
    test = df[df["split"] == "test"]
    main_set = test[is_clean(test) & (test["initial_exam"] == 1)]
    fracatlas = preds[preds["split"] == "test"]
    rows = [
        summarise("Graz test, main set", main_set, threshold, np.random.default_rng(SEED)),
        summarise("Graz test, follow-up exams", test[is_clean(test) & (test["initial_exam"] == 0)], threshold),
        summarise("Graz test, with cast", test[test["cast"] == 1], threshold),
        summarise("FracAtlas test, all regions", fracatlas, threshold),
        summarise("FracAtlas test, hand", fracatlas[fracatlas["region"] == "hand"], threshold),
    ]
    plot_curves(main_set, fracatlas)
    write_report(rows, threshold)


if __name__ == "__main__":
    main()
