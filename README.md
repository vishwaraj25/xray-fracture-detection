# X-Ray Fracture Detection

Detects bone fractures in X-rays of the hand, leg, hip and shoulder.

The project has **two models**:

| | Model | Role |
|---|---|---|
| 1 | **SVM** on hand-crafted HOG + LBP features | Baseline: the classical pipeline from the project brief |
| 2 | **CNN**: ResNet-18 fine-tuned at 384 px | Final model, used by the app |

Both are checked against a **metadata-only reference** that never looks at the image. It shows how far shortcuts
(body region, metal implants) get on their own.

**Dataset:** [FracAtlas](https://doi.org/10.6084/m9.figshare.22363012) (Abedeen et al., *Scientific Data*, 2023):
4,083 X-rays, 719 of them fractured, with fracture boxes drawn by radiologists. CC BY 4.0.

## Pipeline

```
X-ray → build_dataset → preprocessing → feature_extraction → train_svm ─┐
                                      └──────────────────→ train_cnn ─┴→ evaluate → explain → predict / app
```

| Stage | Script | What it does |
|---|---|---|
| Dataset | `build_dataset.py` | Reads the labels, caches downscaled images, groups near-duplicate images with a perceptual hash, and makes a 70/15/15 train/validation/test split that keeps each group together |
| Analysis | `eda.py` | Class balance, fracture rate by region, view, implant and resolution → `reports/eda_summary.md` |
| Preprocessing | `preprocessing.py` | Grayscale, pad to a square without stretching, contrast enhancement (CLAHE); shared by both models and the app |
| Features | `feature_extraction.py` | HOG (shape), multi-scale LBP (texture), intensity histogram and edge density: 6,155 numbers per image |
| SVM | `train_svm.py` | Scaler → PCA → RBF SVM, tuned with grouped 5-fold cross-validation |
| CNN | `train_cnn.py` | ImageNet ResNet-18 at 384 px, augmentation, class-weighted loss, early stopping on validation |
| Evaluation | `evaluate.py` | Test metrics with confidence intervals, threshold chosen on validation → `reports/results.md` |
| Explanation | `explain.py` | Grad-CAM heatmaps scored against the radiologists' boxes → `reports/gradcam.md` |
| Prediction | `predict.py`, `app.py` | Command line and web app |

## Results

On the held-out test split: 627 X-rays, 111 of them fractured. No image in it was used for training or for any choice.

| Model | ROC-AUC [95% CI] | PR-AUC [95% CI] | Sensitivity | Specificity |
|---|---|---|---|---|
| Metadata only (reference, no image) | 0.781 [0.73–0.83] | 0.508 [0.43–0.59] | 0.94 | 0.17 |
| SVM on HOG + LBP | 0.768 [0.71–0.82] | 0.442 [0.36–0.54] | 0.79 | 0.62 |
| **CNN: ResNet-18, 384 px** | **0.868 [0.83–0.91]** | **0.757 [0.68–0.82]** | **0.77** | **0.80** |

Random guessing scores 0.50 ROC-AUC and about 0.18 PR-AUC. The CNN finds 85 of 111 fractures and raises 104 false
alarms out of 516 normal X-rays. Full table, curves and per-region results: [reports/results.md](reports/results.md).

![ROC and PR curves](reports/figures/results_curves.png)

### Key findings

1. **The CNN is clearly better than the SVM.** PR-AUC rises from 0.44 to 0.76.
2. **The dataset has shortcuts.** 98% of X-rays with metal implants are fractured, and the fracture rate ranges from
   6% (hip) to 30% (hand). The metadata-only reference reaches 0.78 ROC-AUC, as good as the SVM, so hand-crafted
   features add little beyond these shortcuts. The CNN still scores 0.86 with implant X-rays removed, so it does more
   than spot metal.
3. **The CNN often, but not always, looks at the fracture.** Its Grad-CAM peak lands within one heatmap cell of the
   radiologist's box in 69% of the fractures it catches. Otherwise it focuses on soft tissue or film edges
   ([reports/gradcam.md](reports/gradcam.md)).
4. **Honest evaluation matters.** The first version of this project used 6 X-rays augmented by rotation, which gave
   99% accuracy with a random split and 7% on unseen X-rays. FracAtlas has no patient IDs, so near-duplicate images
   (206 groups) are found with perceptual hashing and each group stays inside one split.

![Grad-CAM](reports/figures/gradcam.png)

### Decision threshold

A score above the threshold means "fracture". The threshold is set on the validation split as the highest value at
which we can be **95% confident that at least 85% of fractures are caught** (one-sided Clopper–Pearson bound).
Validation has only 113 fractures, so simply hitting 85% there is noisy: on test it landed 4–10 points lower for every
model. The confidence bound adds a safety margin. On test the CNN still catches 77%, because the test split holds
harder hip and shoulder cases that validation barely has.

### What we tried and dropped

Model choices were made on the validation split. On validation, a higher input resolution (224 → 384 px) raised
PR-AUC from 0.74 to 0.80. A deeper network (DenseNet-121) did not help (0.73), and an ensemble of two ResNets
tied the single 384 px model (0.80), so the single model was kept as the simpler choice. Logistic regression, a random
forest and gradient boosting all scored at or below the SVM.

## Run it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# download FracAtlas into data/raw/ (see data/README.md)

python src/run_pipeline.py              # every stage in order (about 1.5 hours on an Apple M2)
python src/run_pipeline.py --skip-cnn   # reuse a trained CNN (a few minutes)

python src/predict.py path/to/xray.jpg [--model svm]
python src/app.py                       # web app on http://127.0.0.1:8000
pytest tests
```

Each stage can also be run on its own, for example `python src/evaluate.py`.

## Web app

A minimal black-and-white interface for analysing one X-ray:

- Drop, choose or paste an image (JPEG, PNG or DICOM), or click one of the held-out test samples.
- The steps (load → preprocess → inference → attention map → decision) animate as they finish, each with its real timing.
- The result shows the finding, the score against the threshold, and the model's test sensitivity and specificity.
- The viewer switches between the original image, the Grad-CAM attention map with its peak ringed, and the exact
  384×384 input the network saw (keys `1`, `2`, `3`; `Esc` for a new analysis).
- For X-rays without a suspected fracture, the attention map is faded and no peak is marked.
- Images are processed in memory on the local machine and never saved.

## Project structure

```
src/            pipeline stages (table above), run_pipeline.py, app.py and static/ (web front end)
tests/          unit tests for preprocessing, features, splitting, thresholds and the app
reports/        data analysis, results, Grad-CAM report, figures and test-set predictions
data/           download instructions; raw/ and processed/ are git-ignored
models/         svm.joblib and cnn.pt (git-ignored; recreate with run_pipeline.py)
```

## Limitations

- **Only tested on FracAtlas.** It has not been checked on X-rays from other hospitals or scanners
  (for example MURA or GRAZPEDWRI-DX), which is where medical models most often fail.
- **No patient IDs.** Near-duplicate grouping reduces leakage between splits but cannot rule it out.
- **Small subgroups.** The test set has only 3 shoulder and 6 hip fractures, so those per-region scores are uncertain.
- **One training run.** The CNN was trained once; a different random seed would give somewhat different numbers.

This is a student project and not a medical device.
