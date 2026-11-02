# X-Ray Fracture Detection

Detecting bone fractures in musculoskeletal X-rays. The project compares a
classical pipeline (hand-crafted HOG/LBP features with SVM, random forest and
other models) against fine-tuned convolutional neural networks and an ensemble of them. Both are checked
against a "shortcut" baseline that never looks at the pixels.

**Dataset:** [FracAtlas](https://doi.org/10.6084/m9.figshare.22363012) (Abedeen et al., *Scientific Data*, 2023):
4,083 X-rays of the hand, leg, hip and shoulder, 719 of them fractured, with radiologist-drawn fracture boxes. CC BY 4.0.

## Results

All numbers are on a held-out test split of 627 images (111 fractured). The model never sees these images during training or tuning.
The decision threshold is fixed on the validation split: the highest threshold at which we can be 95% confident
(one-sided Clopper–Pearson bound) that at least 85% of fractures are caught. See [Operating point](#operating-point).

| Model | ROC-AUC [95% CI] | PR-AUC [95% CI] | Sensitivity | Specificity |
|---|---|---|---|---|
| Metadata only: region, view, implant (no pixels) | 0.781 [0.73–0.83] | 0.508 [0.43–0.59] | 0.94 | 0.17 |
| Logistic regression (HOG + LBP) | 0.689 [0.63–0.74] | 0.316 [0.25–0.41] | 0.78 | 0.49 |
| SVM, RBF kernel (HOG + LBP) | 0.768 [0.72–0.82] | 0.442 [0.36–0.55] | 0.79 | 0.62 |
| Random forest (HOG + LBP) | 0.774 [0.72–0.82] | 0.443 [0.36–0.54] | 0.80 | 0.60 |
| Gradient boosting (HOG + LBP) | 0.764 [0.71–0.81] | 0.433 [0.34–0.53] | 0.81 | 0.54 |
| CNN: ResNet-18, 224 px | 0.865 [0.82–0.90] | 0.726 [0.65–0.80] | 0.83 | 0.68 |
| CNN: DenseNet-121, 224 px | 0.858 [0.82–0.90] | 0.691 [0.61–0.76] | 0.86 | 0.65 |
| CNN: ResNet-18, 384 px | 0.868 [0.82–0.91] | 0.757 [0.68–0.82] | 0.77 | 0.80 |
| **Ensemble: ResNet-18 at 224 + 384 px** | **0.892 [0.85–0.93]** | **0.777 [0.71–0.84]** | **0.86** | **0.78** |

PR-AUC by chance is about 0.18, the fracture rate. The ensemble members are chosen by validation PR-AUC
(`src/ensemble.py`), and the app's default model is the best on validation, so the test split plays no part in either choice. The full table, with precision, F1 and a column without implant images, is in [reports/results.md](reports/results.md).

![ROC and PR curves](reports/figures/results_curves.png)

### Key findings

1. **CNNs are clearly the best models, and resolution matters more than depth.** Moving ResNet-18 from 224 to 384 px
   raised validation PR-AUC from 0.74 to 0.80; the deeper DenseNet-121 at 224 px did not help (0.73). Averaging the
   224 and 384 px ResNets gives the best test results: 96 of 111 fractures caught with 114 false alarms out of 516 normal
   images, versus 92 caught with 163 false alarms for the single 224 px model at the same rule. Compared with the best
   classical model, PR-AUC rises from 0.44 to 0.78.
2. **The dataset contains shortcuts.** 98% of images showing metal implants are fractured, and the fracture rate ranges
   from 6% (hip) to 30% (hand). A model that only sees region, view and implant flags reaches **0.78 ROC-AUC**, which is
   as good as every hand-crafted-feature model. The classical models therefore learn little beyond these shortcuts.
   The ensemble still scores 0.88 ROC-AUC with implant images removed, so it does more than spot metal.
3. **Higher resolution also makes the model look at the fracture more often.** The Grad-CAM peak lands within one heatmap
   cell of the radiologist's box in 69% of correctly flagged fractures at 384 px, against 61% at 224 px; within a strict
   margin, 53% against 25%. In the rest it still focuses on soft tissue, the chest or film edges
   ([384 px](reports/gradcam_cnn_resnet18_384.md), [224 px](reports/gradcam_cnn_resnet18.md)).
4. **Evaluation pitfalls matter.** The earlier version of this project used 6 X-rays augmented by rotation, which gave
   99% accuracy with a random split and 7% on unseen X-rays. FracAtlas has no patient IDs, so near-duplicate
   images (206 groups) are detected with perceptual hashing and each group is kept inside a single split.

![Grad-CAM](reports/figures/gradcam_cnn_resnet18_384.png)

### Operating point

Setting the threshold to give exactly 85% sensitivity on validation gave only 76% on test for the ensemble, and every
model, classical ones included, landed 4–10 points short. Two reasons:

- **Validation has only 113 fractures, so the threshold is noisy.** Re-estimating it on bootstrap resamples of the
  validation split moves the ensemble's test sensitivity anywhere from 69% to 85% (90% interval).
- **The test split has harder cases.** It has 9 hip and shoulder fractures where validation has 1, and its leg and
  mixed-region fractures were caught less often.

The threshold is now the highest one whose 95% lower confidence bound on validation sensitivity is still at least 85%.
On validation this requires catching 103 of 113 fractures (91%) rather than 96. On test the ensemble now catches 86% of
fractures at 78% specificity. The rule was fixed before looking at the test results. The single 384 px model still
reaches only 77%: its validation scores look more confident than its test scores, one reason the ensemble is preferred.

The data analysis, with figures, is in [reports/eda_summary.md](reports/eda_summary.md).

## Web app

A minimal black-and-white interface for analysing a single X-ray:

```bash
python src/app.py        # open http://127.0.0.1:8000
```

- Drop, choose or paste an image (JPEG, PNG or DICOM), or click one of the held-out test samples.
- The pipeline steps (load → preprocess → inference → attention map → decision) animate as they finish, each with its real timing.
- The result shows the finding, the score against the decision threshold, how many ensemble members flag it,
  and the model's test-set sensitivity and specificity. The attention map is the average of the members' Grad-CAMs.
- The viewer switches between the original image, the Grad-CAM attention map with its peak ringed, and the exact
  input the largest network saw (keys `1`, `2`, `3`; `Esc` for a new analysis).
- For images without a suspected fracture, the attention map is faded and no peak is marked, so it does not suggest a finding.
- Images are processed in memory on the local machine and never saved. Light and dark mode follow the system setting.

Code: `src/app.py` (FastAPI server), `src/inference.py` (single-image analysis), `src/static/` (front end).

## Pipeline

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# download FracAtlas into data/raw/ (see data/README.md)

python src/prepare_data.py        # cache images, detect near-duplicates, make splits    (~2 min)
python src/eda.py                 # data analysis -> reports/eda_summary.md + figures
python src/train_baseline.py      # metadata-only shortcut baseline
python src/train_classical.py     # HOG+LBP features, 4 models, grouped CV tuning          (~7 min)
python src/train_cnn.py           # fine-tune ResNet-18 at 224 px (Apple MPS / CUDA / CPU) (~30 min on M2)
python src/train_cnn.py --size 384                 # same at 384 px                       (~80 min on M2)
python src/train_cnn.py --arch densenet121         # optional, did not help               (~60 min on M2)
python src/ensemble.py            # pick the CNN combination with the best validation PR-AUC
python src/evaluate.py            # compare all models -> reports/results.md + figures
python src/explain.py --run cnn_resnet18_384   # Grad-CAM vs fracture boxes
python src/predict.py path/to/xray.jpg [--model svm_rbf]   # default: best model on validation
python src/app.py                 # web app on http://127.0.0.1:8000

pytest tests                      # unit tests
```

| Step | What it does |
|---|---|
| `prepare_data.py` | Downscales the images, derives body region and view, flags 2 label conflicts, groups near-duplicates with a perceptual hash, and makes a 70/15/15 split stratified by label and region and grouped by duplicate cluster |
| `features.py` | Pads each image to a square without distorting it, applies CLAHE contrast enhancement, then extracts HOG, 3-scale uniform LBP, an intensity histogram and edge density (6,155 features) |
| `train_classical.py` | Scaler → PCA → {logistic regression, RBF SVM, random forest, gradient boosting}, tuned with 5-fold `StratifiedGroupKFold` on PR-AUC |
| `train_cnn.py` | ImageNet ResNet-18 (also supports ResNet-50, DenseNet-121, EfficientNet-B0) at any input size (`--size`), augmentation, class-weighted loss, one-cycle learning rate, early stopping on validation PR-AUC, flip test-time augmentation |
| `ensemble.py` | Averages CNN scores, trying every combination of trained runs and keeping the best on validation PR-AUC |
| `evaluate.py` | ROC-AUC and PR-AUC with bootstrap confidence intervals, a threshold chosen on validation, a confusion matrix, per-region results, and a subgroup without implants |
| `explain.py` | Grad-CAM, plus a "pointing game" score against the FracAtlas fracture boxes |

## Project structure

```
src/            pipeline scripts (see the table above)
tests/          unit tests for features, splitting and thresholds
reports/        EDA summary, results, Grad-CAM report and figures (committed)
data/           README with download instructions; raw/ and processed/ are git-ignored
models/         trained models (git-ignored; recreate with the train_* scripts)
notebooks/      earlier exploratory scripts from the first version
```

## Limitations and next steps

- **No patient IDs.** Near-duplicate grouping reduces leakage between splits but cannot rule it out.
- **Small subgroups.** The test set has only 3 shoulder and 6 hip fractures, so those per-region scores are very uncertain.
- **Shortcuts.** To push the CNN toward bone features: train without implant images, or crop to the bone; use the
  fracture boxes as extra supervision (FracAtlas also includes segmentation masks); try larger backbones or a higher resolution (fractures are small).
- **Validation on other data.** Test on a second dataset (e.g. MURA) to check the model works beyond FracAtlas's hospitals and scanners.

This is a student project and not a medical device.
