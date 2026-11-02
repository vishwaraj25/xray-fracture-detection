# X-Ray Fracture Detection

Detecting bone fractures in musculoskeletal X-rays. The project compares a
classical pipeline (hand-crafted HOG/LBP features with SVM, random forest and
other models) against a fine-tuned convolutional neural network. Both are checked
against a "shortcut" baseline that never looks at the pixels.

**Dataset:** [FracAtlas](https://doi.org/10.6084/m9.figshare.22363012) (Abedeen et al., *Scientific Data*, 2023):
4,083 X-rays of the hand, leg, hip and shoulder, 719 of them fractured, with radiologist-drawn fracture boxes. CC BY 4.0.

## Results

All numbers are on a held-out test split of 627 images (111 fractured). The model never sees these images during training or tuning.
The decision threshold is fixed on the validation split to catch 85% of fractures.

| Model | ROC-AUC [95% CI] | PR-AUC [95% CI] | Sensitivity | Specificity |
|---|---|---|---|---|
| Metadata only: region, view, implant (no pixels) | 0.781 [0.73–0.83] | 0.508 [0.43–0.59] | 0.94 | 0.17 |
| Logistic regression (HOG + LBP) | 0.689 [0.63–0.74] | 0.316 [0.25–0.41] | 0.71 | 0.58 |
| SVM, RBF kernel (HOG + LBP) | 0.768 [0.72–0.82] | 0.442 [0.36–0.55] | 0.75 | 0.65 |
| Random forest (HOG + LBP) | 0.774 [0.72–0.82] | 0.443 [0.36–0.54] | 0.74 | 0.64 |
| Gradient boosting (HOG + LBP) | 0.764 [0.71–0.81] | 0.433 [0.34–0.53] | 0.77 | 0.61 |
| **CNN: ResNet-18, fine-tuned** | **0.865 [0.82–0.90]** | **0.726 [0.65–0.80]** | **0.82** | **0.77** |

PR-AUC by chance is about 0.18, the fracture rate. The full table, with precision, F1 and a column without implant images, is in [reports/results.md](reports/results.md).

![ROC and PR curves](reports/figures/results_curves.png)

### Key findings

1. **The CNN is clearly the best model.** It catches 91 of 111 test fractures with 118 false alarms out of 516 normal images.
   Compared with the best classical model, PR-AUC rises from 0.44 to 0.73.
2. **The dataset contains shortcuts.** 98% of images showing metal implants are fractured, and the fracture rate ranges
   from 6% (hip) to 30% (hand). A model that only sees region, view and implant flags reaches **0.78 ROC-AUC**, which is
   as good as every hand-crafted-feature model. The classical models therefore learn little beyond these shortcuts.
   The CNN still scores 0.85 with implant images removed, so it does more than spot metal.
3. **The CNN does not always look at the fracture.** Grad-CAM peaks fall within one heatmap cell of the
   radiologist's box in 62% of correctly flagged fractures. In the rest it focuses on soft tissue, the chest
   or film edges ([reports/gradcam_resnet18.md](reports/gradcam_resnet18.md)).
4. **Evaluation pitfalls matter.** The earlier version of this project used 6 X-rays augmented by rotation, which gave
   99% accuracy with a random split and 7% on unseen X-rays. FracAtlas has no patient IDs, so near-duplicate
   images (206 groups) are detected with perceptual hashing and each group is kept inside a single split.

![Grad-CAM](reports/figures/gradcam_examples.png)

The data analysis, with figures, is in [reports/eda_summary.md](reports/eda_summary.md).

## Pipeline

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# download FracAtlas into data/raw/ (see data/README.md)

python src/prepare_data.py        # cache images, detect near-duplicates, make splits    (~2 min)
python src/eda.py                 # data analysis -> reports/eda_summary.md + figures
python src/train_baseline.py      # metadata-only shortcut baseline
python src/train_classical.py     # HOG+LBP features, 4 models, grouped CV tuning          (~7 min)
python src/train_cnn.py           # fine-tune ResNet-18 (Apple MPS / CUDA / CPU)           (~30 min on M2)
python src/evaluate.py            # compare all models -> reports/results.md + figures
python src/explain.py             # Grad-CAM vs fracture boxes
python src/predict.py path/to/xray.jpg [--model svm_rbf]

pytest tests                      # unit tests
```

| Step | What it does |
|---|---|
| `prepare_data.py` | Downscales the images, derives body region and view, flags 2 label conflicts, groups near-duplicates with a perceptual hash, and makes a 70/15/15 split stratified by label and region and grouped by duplicate cluster |
| `features.py` | Pads each image to a square without distorting it, applies CLAHE contrast enhancement, then extracts HOG, 3-scale uniform LBP, an intensity histogram and edge density (6,155 features) |
| `train_classical.py` | Scaler → PCA → {logistic regression, RBF SVM, random forest, gradient boosting}, tuned with 5-fold `StratifiedGroupKFold` on PR-AUC |
| `train_cnn.py` | ImageNet ResNet-18 (also supports ResNet-50, DenseNet-121, EfficientNet-B0), augmentation, class-weighted loss, one-cycle learning rate, early stopping on validation PR-AUC, flip test-time augmentation |
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
