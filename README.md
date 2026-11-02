# X-Ray Fracture Detection

This project focuses on detecting bone fractures from X-ray images using
classical computer vision techniques and a Support Vector Machine (SVM).

## Pipeline

1. **Preprocessing**: grayscale → resize to 128×128 → CLAHE contrast enhancement → light Gaussian blur
2. **Features**: HOG (shape, 16×16 cells) + uniform LBP histogram (texture, 10 bins)
3. **Model**: `StandardScaler → PCA → SVC`, with the kernel, `C` and number of PCA components tuned by grid search

## Usage

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python src/build_dataset.py      # extract features -> data/features/
python src/train.py              # evaluate + save models/fracture_svm.joblib
python src/predict.py path/to/xray.jpg
```

Images go in `data/fracture_dataset/fractured/` and `data/fracture_dataset/not_fractured/`.

## Evaluation

The dataset has 192 images but only **6 original X-rays** (3 per class). The rest are
rotated copies, named like `11-rotated1-rotated2.jpg`, where `11` is the original.
`build_dataset.py` records the original each image came from and removes exact duplicates.

`train.py` reports two evaluations:

| Method | What it measures | Accuracy |
|---|---|---|
| Random 5-fold CV | Rotated copies of the same X-ray in both train and test (leaky) | 0.99 |
| Leave-one-X-ray-out CV | Performance on an X-ray the model has never seen | 0.07 |

The random-split score is high only because the model recognises the specific X-ray
an image came from. On a new X-ray it does no better than guessing. Getting a reliable
model needs many more **distinct** X-rays (ideally from different patients),
not more rotations of the same ones.

## Structure

- `src/features.py`: preprocessing and HOG + LBP feature extraction
- `src/build_dataset.py`: builds the feature matrix, labels and source-X-ray groups
- `src/train.py`: grouped nested cross-validation, comparison against a random split, saves the final model
- `src/predict.py`: command-line prediction for new images
- `notebooks/`: earlier exploratory scripts, kept for reference
