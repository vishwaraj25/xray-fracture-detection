"""
Metadata-only baseline: predict fractures from body region, view, implant flag and
image resolution, without looking at any pixels.

Whatever this model achieves is what an image model could get from shortcuts alone,
so the image models need to clearly beat it.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import METADATA_PATH, PREDICTIONS_DIR


def metadata_features(df):
    X = pd.get_dummies(df[["region", "view"]].astype(str))
    X["hardware"] = df["hardware"]
    X["multiscan"] = df["multiscan"]
    X["high_res"] = (df[["orig_width", "orig_height"]].max(axis=1) > 1000).astype(int)
    return X.astype(float)


def main():
    df = pd.read_csv(METADATA_PATH)
    X = metadata_features(df)
    train = df["split"] == "train"

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X[train], df.loc[train, "fractured"])

    preds = df.loc[~train, ["image_id", "split", "region", "hardware"]].copy()
    preds["y_true"] = df.loc[~train, "fractured"]
    preds["score"] = model.predict_proba(X[~train])[:, 1]
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(PREDICTIONS_DIR / "metadata_baseline.csv", index=False)

    print("Coefficients (log-odds):")
    print(pd.Series(model.coef_[0], index=X.columns).sort_values().round(2).to_string())


if __name__ == "__main__":
    main()
