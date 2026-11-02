"""
Combine trained CNNs into an ensemble by averaging their fracture probabilities.

    python src/ensemble.py

Every subset of the available CNN runs is scored on the VALIDATION split; the
best subset by validation PR-AUC becomes the ensemble. The test split is not
looked at here, so the test results from evaluate.py stay unbiased.

Writes reports/predictions/ensemble.csv and models/ensemble.json.
"""
import itertools
import json

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import MODELS_DIR, PREDICTIONS_DIR


def load_runs():
    runs = {}
    for path in sorted(PREDICTIONS_DIR.glob("cnn_*.csv")):
        if (MODELS_DIR / f"{path.stem}.pt").exists():
            runs[path.stem] = pd.read_csv(path).set_index("image_id")
    return runs


def average(runs, members):
    base = runs[members[0]].drop(columns="score").copy()
    base["score"] = sum(runs[m]["score"] for m in members) / len(members)
    return base


def main():
    runs = load_runs()
    if len(runs) < 2:
        raise SystemExit("Need at least two trained CNN runs to build an ensemble")

    rows = []
    for k in range(1, len(runs) + 1):
        for members in itertools.combinations(runs, k):
            val = average(runs, list(members)).query("split == 'val'")
            rows.append({
                "members": members,
                "val_pr_auc": average_precision_score(val["y_true"], val["score"]),
                "val_roc_auc": roc_auc_score(val["y_true"], val["score"]),
            })
    table = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False)
    print("Validation scores for each combination:")
    for r in table.itertuples():
        print(f"  PR-AUC {r.val_pr_auc:.3f}  ROC-AUC {r.val_roc_auc:.3f}  {' + '.join(r.members)}")

    best = list(table.iloc[0]["members"])
    if len(best) == 1:
        # A single model won; still build the full ensemble so it can be compared
        best = list(runs)
        print("A single model scored best on validation; ensemble uses all runs for comparison")

    average(runs, best).reset_index().to_csv(PREDICTIONS_DIR / "ensemble.csv", index=False)
    (MODELS_DIR / "ensemble.json").write_text(json.dumps({"members": best}, indent=2))
    print(f"Ensemble: {' + '.join(best)}")


if __name__ == "__main__":
    main()
