"""
Build the static web demo in docs/: the CNN as an ONNX model, the example X-rays and the test metrics.

    python src/export_web.py

The demo runs entirely in the visitor's browser (onnxruntime-web), so it can be hosted for free
on GitHub Pages. This needs the FracAtlas download, models/cnn.pt and the test predictions. Writes:
  docs/model.onnx        the CNN, weights stored in half precision to halve the download
  docs/samples/*.jpg     held-out example X-rays, picked to cover every outcome
  docs/data.json         the examples with their radiologist boxes, test scores and model facts
"""
import json
import shutil

import numpy as np
import onnx
import pandas as pd
import torch
from onnx import TensorProto, helper, numpy_helper
from torch import nn

from config import BASE_DIR, CNN_IMAGE_SIZE, METADATA_PATH, PREDICTIONS_DIR, RAW_DIR, TARGET_SENSITIVITY
from evaluate import threshold_metrics, validation_threshold
from explain import CAM_CELL, annotated_boxes
from predict import LEVELS, LIKELY_SCORE, level_of
from sklearn.metrics import roc_auc_score
from train_cnn import IMAGENET_MEAN, IMAGENET_STD, load_model

WEB_DIR = BASE_DIR / "docs"
SAMPLE_PLAN = [("caught", 3), ("off target", 1), ("missed", 1), ("false alarm", 1), ("cleared", 2)]


class WebModel(nn.Module):
    """Grayscale 0-255 in; fracture logit and a Grad-CAM grid out.

    ResNet-18 ends in global average pooling and one linear layer, so Grad-CAM reduces to the last
    feature maps weighted by that layer's weights. Baking it in lets the browser skip backpropagation.
    """

    def __init__(self, model):
        super().__init__()
        self.m = model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, gray):
        m = self.m
        x = (gray.expand(-1, 3, -1, -1) / 255 - self.mean) / self.std
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        features = m.layer4(m.layer3(m.layer2(m.layer1(x))))
        logit = m.fc(features.mean(dim=(2, 3))).squeeze(1)
        cam = (features * m.fc.weight.view(1, -1, 1, 1)).sum(dim=1)
        return logit, cam


def halve_weights(model):
    """Store large float32 weights as float16 and cast them back at load time: half the file size."""
    graph = model.graph
    casts, renamed = [], {}
    for init in list(graph.initializer):
        if init.data_type == TensorProto.FLOAT and np.prod(init.dims) > 1024:
            array = numpy_helper.to_array(init)
            graph.initializer.remove(init)
            graph.initializer.append(numpy_helper.from_array(array.astype(np.float16), init.name + "_f16"))
            renamed[init.name] = init.name + "_f32"
            casts.append(helper.make_node("Cast", [init.name + "_f16"], [init.name + "_f32"], to=TensorProto.FLOAT))
    for node in graph.node:
        node.input[:] = [renamed.get(name, name) for name in node.input]
    nodes = casts + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    return model


def export_model(path):
    web = WebModel(load_model(torch.device("cpu"))).eval()
    dummy = torch.zeros(2, 1, CNN_IMAGE_SIZE, CNN_IMAGE_SIZE)
    tmp = path.with_suffix(".fp32.onnx")
    torch.onnx.export(web, dummy, str(tmp), input_names=["gray"], output_names=["logit", "cam"],
                      opset_version=17, dynamo=False)
    model = halve_weights(onnx.load(tmp))
    tmp.unlink()
    onnx.save(model, path, save_as_external_data=False)
    print(f"Saved {path.name}: {path.stat().st_size / 1e6:.1f} MB")
    return web


def outcome(row, threshold):
    flagged = row["score"] >= threshold
    if not row["fractured"]:
        return "false alarm" if flagged else "cleared"
    if not flagged:
        return "missed"
    # Grad-CAM pointing game from explain.py
    return "caught" if row["hit_cell"] else "off target"


def pick_samples(preds, threshold):
    """Held-out test X-rays chosen to cover each outcome, so the demo shows failures as well as successes."""
    hits = pd.read_csv(PREDICTIONS_DIR / "gradcam.csv")[["image_id", "hit_cell"]]
    df = pd.read_csv(METADATA_PATH).merge(preds[["image_id", "score"]], on="image_id")
    df = df.merge(hits, on="image_id", how="left")
    # Single-region X-rays without implants, one per near-duplicate group
    test = df[(df["split"] == "test") & (df["hardware"] == 0) & (df["region"] != "mixed")]
    test = test.drop_duplicates("dup_group").copy()
    test["outcome"] = test.apply(outcome, axis=1, threshold=threshold)

    picks = []
    for name, count in SAMPLE_PLAN:
        pool = test[test["outcome"] == name].sample(frac=1, random_state=7)
        # Prefer a different body region for each pick of the same outcome
        chosen = pool.drop_duplicates("region").head(count)
        picks += chosen.to_dict("records") + pool.drop(chosen.index).head(count - len(chosen)).to_dict("records")
    return picks


def export_data():
    preds = pd.read_csv(PREDICTIONS_DIR / "cnn.csv")
    threshold = float(validation_threshold(preds))
    test = preds[preds["split"] == "test"]
    metrics = threshold_metrics(test["y_true"].to_numpy(), test["score"].to_numpy(), threshold)

    shutil.rmtree(WEB_DIR / "samples", ignore_errors=True)
    (WEB_DIR / "samples").mkdir(parents=True)
    samples = []
    for row in pick_samples(preds, threshold):
        sample_id = row["image_id"].removesuffix(".jpg")
        shutil.copy(RAW_DIR / "images" / row["folder"] / row["image_id"], WEB_DIR / "samples" / f"{sample_id}.jpg")
        samples.append({
            "id": sample_id,
            "region": row["region"],
            "label": "Fractured" if row["fractured"] else "Not fractured",
            "outcome": row["outcome"],
            "boxes": [dict(zip("xywh", box)) for box in annotated_boxes(row["image_id"])],
        })

    # Out of 100 held-out X-rays at each level, how many really were fractured
    held_out = preds[preds["split"] != "train"]
    per_100 = {name: int(round(100 * g["y_true"].mean())) for name, g in
               held_out.groupby(held_out["score"].map(lambda v: level_of(v, threshold)))}
    scores = lambda label: sorted(round(float(s), 4) for s in test.loc[test["y_true"] == label, "score"])  # noqa: E731
    data = {
        "info": {
            "model": "ResNet-18, 384 px",
            "dataset": "FracAtlas + GRAZPEDWRI-DX",
            "threshold": threshold,
            "likely_score": LIKELY_SCORE,
            "target_sensitivity": TARGET_SENSITIVITY,
            "levels": per_100,
            "cam_cell": CAM_CELL / CNN_IMAGE_SIZE,
            "test": {"images": int(len(test)), "roc_auc": float(roc_auc_score(test["y_true"], test["score"])),
                     "sensitivity": float(metrics["sensitivity"]), "specificity": float(metrics["specificity"])},
        },
        "level_text": {name: {"finding": text[0], "note": text[1]} for name, text in LEVELS.items()},
        "performance": {"threshold": threshold, "roc_auc": float(roc_auc_score(test["y_true"], test["score"])),
                        "fractured": scores(1), "normal": scores(0)},
        "samples": samples,
    }
    (WEB_DIR / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    print(f"Saved {len(samples)} samples and data.json")


def main():
    WEB_DIR.mkdir(exist_ok=True)
    export_model(WEB_DIR / "model.onnx")
    export_data()


if __name__ == "__main__":
    main()
