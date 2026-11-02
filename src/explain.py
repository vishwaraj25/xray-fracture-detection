"""
Grad-CAM explanations for a trained CNN, checked against FracAtlas fracture boxes.

  python src/explain.py                      # cnn_resnet18
  python src/explain.py --run cnn_resnet18_384

Writes:
  - reports/figures/gradcam_<run>.png: heatmaps with the radiologist's boxes
  - reports/gradcam_<run>.md: a "pointing game" score. For fractured test images
    the model flags, how often the hottest Grad-CAM pixel falls on an annotated
    fracture box. High = the model looks at the fracture; low = it relies on something else.
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import FIGURES_DIR, METADATA_PATH, PREDICTIONS_DIR, RAW_DIR, REPORTS_DIR
from features import load_square
from plotting import FRACTURED, TEXT_MUTED, plt, save

TARGET_LAYERS = {
    "resnet18": lambda m: m.layer4,
    "resnet50": lambda m: m.layer4,
    "densenet121": lambda m: m.features,
    "efficientnet_b0": lambda m: m.features,
}

# The last conv block of these networks has stride 32
CAM_CELL = 32
# Strict pointing-game tolerance, in pixels of a 224 px input (scaled for other sizes)
STRICT_MARGIN_224 = 8


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.activations = None
        self.gradients = None
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, image):
        """Heatmap at the input's resolution (normalised to 0..1) and the sigmoid score."""
        self.model.zero_grad()
        logit = self.model(image.unsqueeze(0))
        logit.sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=tuple(image.shape[-2:]), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        return cam / (cam.max() + 1e-8), torch.sigmoid(logit).item()


def fracture_boxes(image_id, original_shape, square_size, input_size):
    """YOLO boxes (normalised to the original image) mapped into the square model input."""
    path = RAW_DIR / "Annotations" / "YOLO" / image_id.replace(".jpg", ".txt")
    if not path.exists():
        return []
    h, w = original_shape
    scale = square_size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    left, top = (square_size - new_w) // 2, (square_size - new_h) // 2
    to_input = input_size / square_size
    boxes = []
    for line in path.read_text().split("\n"):
        if not line.strip():
            continue
        _, cx, cy, bw, bh = map(float, line.split())
        x0 = (left + (cx - bw / 2) * new_w) * to_input
        y0 = (top + (cy - bh / 2) * new_h) * to_input
        boxes.append((x0, y0, bw * new_w * to_input, bh * new_h * to_input))
    return boxes


def hit(cam, boxes, margin):
    """Pointing game: is the CAM peak inside (or within `margin` px of) any box?"""
    y, x = np.unravel_index(cam.argmax(), cam.shape)
    return any(bx - margin <= x <= bx + bw + margin and by - margin <= y <= by + bh + margin
               for bx, by, bw, bh in boxes)


def main():
    from evaluate import threshold_for_sensitivity
    from train_cnn import eval_transform, get_device, image_path, load_checkpoint, load_size, to_tensor

    parser = argparse.ArgumentParser(description="Grad-CAM for a trained fracture CNN")
    parser.add_argument("--run", default="cnn_resnet18", help="checkpoint name in models/")
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    device = get_device()
    model, arch, input_size = load_checkpoint(args.run, device)
    square_size = load_size(input_size)
    transform = eval_transform(input_size)
    cam = GradCAM(model, TARGET_LAYERS[arch](model))
    strict = STRICT_MARGIN_224 * input_size / 224

    meta = pd.read_csv(METADATA_PATH).set_index("image_id")
    preds = pd.read_csv(PREDICTIONS_DIR / f"{args.run}.csv")
    val = preds[preds["split"] == "val"]
    threshold = threshold_for_sensitivity(val["y_true"].to_numpy(), val["score"].to_numpy())

    flagged = preds[(preds["split"] == "test") & (preds["y_true"] == 1) & (preds["score"] >= threshold)]
    results = []
    for image_id in flagged["image_id"]:
        row = meta.loc[image_id]
        gray = load_square(image_path({**row, "image_id": image_id}, square_size), square_size)
        tensor = transform(to_tensor(gray)).to(device)
        heatmap, score = cam(tensor)
        boxes = fracture_boxes(image_id, (row["orig_height"], row["orig_width"]), square_size, input_size)
        if boxes:
            results.append({"image_id": image_id, "score": score, "hit": hit(heatmap, boxes, strict),
                            "hit_cell": hit(heatmap, boxes, CAM_CELL),
                            "hardware": row["hardware"], "dup_group": row["dup_group"],
                            "heatmap": heatmap, "boxes": boxes,
                            "image": np.asarray(transform.transforms[0](to_tensor(gray))[0])})

    table = pd.DataFrame(results)
    summary = {
        "all": table,
        "no implant": table[table["hardware"] == 0],
        "metal implant": table[table["hardware"] == 1],
    }
    lines = [f"| images | n | peak within {strict:.0f} px | peak within one CAM cell ({CAM_CELL} px) |",
             "|---|---|---|---|"]
    for label, part in summary.items():
        lines.append(f"| {label} | {len(part)} | {part['hit'].mean():.0%} | {part['hit_cell'].mean():.0%} |")
    print("\n".join(lines))

    # Show the most confident hits and misses, one image per near-duplicate group
    unique = table.drop_duplicates("dup_group").to_dict("records")
    hits_list = sorted([r for r in unique if r["hit_cell"]], key=lambda r: -r["score"])
    miss_list = sorted([r for r in unique if not r["hit_cell"]], key=lambda r: -r["score"])
    half = args.examples // 2
    chosen = hits_list[:half] + miss_list[:args.examples - min(half, len(hits_list))]
    chosen = chosen[:args.examples]
    cols = 4
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
    for ax, r in zip(np.atleast_1d(axes).ravel(), chosen):
        ax.imshow(r["image"], cmap="gray")
        ax.imshow(r["heatmap"], cmap="inferno", alpha=0.45)
        for bx, by, bw, bh in r["boxes"]:
            ax.add_patch(plt.Rectangle((bx, by), bw, bh, fill=False, edgecolor="#1baf7a", linewidth=2))
        verdict = "on fracture" if r["hit_cell"] else "elsewhere"
        implant = " (implant)" if r["hardware"] else ""
        ax.set_title(f"p={r['score']:.2f}{implant}\nfocus {verdict}", fontsize=9, loc="center",
                     color=FRACTURED if not r["hit_cell"] else TEXT_MUTED)
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    fig.suptitle("Grad-CAM (heat) vs annotated fracture box (green)", x=0.02, ha="left",
                 fontsize=12, fontweight="semibold")
    figure = f"gradcam_{args.run}.png"
    save(fig, FIGURES_DIR / figure)

    (REPORTS_DIR / f"gradcam_{args.run}.md").write_text(
        f"# Grad-CAM: {args.run}\n\n"
        "Pointing game on test fractures the model correctly flagged: how often the Grad-CAM peak "
        f"lies on an annotated fracture box. The CAM grid is coarse ({CAM_CELL} px cells on a "
        f"{input_size} px input), so both a strict and a one-cell tolerance are shown.\n\n"
        + "\n".join(lines) + "\n\n"
        f"![Grad-CAM examples](figures/{figure})\n"
    )


if __name__ == "__main__":
    main()
