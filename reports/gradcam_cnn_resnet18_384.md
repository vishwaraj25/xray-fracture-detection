# Grad-CAM: cnn_resnet18_384

Pointing game on test fractures the model correctly flagged: how often the Grad-CAM peak lies on an annotated fracture box. The CAM grid is coarse (32 px cells on a 384 px input), so both a strict and a one-cell tolerance are shown.

| images | n | peak within 14 px | peak within one CAM cell (32 px) |
|---|---|---|---|
| all | 84 | 52% | 69% |
| no implant | 71 | 49% | 69% |
| metal implant | 13 | 69% | 69% |

![Grad-CAM examples](figures/gradcam_cnn_resnet18_384.png)
