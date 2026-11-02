# Data

This project uses **FracAtlas** (Abedeen et al., *Scientific Data* 10, 521, 2023):
4,083 musculoskeletal X-rays (hand, leg, hip, shoulder), 719 of them fractured, with
fracture bounding boxes. License: CC BY 4.0.

Download (~340 MB) from Figshare: https://doi.org/10.6084/m9.figshare.22363012

```bash
mkdir -p data/raw && cd data/raw
curl -L -o FracAtlas.zip https://ndownloader.figshare.com/files/65518038
unzip -q FracAtlas.zip && rm FracAtlas.zip
```

Layout after extraction and `python src/build_dataset.py`:

- `raw/FracAtlas/`: the original dataset (images, `dataset.csv`, annotations)
- `processed/images/`: grayscale copies downscaled to at most 384 px
- `processed/metadata.csv`: labels, body region, view, image statistics, near-duplicate group and train/val/test split
- `processed/features/`: cached hand-crafted features

Git ignores everything in `raw/` and `processed/`.
