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

## Second dataset: GRAZPEDWRI-DX

**GRAZPEDWRI-DX** (Nagy et al., *Scientific Data* 9, 222, 2022): paediatric wrist X-rays from Graz, Austria.
License: CC BY 4.0. The images come in four zips of about 4 GB each; this project uses part 1 (5,031 labelled X-rays).

```bash
mkdir -p data/raw/GRAZPEDWRI-DX && cd data/raw/GRAZPEDWRI-DX
curl -L -o dataset.csv https://ndownloader.figshare.com/files/35026432
curl -L -o images_part1.zip https://ndownloader.figshare.com/files/34268828   # 4.0 GB
```

`python src/graz.py --prepare` reads the zip without extracting it and writes:

- `processed/graz/`: 8-bit copies of the 16-bit images, padded to a square (about 320 MB)
- `processed/graz_metadata.csv`: labels, cast/metal flags and a 60/20/20 train/val/test split by patient

After that the zip is no longer needed and can be deleted. The train and validation patients are used by
`train_cnn.py --with-graz`; the test patients only by `graz.py`.

Git ignores everything in `raw/` and `processed/`.
