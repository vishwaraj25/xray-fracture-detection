"""
Exploratory data analysis of FracAtlas.

Run build_dataset.py first. Writes figures to reports/figures/ and a summary
of the key numbers to reports/eda_summary.md.
"""
import numpy as np
import pandas as pd
from PIL import Image

from config import CACHE_SIZE, FIGURES_DIR, IMAGES_DIR, METADATA_PATH, REGIONS, REPORTS_DIR, SEED
from plotting import FRACTURED, NOT_FRACTURED, TEXT_MUTED, plt, save


def fracture_rate_with_ci(df, column):
    """Fracture rate per category with a 95% Wilson confidence interval."""
    rows = []
    for value, grp in df.groupby(column):
        n, k = len(grp), grp["fractured"].sum()
        p, z = k / n, 1.96
        centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
        rows.append({column: value, "n": n, "fractured": k, "rate": p,
                     "lo": centre - half, "hi": centre + half})
    return pd.DataFrame(rows).sort_values("rate", ascending=False)


def plot_class_balance(df):
    counts = df["fractured"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.barh(["Not fractured", "Fractured"], counts.values,
                   color=[NOT_FRACTURED, FRACTURED], height=0.55)
    for bar, n in zip(bars, counts.values):
        ax.text(bar.get_width() + 40, bar.get_y() + bar.get_height() / 2,
                f"{n:,} ({n / len(df):.0%})", va="center", color=TEXT_MUTED)
    ax.set_xlim(0, counts.max() * 1.25)
    ax.set_title("Class balance: about 1 in 6 images is fractured")
    ax.grid(axis="y", visible=False)
    save(fig, FIGURES_DIR / "eda_class_balance.png")


def plot_rate_by(df, column, title, filename):
    rates = fracture_rate_with_ci(df, column)
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(rates) + 1.2))
    y = np.arange(len(rates))[::-1]
    ax.barh(y, rates["rate"], color=FRACTURED, height=0.55)
    ax.errorbar(rates["rate"], y, xerr=[rates["rate"] - rates["lo"], rates["hi"] - rates["rate"]],
                fmt="none", ecolor=TEXT_MUTED, elinewidth=1, capsize=3)
    ax.set_yticks(y, [f"{v}  (n={n:,})" for v, n in zip(rates[column], rates["n"])])
    overall = df["fractured"].mean()
    ax.axvline(overall, color=TEXT_MUTED, linestyle="--", linewidth=1)
    ax.text(overall, -0.75, f" overall {overall:.0%}", color=TEXT_MUTED, fontsize=9, va="bottom")
    ax.set_ylim(-0.8, len(rates) - 0.4)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Fracture rate (95% CI)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    save(fig, FIGURES_DIR / filename)
    return rates


def plot_image_sizes(df):
    """Image sizes are a handful of fixed values, so compare fracture rate by resolution."""
    sizes = df.groupby(["orig_width", "orig_height"]).size().sort_values(ascending=False)
    df["resolution"] = np.where(df[["orig_width", "orig_height"]].max(axis=1) > 1000,
                                "high-res (>1000px)", "low-res (≤1000px)")
    rates = plot_rate_by(df, "resolution", "High-resolution scans are fractured more often",
                         "eda_rate_by_resolution.png")
    return sizes, rates


def plot_intensity(df):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, column, label in [(axes[0], "mean_intensity", "Mean brightness"),
                              (axes[1], "std_intensity", "Contrast (std of pixels)")]:
        bins = np.linspace(df[column].min(), df[column].max(), 40)
        for value, color, name in [(0, NOT_FRACTURED, "Not fractured"), (1, FRACTURED, "Fractured")]:
            ax.hist(df.loc[df["fractured"] == value, column], bins=bins, density=True,
                    histtype="step", linewidth=2, color=color, label=name)
        ax.set_xlabel(label)
        ax.set_yticks([])
    axes[0].set_title("Brightness and contrast overlap almost completely")
    axes[1].legend()
    save(fig, FIGURES_DIR / "eda_intensity.png")


def plot_fracture_count(df):
    counts = df.loc[df["fractured"] == 1, "fracture_count"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(counts.index.astype(str), counts.values, color=FRACTURED, width=0.6)
    for x, n in zip(counts.index.astype(str), counts.values):
        ax.text(x, n + 8, str(n), ha="center", color=TEXT_MUTED)
    ax.set_xlabel("Annotated fractures in the image")
    ax.set_ylabel("Images")
    ax.set_title("Most fractured images have one fracture")
    ax.grid(axis="x", visible=False)
    save(fig, FIGURES_DIR / "eda_fracture_count.png")


def plot_sample_grid(df):
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2, len(REGIONS), figsize=(2.2 * len(REGIONS), 5))
    for col, region in enumerate(REGIONS):
        for row, label in enumerate([0, 1]):
            pool = df[(df["region"] == region) & (df["fractured"] == label)]
            ax = axes[row, col]
            ax.axis("off")
            if pool.empty:
                continue
            path = pool.iloc[rng.integers(len(pool))]["path"]
            ax.imshow(Image.open(IMAGES_DIR / path), cmap="gray")
            if row == 0:
                ax.set_title(region.capitalize(), loc="center")
        axes[0, 0].text(-0.08, 0.5, "Not fractured", transform=axes[0, 0].transAxes,
                        rotation=90, va="center", ha="right", color=NOT_FRACTURED)
        axes[1, 0].text(-0.08, 0.5, "Fractured", transform=axes[1, 0].transAxes,
                        rotation=90, va="center", ha="right", color=FRACTURED)
    save(fig, FIGURES_DIR / "eda_samples.png")


def plot_splits(df):
    table = pd.crosstab(df["split"], df["region"]).reindex(["train", "val", "test"])
    rates = df.groupby("split")["fractured"].mean().reindex(["train", "val", "test"])
    return table, rates


def main():
    df = pd.read_csv(METADATA_PATH)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_class_balance(df)
    region_rates = plot_rate_by(df, "region", "Fracture rate differs a lot by body region",
                                "eda_rate_by_region.png")
    view_rates = plot_rate_by(df, "view", "Fracture rate by view", "eda_rate_by_view.png")
    df["hardware_label"] = df["hardware"].map({0: "no implant", 1: "metal implant"})
    hardware_rates = plot_rate_by(df, "hardware_label",
                                  "Images with metal implants are mostly fractured",
                                  "eda_rate_by_hardware.png")
    sizes, resolution_rates = plot_image_sizes(df)
    plot_intensity(df)
    plot_fracture_count(df)
    plot_sample_grid(df)
    split_regions, split_rates = plot_splits(df)

    dup_sizes = df["dup_group"].value_counts()
    dup_groups = df[df["dup_group"].isin(dup_sizes[dup_sizes > 1].index)].groupby("dup_group")
    mixed_label_groups = int((dup_groups["fractured"].nunique() > 1).sum())

    def md_rates(rates, column):
        lines = [f"| {column} | images | fractured | rate (95% CI) |", "|---|---|---|---|"]
        for r in rates.itertuples():
            lines.append(f"| {getattr(r, column)} | {r.n:,} | {r.fractured} | "
                         f"{r.rate:.1%} ({r.lo:.1%}–{r.hi:.1%}) |")
        return "\n".join(lines)

    top_sizes = ", ".join(f"{w}×{h} ({n:,})" for (w, h), n in sizes.head(3).items())
    summary = f"""# FracAtlas: exploratory data analysis

Generated by `src/eda.py`. Figures are in `reports/figures/`.

## Overview

- **{len(df):,} images**, **{df['fractured'].sum()} fractured ({df['fractured'].mean():.1%})**. The classes are imbalanced, so accuracy alone is misleading: predicting "not fractured" every time scores {1 - df['fractured'].mean():.1%}.
- {(df['fracture_count'] > 1).sum()} images have more than one annotated fracture.
- {int(df['hardware'].sum())} images show metal implants, and {int(df['multiscan'].sum())} contain several scans in one image.
- **Data quality:** {int(df['label_conflict'].sum())} images are stored in the `Non_fractured` folder but labelled fractured in `dataset.csv`. The CSV label is used. One JPEG is slightly truncated but still readable.
- Most common original sizes: {top_sizes}. Every image is downscaled once to at most {CACHE_SIZE}px on the longer side.

## Fracture rate by body region

{md_rates(region_rates, 'region')}

## Fracture rate by view

{md_rates(view_rates, 'view')}

## Image resolution

{md_rates(resolution_rates, 'resolution')}

The images come from a few devices and export settings. High-resolution scans are fractured more often,
so a model could partly learn the scanner instead of the bone. Every image is resized to the same input size before modelling.

## Metal implants

{md_rates(hardware_rates.rename(columns={'hardware_label': 'implant'}), 'implant')}

Metal implants are a strong shortcut: a model can learn "metal means fractured"
without looking at the bone. Results are therefore also reported separately for images without implants.

## Near-duplicates and leakage

FracAtlas has no patient IDs. Perceptual hashing finds **{(dup_sizes > 1).sum()} groups of near-identical images ({dup_sizes[dup_sizes > 1].sum()} images)**,
for example the same scan exported twice or the same limb before and after surgery.
Each group is kept inside one split. {mixed_label_groups} of these groups contain both labels,
which usually means a follow-up scan of the same patient.

## Train / validation / test split

Stratified by label and body region, and grouped by near-duplicate cluster.

| split | images | fracture rate | {' | '.join(split_regions.columns)} |
|---|---|---|{'---|' * len(split_regions.columns)}
""" + "\n".join(
        f"| {s} | {split_regions.loc[s].sum():,} | {split_rates[s]:.1%} | "
        + " | ".join(str(v) for v in split_regions.loc[s]) + " |"
        for s in split_regions.index
    ) + "\n"

    (REPORTS_DIR / "eda_summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
