"""
Run the whole project, one stage after another.

    python src/run_pipeline.py                 # everything (about 1.5 hours on an M2)
    python src/run_pipeline.py --skip-cnn      # reuse the trained CNN (a few minutes)
    python src/run_pipeline.py --from evaluate # restart from a stage

Each stage is a script in src/ that can also be run on its own.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent

STAGES = [
    ("build_dataset", "cache images, find near-duplicates, split train/val/test"),
    ("eda", "data analysis -> reports/eda_summary.md"),
    ("train_svm", "HOG + LBP features -> SVM baseline"),
    ("train_cnn", "fine-tune ResNet-18 at 384 px"),
    ("evaluate", "compare SVM and CNN on the test split -> reports/results.md"),
    ("explain", "Grad-CAM vs fracture boxes -> reports/gradcam.md"),
]


def main():
    names = [name for name, _ in STAGES]
    parser = argparse.ArgumentParser(description="Run the fracture detection pipeline")
    parser.add_argument("--from", dest="start", choices=names, default=names[0], help="first stage to run")
    parser.add_argument("--skip-cnn", action="store_true", help="skip CNN training (uses models/cnn.pt)")
    args = parser.parse_args()

    for name, description in STAGES[names.index(args.start):]:
        if name == "train_cnn" and args.skip_cnn:
            print(f"\n== {name}: skipped")
            continue
        print(f"\n== {name}: {description}", flush=True)
        start = time.time()
        subprocess.run([sys.executable, str(SRC / f"{name}.py")], check=True)
        print(f"== {name} done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
