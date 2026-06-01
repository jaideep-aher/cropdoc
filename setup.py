"""
setup.py
--------
One-shot project setup: download data → extract features → train all models.

Usage:
    python setup.py                    # full pipeline
    python setup.py --skip-download    # skip if data already present
    python setup.py --model deep       # train only deep model
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    """Run a subprocess command, raising on non-zero exit."""
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="CropDoc full setup pipeline")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--model", default="all", choices=["naive", "classical", "deep", "all"])
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    if not args.skip_download:
        run([sys.executable, "scripts/make_dataset.py"])

    if not args.skip_features:
        run([sys.executable, "scripts/build_features.py"])

    run([sys.executable, "scripts/model.py", "--model", args.model, "--epochs", str(args.epochs)])

    print("\n✅ Setup complete. Run `python main.py` to start the inference server.")


if __name__ == "__main__":
    main()
