"""
build_features.py
-----------------
Extracts HOG + color histogram features from raw images for the classical ML pipeline.
Outputs: data/processed/X_{split}.npy, data/processed/y_{split}.npy

Attribution: scikit-image HOG implementation used as documented at
https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Tuple

import cv2
from skimage.feature import hog
from tqdm import tqdm


RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
IMG_SIZE = (128, 128)
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    channel_axis=-1,
)
COLOR_BINS = 32


def extract_features(img_path: Path) -> np.ndarray:
    """
    Extract a feature vector combining HOG and HSV color histogram.

    Args:
        img_path: Path to image file.

    Returns:
        1-D feature vector (float32).
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return np.zeros(1, dtype=np.float32)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, IMG_SIZE)

    # HOG on RGB
    hog_feat = hog(img_rgb, **HOG_PARAMS).astype(np.float32)

    # HSV color histogram
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hist_feats = []
    for ch in range(3):
        hist, _ = np.histogram(img_hsv[:, :, ch], bins=COLOR_BINS, density=True)
        hist_feats.append(hist.astype(np.float32))
    color_feat = np.concatenate(hist_feats)

    return np.concatenate([hog_feat, color_feat])


def build_split(split: str, class_names: list[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix and label array for one data split.

    Args:
        split: One of 'train', 'val', 'test'.
        class_names: Ordered list of class label strings.

    Returns:
        Tuple of (X, y) numpy arrays.
    """
    split_dir = RAW_DIR / split
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    X, y = [], []

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label_idx = class_to_idx.get(class_dir.name, -1)
        if label_idx == -1:
            continue
        img_paths = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        for img_path in tqdm(img_paths, desc=f"{split}/{class_dir.name}", leave=False):
            feat = extract_features(img_path)
            X.append(feat)
            y.append(label_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main(splits: list[str] = ("train", "val", "test")) -> None:
    """Extract and save features for all specified splits."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    class_names = (RAW_DIR / "classes.txt").read_text().splitlines()

    for split in splits:
        print(f"[build_features] Processing {split} ...")
        X, y = build_split(split, class_names)
        np.save(PROC_DIR / f"X_{split}.npy", X)
        np.save(PROC_DIR / f"y_{split}.npy", y)
        print(f"[build_features] {split}: X={X.shape}, y={y.shape}")

    print("[build_features] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build HOG+color features from raw images")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()
    main(args.splits)
