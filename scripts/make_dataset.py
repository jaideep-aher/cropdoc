"""
make_dataset.py
---------------
Downloads PlantVillage from HuggingFace Hub and creates train/val/test splits
saved as image directories under data/raw/.

Attribution: Dataset from PlantVillage (Hughes & Salathé, 2015).
HuggingFace mirror: https://huggingface.co/datasets/HanochSkandera/PlantVillage
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm
from PIL import Image


RAW_DIR = Path("data/raw")
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}


def download_and_split(
    dataset_name: str = "HanochSkandera/PlantVillage",
    output_dir: Path = RAW_DIR,
    seed: int = 42,
) -> None:
    """
    Download PlantVillage and write train/val/test image folders.

    Args:
        dataset_name: HuggingFace dataset identifier.
        output_dir: Root directory for output splits.
        seed: Random seed for reproducible splits.
    """
    print(f"[make_dataset] Downloading {dataset_name} ...")
    ds = load_dataset(dataset_name, split="train")

    # Split
    ds_train_val = ds.train_test_split(test_size=SPLITS["test"], seed=seed)
    ds_train = ds_train_val["train"].train_test_split(
        test_size=SPLITS["val"] / (SPLITS["train"] + SPLITS["val"]), seed=seed
    )

    splits = {
        "train": ds_train["train"],
        "val": ds_train["test"],
        "test": ds_train_val["test"],
    }

    label_names = ds.features["label"].names

    for split_name, split_ds in splits.items():
        print(f"[make_dataset] Writing {split_name} ({len(split_ds)} images) ...")
        for item in tqdm(split_ds, desc=split_name):
            label = label_names[item["label"]]
            dest_dir = output_dir / split_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            img: Image.Image = item["image"]
            # Use a hash of the image bytes as filename for reproducibility
            img_path = dest_dir / f"{hash(img.tobytes()) & 0xFFFFFFFF}.jpg"
            if not img_path.exists():
                img.save(img_path, "JPEG", quality=95)

    print("[make_dataset] Done.")
    _write_class_list(label_names, output_dir)


def _write_class_list(label_names: list[str], output_dir: Path) -> None:
    """Persist class names for use by other scripts."""
    class_file = output_dir / "classes.txt"
    class_file.write_text("\n".join(label_names))
    print(f"[make_dataset] Class list written to {class_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and split PlantVillage dataset")
    parser.add_argument("--dataset", default="HanochSkandera/PlantVillage")
    parser.add_argument("--output_dir", default="data/raw", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    download_and_split(args.dataset, args.output_dir, args.seed)
