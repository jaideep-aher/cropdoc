"""
model.py
--------
Train and evaluate all three modeling approaches:
  1. Naive baseline   — majority-class per crop group
  2. Classical ML     — HOG+color features → RandomForest
  3. Deep Learning    — EfficientNetV2-S fine-tuned via timm

Weights are pushed to HuggingFace Hub after training.
Usage:
    python scripts/model.py --model naive
    python scripts/model.py --model classical
    python scripts/model.py --model deep
    python scripts/model.py --model all
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns


PROC_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
MODELS_DIR = Path("models")
OUT_DIR = Path("data/outputs")
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 15
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_class_names() -> list[str]:
    """Load ordered class names from disk."""
    return (RAW_DIR / "classes.txt").read_text().splitlines()


def save_metrics(name: str, metrics: dict) -> None:
    """Persist evaluation metrics as JSON."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}_metrics.json"
    path.write_text(json.dumps(metrics, indent=2))
    print(f"[model] Metrics saved → {path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], name: str) -> None:
    """Save a confusion matrix PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(cm, annot=False, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {name}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}_cm.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[model] Confusion matrix saved → {OUT_DIR / name}_cm.png")


# ---------------------------------------------------------------------------
# 1. Naive Baseline
# ---------------------------------------------------------------------------

class NaiveBaseline:
    """
    Majority-class predictor within each crop group.

    For each input, predicts the most common disease label among all training
    samples that share the same crop (prefix before '__' in class name).
    """

    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names
        self.crop_majority: dict[str, int] = {}

    def fit(self, y_train: np.ndarray) -> "NaiveBaseline":
        """Compute majority class per crop from training labels."""
        # Group labels by crop prefix
        crop_counts: dict[str, dict[int, int]] = {}
        for label_idx in y_train:
            crop = self.class_names[label_idx].split("___")[0]
            crop_counts.setdefault(crop, {})
            crop_counts[crop][label_idx] = crop_counts[crop].get(label_idx, 0) + 1

        for crop, counts in crop_counts.items():
            self.crop_majority[crop] = max(counts, key=counts.get)

        # Global majority as fallback
        unique, cnts = np.unique(y_train, return_counts=True)
        self.global_majority = int(unique[np.argmax(cnts)])
        return self

    def predict(self, y_labels: np.ndarray) -> np.ndarray:
        """
        Predict majority class per crop for each sample.

        Args:
            y_labels: True label indices (used only to look up crop group).

        Returns:
            Predicted label indices.
        """
        preds = []
        for label_idx in y_labels:
            crop = self.class_names[label_idx].split("___")[0]
            preds.append(self.crop_majority.get(crop, self.global_majority))
        return np.array(preds)


def train_naive(class_names: list[str]) -> None:
    """Train and evaluate the naive baseline model."""
    print("\n[model] === Naive Baseline ===")
    y_train = np.load(PROC_DIR / "y_train.npy")
    y_val = np.load(PROC_DIR / "y_val.npy")

    baseline = NaiveBaseline(class_names).fit(y_train)
    y_pred = baseline.predict(y_val)

    acc = accuracy_score(y_val, y_pred)
    print(f"[model] Naive val accuracy: {acc:.4f}")
    save_metrics("naive", {"val_accuracy": acc})
    plot_confusion_matrix(y_val, y_pred, class_names, "naive")
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(baseline, MODELS_DIR / "naive_baseline.pkl")


# ---------------------------------------------------------------------------
# 2. Classical ML — Random Forest
# ---------------------------------------------------------------------------

def train_classical(class_names: list[str]) -> None:
    """Train and evaluate a Random Forest on HOG+color features."""
    print("\n[model] === Classical ML (Random Forest) ===")
    X_train = np.load(PROC_DIR / "X_train.npy")
    y_train = np.load(PROC_DIR / "y_train.npy")
    X_val = np.load(PROC_DIR / "X_val.npy")
    y_val = np.load(PROC_DIR / "y_val.npy")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )),
    ])

    t0 = time.time()
    pipe.fit(X_train, y_train)
    print(f"[model] Training time: {time.time() - t0:.1f}s")

    y_pred = pipe.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"[model] Classical val accuracy: {acc:.4f}")
    print(classification_report(y_val, y_pred, target_names=class_names, zero_division=0))

    save_metrics("classical", {"val_accuracy": acc})
    plot_confusion_matrix(y_val, y_pred, class_names, "classical")
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / "classical_rf.pkl")


# ---------------------------------------------------------------------------
# 3. Deep Learning — EfficientNetV2-S
# ---------------------------------------------------------------------------

def get_transforms(split: str):
    """Return train or eval torchvision transforms."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def train_deep(class_names: list[str], epochs: int = EPOCHS, push_to_hub: bool = True) -> None:
    """Fine-tune EfficientNetV2-S on PlantVillage."""
    print(f"\n[model] === Deep Learning (EfficientNetV2-S) on {DEVICE} ===")
    num_classes = len(class_names)

    train_ds = datasets.ImageFolder(str(RAW_DIR / "train"), transform=get_transforms("train"))
    val_ds = datasets.ImageFolder(str(RAW_DIR / "val"), transform=get_transforms("val"))
    test_ds = datasets.ImageFolder(str(RAW_DIR / "test"), transform=get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Load pre-trained model
    model = timm.create_model("efficientnetv2_s", pretrained=True, num_classes=num_classes)
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    MODELS_DIR.mkdir(exist_ok=True)

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        train_acc = correct / total

        # --- Validate ---
        val_acc = _evaluate(model, val_loader)
        scheduler.step()

        print(f"[model] Epoch {epoch:02d}/{epochs} | "
              f"loss={total_loss/total:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODELS_DIR / "efficientnetv2_s_best.pth")
            print(f"[model]   ✓ New best model saved (val_acc={val_acc:.4f})")

    # --- Test evaluation ---
    model.load_state_dict(torch.load(MODELS_DIR / "efficientnetv2_s_best.pth", map_location=DEVICE))
    test_acc, y_true, y_pred = _evaluate_with_preds(model, test_loader)
    print(f"\n[model] Test accuracy: {test_acc:.4f}")

    save_metrics("deep", {"best_val_accuracy": best_val_acc, "test_accuracy": test_acc})
    plot_confusion_matrix(np.array(y_true), np.array(y_pred), class_names, "deep")

    # Save class mapping alongside model
    (MODELS_DIR / "class_names.json").write_text(json.dumps(class_names))

    if push_to_hub:
        _push_to_hub()


def _evaluate(model: nn.Module, loader: DataLoader) -> float:
    """Return accuracy over a DataLoader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return correct / total


def _evaluate_with_preds(model: nn.Module, loader: DataLoader):
    """Return accuracy and raw predictions."""
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs).argmax(1).cpu().tolist()
            all_true.extend(labels.tolist())
            all_pred.extend(preds)
    acc = accuracy_score(all_true, all_pred)
    return acc, all_true, all_pred


def _push_to_hub() -> None:
    """Push best model weights + class names to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        repo_id = "jaideep-aher/cropdoc-efficientnetv2"
        api.create_repo(repo_id, exist_ok=True)
        api.upload_folder(folder_path=str(MODELS_DIR), repo_id=repo_id, ignore_patterns=["*.pkl"])
        print(f"[model] Model pushed to HF Hub: {repo_id}")
    except Exception as e:
        print(f"[model] HF Hub push skipped: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CropDoc models")
    parser.add_argument("--model", choices=["naive", "classical", "deep", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--no-hub", action="store_true", help="Skip HuggingFace Hub push")
    args = parser.parse_args()

    class_names = load_class_names()

    if args.model in ("naive", "all"):
        train_naive(class_names)
    if args.model in ("classical", "all"):
        train_classical(class_names)
    if args.model in ("deep", "all"):
        train_deep(class_names, epochs=args.epochs, push_to_hub=not args.no_hub)
