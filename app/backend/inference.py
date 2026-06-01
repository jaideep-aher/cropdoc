"""
inference.py
------------
Model loading and inference logic for the CropDoc API.
Downloads weights from HuggingFace Hub on first run, then caches locally.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

from app.backend.treatments import get_treatment, Treatment


HF_REPO = "jaideep-aher/cropdoc-efficientnetv2"
MODEL_FILENAME = "efficientnetv2_s_best.pth"
CLASS_FILENAME = "class_names.json"
LOCAL_CACHE = Path("models")
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PREPROCESS = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _resolve_path(filename: str) -> Path:
    """Return local model path, downloading from HF Hub if needed."""
    local = LOCAL_CACHE / filename
    if local.exists():
        return local
    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"[inference] Downloading {filename} from HF Hub ...")
    path = hf_hub_download(repo_id=HF_REPO, filename=filename, local_dir=str(LOCAL_CACHE))
    return Path(path)


@lru_cache(maxsize=1)
def _load_model():
    """Load EfficientNetV2-S weights (cached after first call)."""
    class_path = _resolve_path(CLASS_FILENAME)
    class_names = json.loads(class_path.read_text())
    num_classes = len(class_names)

    model = timm.create_model("efficientnetv2_s", pretrained=False, num_classes=num_classes)
    weights_path = _resolve_path(MODEL_FILENAME)
    state = torch.load(str(weights_path), map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model, class_names


def predict(image: Image.Image, top_k: int = 5) -> dict:
    """
    Run inference on a PIL image and return top-k predictions with treatments.

    Args:
        image: RGB PIL image.
        top_k: Number of top predictions to return.

    Returns:
        Dict with keys: top_prediction, confidence, severity, treatments, top_k_predictions.
    """
    model, class_names = _load_model()

    tensor = PREPROCESS(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    top_probs, top_indices = probs.topk(top_k)
    top_probs = top_probs.cpu().tolist()
    top_indices = top_indices.cpu().tolist()

    top_class = class_names[top_indices[0]]
    top_conf = top_probs[0]
    treatment = get_treatment(top_class)

    # Severity scoring: combine model confidence and database hint
    severity = _score_severity(top_conf, treatment.severity_hint)

    top_k_results = [
        {"class": class_names[i], "probability": round(p, 4)}
        for i, p in zip(top_indices, top_probs)
    ]

    return {
        "top_prediction": top_class,
        "display_name": _format_display_name(top_class),
        "confidence": round(top_conf, 4),
        "severity": severity,
        "is_healthy": "healthy" in top_class.lower(),
        "treatment": {
            "chemical_controls": treatment.chemical_controls,
            "biological_controls": treatment.biological_controls,
            "cultural_controls": treatment.cultural_controls,
            "prevention": treatment.prevention,
            "note": treatment.note,
        },
        "top_k_predictions": top_k_results,
    }


def _score_severity(confidence: float, db_hint: str) -> str:
    """
    Blend model confidence with disease severity database hint.

    Args:
        confidence: Model softmax confidence for top class.
        db_hint: Pre-set severity from treatment DB ('mild'|'moderate'|'severe').

    Returns:
        Final severity label.
    """
    hint_score = {"mild": 1, "moderate": 2, "severe": 3}.get(db_hint, 1)
    conf_score = 1 if confidence < 0.5 else (2 if confidence < 0.85 else 3)
    combined = round((hint_score + conf_score) / 2)
    return {1: "mild", 2: "moderate", 3: "severe"}[combined]


def _format_display_name(class_name: str) -> str:
    """Convert 'Tomato___Early_blight' → 'Tomato — Early Blight'."""
    parts = class_name.split("___")
    crop = parts[0].replace("_", " ")
    if len(parts) > 1:
        disease = parts[1].replace("_", " ").title()
        return f"{crop} — {disease}"
    return crop
