"""
api.py
------
FastAPI application exposing CropDoc inference endpoints.

Endpoints:
    POST /predict  — Upload a leaf image, receive disease prediction + treatment plan
    GET  /health   — Liveness check
    GET  /classes  — List all 38 supported disease classes
"""

import io
import json
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.backend.inference import predict


app = FastAPI(
    title="CropDoc API",
    description="Plant disease detection and treatment recommendation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve React build if present
FRONTEND_BUILD = Path("app/frontend/dist")
if FRONTEND_BUILD.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_BUILD / "assets")), name="assets")


@app.get("/health")
async def health() -> dict:
    """Liveness check."""
    return {"status": "ok", "timestamp": time.time()}


@app.get("/classes")
async def list_classes() -> dict:
    """Return all supported disease class names."""
    class_file = Path("models/class_names.json")
    if not class_file.exists():
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    classes = json.loads(class_file.read_text())
    return {"classes": classes, "count": len(classes)}


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)) -> JSONResponse:
    """
    Predict plant disease from an uploaded leaf image.

    Args:
        file: JPEG or PNG image upload.

    Returns:
        JSON with disease prediction, confidence, severity, and treatment plan.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG or PNG)")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    try:
        result = predict(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return JSONResponse(content=result)


# Serve React index.html for all non-API routes (SPA fallback)
@app.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    from fastapi.responses import FileResponse
    index = FRONTEND_BUILD / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"detail": "Frontend not built. See app/frontend/"}, status_code=404)
