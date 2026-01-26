import io
import json
from typing import List

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI(title="Plant Disease Classifier", version="1.0.0")

model: tf.keras.Model | None = None
class_names: List[str] = []


def load_resources() -> None:
    global model, class_names
    model = tf.keras.models.load_model("models/plant_disease_model.keras")
    with open("models/class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)


@app.on_event("startup")
def startup_event() -> None:
    load_resources()


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File is required")

    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="File must be an image (jpg or png)")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="File is empty")

    if model is None or not class_names:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    img_array = preprocess_image(file_bytes)
    predictions = model.predict(img_array)
    disease_idx = int(np.argmax(predictions[0]))
    disease = class_names[disease_idx]
    confidence = float(predictions[0][disease_idx] * 100)

    return JSONResponse({"disease": disease, "confidence": confidence})