import io
import json
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from PIL import Image

# Appwrite Functions: expose a `main(req, res)` handler. Everything else is module-level
# to take advantage of cold-start caching across invocations.

model: tf.keras.Model | None = None
class_names: List[str] = []


def load_resources() -> None:
    """Load the TensorFlow model and class labels once per cold start."""
    global model, class_names
    if model is None:
        model = tf.keras.models.load_model("models/plant_disease_model.keras")
    if not class_names:
        with open("models/class_names.json", "r", encoding="utf-8") as f:
            class_names.extend(json.load(f))


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Resize and normalize the uploaded image for the model."""
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)


def build_response(status_code: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"statusCode": status_code, "headers": {"Content-Type": "application/json"}, "body": json.dumps(payload)}


def main(req, res):  # Appwrite entrypoint
    """Handle image classification requests.

    Expected usage: POST raw image bytes with Content-Type image/jpeg or image/png.
    Multipart may work if req.body contains the file bytes only; otherwise, prefer raw bytes.
    """

    try:
        load_resources()
    except Exception:
        return res.json(build_response(500, {"error": "Failed to load model"}))

    content_type = getattr(req, "content_type", "") or req.headers.get("content-type", "")
    if "image" not in content_type:
        return res.json(build_response(400, {"error": "Content-Type must be image/jpeg or image/png"}))

    body = getattr(req, "body", None)
    if body in (None, b""):
        return res.json(build_response(400, {"error": "Request body is empty"}))

    # If Appwrite passes body as str, convert to bytes.
    if isinstance(body, str):
        body = body.encode()

    try:
        img_array = preprocess_image(body)
        predictions = model.predict(img_array)
        disease_idx = int(np.argmax(predictions[0]))
        disease = class_names[disease_idx]
        confidence = float(predictions[0][disease_idx] * 100)
    except Exception:
        return res.json(build_response(400, {"error": "Could not process image"}))

    return res.json(build_response(200, {"disease": disease, "confidence": confidence}))