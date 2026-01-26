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
    return res.json({"ok": True})