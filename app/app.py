"""
================================================
 MNIST Digit Recognizer — Flask Web Application
================================================
Run with:
    python app/app.py

Then open:  http://127.0.0.1:5000
"""

import os
import io
import base64
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "mnist_cnn.h5")

app = Flask(__name__)

# ── Load model once at startup (not on every request) ─────────────────────────
print("Loading model …", end=" ", flush=True)
model = tf.keras.models.load_model(MODEL_PATH)
print("done ✓")


# =============================================================================
# HELPERS
# =============================================================================

def preprocess_canvas(data_url: str) -> np.ndarray:
    """
    Convert a base-64 canvas PNG → 28×28 float32 array ready for the model.

    Steps
    -----
    1. Strip the data-URL header and decode bytes.
    2. Open with Pillow; convert to grayscale.
    3. Invert so the digit is WHITE on BLACK (matches MNIST convention).
    4. Resize to 28×28 with high-quality resampling.
    5. Normalise to [0, 1] and add batch + channel dims.
    """
    # 1 — decode base64
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)

    # 2 — open and convert to greyscale
    img = Image.open(io.BytesIO(img_bytes)).convert("L")

    # 3 — invert (canvas is black bg + white stroke; MNIST is white digit + black bg)
    img = ImageOps.invert(img)

    # 4 — resize to 28×28
    img = img.resize((28, 28), Image.LANCZOS)

    # 5 — normalise, reshape → (1, 28, 28, 1)
    arr = np.array(img, dtype="float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr


# =============================================================================
# ROUTES
# =============================================================================

@app.route("/")
def index():
    """Serve the main drawing page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept a JSON body with { "image": "<data-url>" },
    run inference, and return { "digit": int, "confidence": float,
                                "probabilities": [float×10] }.
    """
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data received"}), 400

    try:
        arr = preprocess_canvas(data["image"])
        probs = model.predict(arr, verbose=0)[0]          # shape (10,)
        digit = int(np.argmax(probs))
        confidence = float(probs[digit]) * 100

        return jsonify({
            "digit":         digit,
            "confidence":    round(confidence, 2),
            "probabilities": [round(float(p) * 100, 2) for p in probs],
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
