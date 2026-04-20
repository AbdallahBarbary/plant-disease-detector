"""
app.py — Plant Disease Detection Flask App
==========================================
Run locally:  python app.py
Then open:    http://localhost:5000
"""

import os
import json
import uuid
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image

# ── CONFIG ─────────────────────────────────────────────────────────────────
IMG_SIZE        = 224
MODEL_PATH      = "model/plant_disease_model.h5"
NAMES_PATH      = "model/class_names.json"
UPLOAD_FOLDER   = "uploads"
ALLOWED_EXT     = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_MB  = 10

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]    = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── LOAD MODEL ONCE AT STARTUP ─────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(NAMES_PATH) as f:
    class_names = json.load(f)  # {str(index): "ClassName"}

print(f"✅ Model loaded — {len(class_names)} classes")

# ── DISEASE INFO MAP ───────────────────────────────────────────────────────
# Friendly display names + treatment tips per disease
# Keys match PlantVillage class name format (Plant___Disease)
DISEASE_INFO = {
    "healthy": {
        "display": "Healthy",
        "tip": "Your plant looks healthy! Keep up the good care.",
        "severity": "none",
    },
    "Apple___Apple_scab": {
        "display": "Apple Scab",
        "tip": "Remove infected leaves. Apply fungicide (captan or myclobutanil) in early spring.",
        "severity": "moderate",
    },
    "Apple___Black_rot": {
        "display": "Apple Black Rot",
        "tip": "Prune infected branches. Apply copper-based fungicide. Remove mummified fruit.",
        "severity": "high",
    },
    "Apple___Cedar_apple_rust": {
        "display": "Cedar Apple Rust",
        "tip": "Apply fungicide at bud break. Remove nearby juniper/cedar trees if possible.",
        "severity": "moderate",
    },
    "Tomato___Late_blight": {
        "display": "Tomato Late Blight",
        "tip": "Remove infected plants immediately. Apply copper fungicide. Avoid overhead watering.",
        "severity": "high",
    },
    "Tomato___Early_blight": {
        "display": "Tomato Early Blight",
        "tip": "Remove lower infected leaves. Apply chlorothalonil fungicide. Mulch around base.",
        "severity": "moderate",
    },
    "Tomato___healthy": {
        "display": "Healthy Tomato",
        "tip": "Plant is healthy! Ensure consistent watering and full sun exposure.",
        "severity": "none",
    },
    # Default fallback used for any class not explicitly listed
    "_default": {
        "display": None,  # will use class name
        "tip": "Consult a local agricultural expert for treatment options.",
        "severity": "unknown",
    },
}

def get_disease_info(class_name):
    """Return display info for a given class name."""
    if class_name in DISEASE_INFO:
        info = DISEASE_INFO[class_name].copy()
    else:
        info = DISEASE_INFO["_default"].copy()

    # Check if "healthy" is in name even if not in map
    if "healthy" in class_name.lower() and class_name not in DISEASE_INFO:
        info["severity"] = "none"
        info["tip"] = "Your plant looks healthy!"

    if info["display"] is None:
        # Format class name: "Tomato___Early_blight" → "Tomato: Early Blight"
        parts = class_name.split("___")
        if len(parts) == 2:
            info["display"] = f"{parts[0].replace('_', ' ')}: {parts[1].replace('_', ' ').title()}"
        else:
            info["display"] = class_name.replace("_", " ").title()

    return info

# ── HELPER FUNCTIONS ───────────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def preprocess_image(image_path):
    """Load image, resize to 224x224, normalize to [0,1], add batch dim."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # shape: (1, 224, 224, 3)

def predict(image_path):
    """Run inference. Returns top 3 predictions."""
    img_array = preprocess_image(image_path)
    preds = model.predict(img_array, verbose=0)[0]  # shape: (NUM_CLASSES,)

    top3_indices = np.argsort(preds)[::-1][:3]
    results = []
    for idx in top3_indices:
        class_name  = class_names[str(idx)]
        confidence  = float(preds[idx]) * 100
        info        = get_disease_info(class_name)
        results.append({
            "class_name":  class_name,
            "display":     info["display"],
            "confidence":  round(confidence, 2),
            "tip":         info["tip"],
            "severity":    info["severity"],
        })
    return results

# ── ROUTES ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload PNG, JPG, or JPEG."}), 400

    # Save with unique name to avoid collisions
    ext      = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        predictions = predict(filepath)
    except Exception as e:
        os.remove(filepath)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Clean up uploaded file after prediction
    os.remove(filepath)

    return jsonify({"predictions": predictions})

@app.route("/health")
def health():
    """Health check endpoint — useful for deployment."""
    return jsonify({"status": "ok", "classes": len(class_names)})

# ── MAIN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
