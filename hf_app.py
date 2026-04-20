"""
hf_app.py — Hugging Face Spaces deployment (Gradio version)
============================================================
This is an ALTERNATIVE to the Flask app for deploying on Hugging Face Spaces.
Hugging Face Spaces supports Gradio natively — no server config needed.

TO DEPLOY ON HUGGING FACE SPACES:
1. Create account at huggingface.co
2. New Space → SDK: Gradio → Python 3.10
3. Upload these files:
     hf_app.py          (rename to app.py in the Space)
     model/plant_disease_model.h5
     model/class_names.json
     requirements.txt
4. HF will install deps and launch automatically.
5. Your public URL: https://huggingface.co/spaces/YOUR_USERNAME/plant-disease-detector

Run locally: python hf_app.py
"""

import json
import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image

# ── CONFIG ─────────────────────────────────────────────────────────────────
IMG_SIZE   = 224
MODEL_PATH = "model/plant_disease_model.h5"
NAMES_PATH = "model/class_names.json"

# ── LOAD MODEL ─────────────────────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(NAMES_PATH) as f:
    class_names = json.load(f)
print(f"✅ Loaded — {len(class_names)} classes")

# ── PREDICT FUNCTION ───────────────────────────────────────────────────────
def predict(image: Image.Image):
    """Called by Gradio with a PIL image. Returns dict of {label: confidence}."""
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    top5  = np.argsort(preds)[::-1][:5]

    results = {}
    for idx in top5:
        name  = class_names[str(idx)]
        # Format: "Tomato___Late_blight" → "Tomato: Late Blight"
        parts = name.split("___")
        label = f"{parts[0].replace('_',' ')}: {parts[1].replace('_',' ').title()}" if len(parts)==2 else name
        results[label] = float(preds[idx])

    return results

# ── GRADIO UI ──────────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a plant leaf image"),
    outputs=gr.Label(num_top_classes=5, label="Disease Prediction"),
    title="🌿 Plant Disease Detector",
    description=(
        "Upload a leaf photo to detect plant diseases using a MobileNetV2 CNN "
        "trained on the PlantVillage dataset (38 disease classes). "
        "Built by **Abdallah Elbarbary** · MIU Computer Science — AI"
    ),
    examples=[],  # Add example image paths here after training
    theme=gr.themes.Soft(primary_hue="green"),
)

if __name__ == "__main__":
    demo.launch()
