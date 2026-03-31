from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import os
from PIL import Image
import io
import base64
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
CORS(app)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH = "aromatic_model_optimized.keras"   # your saved model filename
IMG_SIZE   = (160, 160)                          # must match training target_size

# ─── CLASS NAMES ──────────────────────────────────────────────────────────────
# These must be in the SAME ORDER as train_data.class_indices from your training.
# To find the exact order, run this in your Colab notebook:
#   print(train_data.class_indices)
# It prints something like: {'Basil': 0, 'CurryLeaf': 1, 'Mint': 2, ...}
# List them here sorted by their index value (0, 1, 2, ...).
CLASS_NAMES = [
    "Aloevera",                   # index 0
    "Amruthaballi",               # index 1
    "Betel",                      # index 2
    "Bhrami",                     # index 3
    "Bringaraja",                 # index 4
    "Citron lime (herelikai)",    # index 5
    "Coffee",                     # index 6
    "Common rue (naagdalli)",     # index 7
    "Coriender",                  # index 8
    "Curry",                      # index 9
    "Doddpathre",                 # index 10
    "Eucalyptus",                 # index 11
    "Ginger",                     # index 12
    "Henna",                      # index 13
    "Hibiscus",                   # index 14
    "Jasmine",                    # index 15
    "Lemon",                      # index 16
    "Lemongrass",                 # index 17
    "Malabar Nut",                # index 18
    "Marigold",                   # index 19
    "Mint",                       # index 20
    "Neem",                       # index 21
    "Nelavembu",                  # index 22
    "Onion",                      # index 23
    "Parijatha",                  # index 24
    "Pepper",                     # index 25
    "Rose",                       # index 26
    "Sampige",                    # index 27
    "Tulsi",                      # index 28
    "Turmeric",                   # index 29
    "camphor",                    # index 30
    "kamakasturi",                # index 31
]

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
model = None

def load_model():
    global model
    try:
        from tensorflow import keras

        model_path = os.path.join(os.getcwd(), MODEL_PATH)

        if not os.path.exists(model_path):
            print(f"❌ Model file not found at: {model_path}")
            return

        model = keras.models.load_model(model_path)
        print("✅ Model loaded successfully.")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")

# ✅ LOAD MODEL WHEN APP STARTS (IMPORTANT FOR RENDER)
load_model()

# ─── PREPROCESS ───────────────────────────────────────────────────────────────
# Uses MobileNetV2's preprocess_input — MUST match what was used during training.
# This scales pixels to [-1, 1], NOT the simple /255.0 normalization.
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)              # MobileNetV2 preprocessing → [-1, 1]
    return np.expand_dims(arr, axis=0)       # shape: (1, 160, 160, 3)

# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Ensure model is loaded
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 503

    # Ensure image is provided
    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "No image provided."}), 400

    file = request.files["image"]

    try:
        # Read and preprocess
        image_bytes = file.read()
        tensor = preprocess_image(image_bytes)

        # Make prediction
        preds = model.predict(tensor)[0]  # shape: (num_classes,)
        top_idx = int(np.argmax(preds))
        top_conf = float(preds[top_idx])

        # Top-3 classes
        top3_idx = np.argsort(preds)[::-1][:3]
        top3 = [
            {"class": CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}",
             "confidence": round(float(preds[i]) * 100, 2)}
            for i in top3_idx
        ]

        # Encode image for preview
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        ext = file.content_type or "image/jpeg"

        # Return JSON
        return jsonify({
            "prediction": CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else f"Class {top_idx}",
            "confidence": round(top_conf * 100, 2),
            "top3": top3,
            "image_data": f"data:{ext};base64,{img_b64}",
        })

    except Exception as e:
        # ALWAYS return JSON, never empty
        import traceback
        print("❌ Prediction error:", e)
        traceback.print_exc()
        return jsonify({"error": "Prediction failed.", "details": str(e)}), 500