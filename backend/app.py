print("🚀 Starting NeuroPulse AI backend...")

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
import os

# =========================================================
# TensorFlow / Keras Setup
# =========================================================
try:
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model
    print("✅ TensorFlow loaded successfully")
except Exception as e:
    print("❌ TensorFlow import failed. Install it via: pip install tensorflow")
    raise e

# =========================================================
# Import Utils
# =========================================================
try:
    from utils import detect_anomaly
    print("✅ utils.py loaded successfully")
except ImportError:
    print("❌ Error: 'utils.py' not found in the same folder as app.py")
    exit(1)

# =========================================================
# Flask App Setup
# =========================================================
# Ensure static folder points correctly to frontend
app = Flask(
    __name__,
    static_folder="../frontend", 
    static_url_path=""
)

# =========================================================
# Model Architecture
# =========================================================
INPUT_DIM = 720

def build_model():
    input_layer = Input(shape=(INPUT_DIM,))
    encoded = Dense(256, activation="relu")(input_layer)
    encoded = Dense(128, activation="relu")(encoded)
    encoded = Dense(64, activation="relu")(encoded)

    decoded = Dense(128, activation="relu")(encoded)
    decoded = Dense(256, activation="relu")(decoded)
    output_layer = Dense(INPUT_DIM, activation="linear")(decoded)

    return Model(input_layer, output_layer)

model = build_model()

# =========================================================
# Load Weights
# =========================================================
# Path to your weights file
WEIGHTS_PATH = "model/neuropulse_autoencoder_fixed.h5"

print(f"📦 Loading model weights from: {WEIGHTS_PATH}")
if os.path.exists(WEIGHTS_PATH):
    try:
        model.load_weights(WEIGHTS_PATH)
        print("✅ Model weights loaded successfully")
    except Exception as e:
        print(f"❌ Error loading weights file: {e}")
else:
    print(f"⚠️ WARNING: Weight file not found at {WEIGHTS_PATH}")
    print("   The system will run with an untrained random model (results will be inaccurate).")

# =========================================================
# Routes
# =========================================================

@app.route("/")
def home():
    # Serves the index.html from frontend folder
    return send_from_directory(app.static_folder, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        # -------------------------------------------------
        # Robust CSV Reading (Fixes 'ValueError' issues)
        # -------------------------------------------------
        try:
            # Read CSV
            df = pd.read_csv(file, header=None)
            
            # Force conversion to numeric (turns text/headers into NaN)
            data_col = pd.to_numeric(df.iloc[:, 0], errors='coerce')
            
            # Drop the NaNs (removes headers or bad lines)
            ecg = data_col.dropna().values.astype(np.float32)
            
        except Exception as csv_err:
            print(f"CSV Read Error: {csv_err}")
            return jsonify({"error": "Invalid CSV format. Ensure column 1 contains numbers."}), 400

        print(f"✅ Data Loaded. Samples: {len(ecg)}")

        if len(ecg) < INPUT_DIM:
            return jsonify({
                "error": f"Signal too short. Need at least {INPUT_DIM} data points."
            }), 400

        # -------------------------------------------------
        # Run Detection
        # -------------------------------------------------
        # We expect two return values: mask (array) and percentage (float)
        _, anomaly_pct = detect_anomaly(ecg, model)

        print(f"✅ Analysis Result: {anomaly_pct:.2f}% Anomaly")

        # -------------------------------------------------
        # Severity Logic
        # -------------------------------------------------
        if anomaly_pct <= 10:
            result_text = "Very Normal ECG"
        elif anomaly_pct <= 30:
            result_text = "Mostly Normal (No problem)"
        elif anomaly_pct <= 60:
            result_text = "Suspicious ECG"
        else:
            result_text = "Highly Abnormal ECG"

        return jsonify({
            "result": result_text,
            "anomaly_percentage": anomaly_pct
        })

    except Exception as e:
        print(f"❌ Server Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🌐 Server running on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)