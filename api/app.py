"""
api/app.py
==========
Flask REST API for the Hearing Loss Prediction System.

Endpoints
---------
GET  /health
    Returns API status.

POST /api/predict
    Accept audiometric features as JSON and return a prediction.
    Body (all dB HL values):
    {
        "patient_id": 1,            (optional — persists to DB if provided)
        "age": 45,
        "gender": 1,
        "noise_exposure": 10,
        "tinnitus": 0,
        "hearing_250hz_left": 15,  "hearing_250hz_right": 20,
        "hearing_500hz_left": 20,  "hearing_500hz_right": 25,
        "hearing_1000hz_left": 25, "hearing_1000hz_right": 30,
        "hearing_2000hz_left": 30, "hearing_2000hz_right": 35,
        "hearing_4000hz_left": 35, "hearing_4000hz_right": 40,
        "hearing_8000hz_left": 40, "hearing_8000hz_right": 45
    }

POST /api/predict/audio
    Upload a WAV/MP3 file; librosa extracts features for prediction.
    Form data: file=<audio file>

POST /api/patients
    Add a new patient. Body: {name, age, gender, noise_exposure, tinnitus}

GET  /api/patients
    List all patients.

GET  /api/patients/<id>/predictions
    Predictions history for a patient.
"""

import os
import sys
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_PATH, SCALER_PATH, MODELS_DIR, UPLOADS_DIR,
    FREQ_COLS_LEFT, FREQ_COLS_RIGHT, DEMO_COLS, LABELS,
    API_CONFIG,
)

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = API_CONFIG["max_content_length"]
os.makedirs(UPLOADS_DIR, exist_ok=True)

ALLOWED = {"wav", "mp3", "ogg", "flac"}


# ---------------------------------------------------------------------------
# Load model + scaler once at startup
# ---------------------------------------------------------------------------

import joblib

def _load_artifacts():
    """Load model, scaler and feature list. Raises if model not trained yet."""
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feat_path = os.path.join(MODELS_DIR, "feature_names.json")
    with open(feat_path) as f:
        features = json.load(f)
    return model, scaler, features

try:
    _model, _scaler, _features = _load_artifacts()
    print("[api] Model loaded successfully.")
except Exception as e:
    _model = _scaler = _features = None
    print(f"[api] WARNING — model not loaded: {e}")
    print("[api] Run  python run.py  first to train the model.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _predict_from_array(arr: np.ndarray) -> dict:
    """Scale input, run model, return structured result dict."""
    arr_scaled = _scaler.transform(arr.reshape(1, -1))
    cls_idx    = int(_model.predict(arr_scaled)[0])
    probs      = _model.predict_proba(arr_scaled)[0]
    label      = LABELS[cls_idx]
    confidence = float(probs[cls_idx])

    return {
        "severity_class": cls_idx,
        "severity_label": label,
        "confidence":     round(confidence, 4),
        "probabilities": {LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


def _model_ready():
    if _model is None:
        return jsonify({"error": "Model not trained yet. Run python run.py first."}), 503
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _model is not None})


@app.post("/api/predict")
def predict_json():
    """Predict hearing loss severity from manual audiometric input."""
    err = _model_ready()
    if err: return err

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    # Build feature vector in the same order the model was trained on
    try:
        feature_vec = np.array([data.get(f, 0.0) for f in _features], dtype=float)
    except Exception as e:
        return jsonify({"error": f"Feature extraction failed: {e}"}), 400

    result = _predict_from_array(feature_vec)

    # Optional: persist to DB
    patient_id = data.get("patient_id")
    if patient_id:
        try:
            from database.db_manager import insert_audiometric, insert_prediction
            insert_audiometric(patient_id, data)
            insert_prediction(patient_id, result["severity_label"],
                              result["severity_class"], result["confidence"])
        except Exception:
            pass   # DB optional — don't fail the response

    return jsonify({"input_source": "manual", **result}), 200


@app.post("/api/predict/audio")
def predict_audio():
    """Predict from an uploaded audio file using librosa features."""
    err = _model_ready()
    if err: return err

    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    f = request.files["file"]
    if not f.filename or not _allowed_file(f.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED}"}), 400

    # Save temporarily
    filename = secure_filename(f.filename)
    save_path = os.path.join(UPLOADS_DIR, filename)
    f.save(save_path)

    try:
        from src.audio_features import extract_features
        audio_feats = extract_features(save_path)        # shape (48,)
    except Exception as e:
        os.remove(save_path)
        return jsonify({"error": f"Audio processing failed: {e}"}), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

    # Audio feature vector length may differ from tabular model's feature count.
    # Pad/truncate to match (the model was trained on tabular data).
    n_feat = len(_features)
    if len(audio_feats) < n_feat:
        audio_feats = np.pad(audio_feats, (0, n_feat - len(audio_feats)))
    else:
        audio_feats = audio_feats[:n_feat]

    result = _predict_from_array(audio_feats)
    return jsonify({"input_source": "audio", **result}), 200


@app.post("/api/patients")
def add_patient():
    data = request.get_json(force=True)
    try:
        from database.db_manager import insert_patient
        pid = insert_patient(
            name           = data["name"],
            age            = data["age"],
            gender         = data["gender"],
            noise_exposure = data.get("noise_exposure", 0),
            tinnitus       = data.get("tinnitus", 0),
        )
        if pid:
            return jsonify({"patient_id": pid}), 201
        return jsonify({"error": "DB insert failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/api/patients")
def list_patients():
    from database.db_manager import get_all_patients
    return jsonify(get_all_patients()), 200


@app.get("/api/patients/<int:patient_id>/predictions")
def patient_predictions(patient_id: int):
    from database.db_manager import get_patient_predictions
    return jsonify(get_patient_predictions(patient_id)), 200


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(
        host  = API_CONFIG["host"],
        port  = API_CONFIG["port"],
        debug = API_CONFIG["debug"],
    )
