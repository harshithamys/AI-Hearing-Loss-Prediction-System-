"""config.py — Central configuration for Hearing Loss Prediction System"""

import os
from dotenv import load_dotenv

# Load .env file (if present) before reading env vars
load_dotenv()

# =============================================================================
# Base Paths
# =============================================================================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

RAW_DATA_PATH       = os.path.join(BASE_DIR, "data", "raw",       "audiometric_data.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")
MODEL_PATH          = os.path.join(MODELS_DIR, "rf_model.pkl")
SCALER_PATH         = os.path.join(MODELS_DIR, "scaler.pkl")
METRICS_PATH        = os.path.join(MODELS_DIR, "metrics.json")

# --- Database ---
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", "3306")),
    "user":     os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "hearing_loss_db"),
}

# --- Model ---
MODEL_PARAMS = {"n_estimators": 200, "max_depth": 20, "random_state": 42, "n_jobs": -1}
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# --- Audio (librosa) ---
SAMPLE_RATE = 22050
N_MFCC      = 13

# --- Features ---
FREQ_COLS_LEFT  = [f"hearing_{f}hz_left"  for f in [250, 500, 1000, 2000, 4000, 8000]]
FREQ_COLS_RIGHT = [f"hearing_{f}hz_right" for f in [250, 500, 1000, 2000, 4000, 8000]]
DEMO_COLS       = ["age", "gender", "noise_exposure", "tinnitus"]
TARGET_COL      = "hearing_loss_severity"
LABELS          = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}

# --- API ---
API_CONFIG = {
    "host":               os.getenv("API_HOST", "0.0.0.0"),
    "port":               int(os.getenv("API_PORT", "5000")),
    "debug":              os.getenv("FLASK_DEBUG", "False").lower() == "true",
    "max_content_length": 16 * 1024 * 1024,   # 16 MB
}
