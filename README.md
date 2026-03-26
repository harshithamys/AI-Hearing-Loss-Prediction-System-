# AI-Based Hearing Loss Prediction System

Classifies hearing loss severity — **Normal / Mild / Moderate / Severe** — from
audiometric measurements using a Random Forest model, a Flask REST API, and MySQL storage.

---

## Project Overview

| Layer       | Tech                                      |
|-------------|-------------------------------------------|
| Language    | Python 3.9+                               |
| ML Model    | Random Forest (scikit-learn)              |
| Audio feats | librosa (MFCCs, chroma, spectral contrast)|
| Database    | MySQL                                     |
| API         | Flask + flask-cors                        |

### WHO Hearing Loss Classification
| Class | Label    | PTA (better ear) |
|-------|----------|-----------------|
| 0     | Normal   | ≤ 25 dB HL      |
| 1     | Mild     | 26–40 dB HL     |
| 2     | Moderate | 41–60 dB HL     |
| 3     | Severe   | > 60 dB HL      |

PTA = Pure Tone Average at 500 / 1000 / 2000 / 4000 Hz.

---

## Directory Structure

```
hearing-loss-prediction/
├── api/
│   └── app.py              ← Flask REST API
├── data/
│   ├── raw/                ← raw CSV (generated or Kaggle)
│   ├── processed/          ← cleaned + feature-engineered CSV
│   ├── generate_data.py    ← synthetic data generator
│   └── README.md           ← dataset documentation
├── database/
│   ├── schema.sql          ← MySQL table definitions
│   └── db_manager.py       ← insert / query helpers
├── docs/                   ← architecture & API docs
├── models/                 ← saved .pkl, metrics.json, plots
├── notebooks/              ← Jupyter EDA notebook
├── src/
│   ├── preprocess.py       ← load → clean → feature engineer → split → scale
│   ├── train.py            ← Random Forest training (+ GridSearchCV)
│   ├── evaluate.py         ← accuracy / F1 / confusion matrix / plots
│   └── audio_features.py   ← librosa feature extraction
├── tests/
│   └── test_pipeline.py
├── config.py               ← all settings in one place
├── run.py                  ← one-command pipeline runner
└── requirements.txt
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment (optional)
```bash
cp .env.example .env
# Edit .env — set DB credentials if using MySQL
```

### 3. Initialize MySQL (optional)
```bash
mysql -u root -p < database/schema.sql
```

---

## Running the Pipeline

### Full pipeline (generate data → train → evaluate)
```bash
python run.py
```

### With hyperparameter tuning
```bash
python run.py --tune
```

### Then start the API
```bash
python api/app.py
# or
python run.py --api
```

---

## API Endpoints

### Health check
```
GET  /health
```

### Predict from manual audiometric data
```
POST /api/predict
Content-Type: application/json

{
  "age": 55, "gender": 1, "noise_exposure": 15, "tinnitus": 1,
  "hearing_250hz_left": 20,  "hearing_250hz_right": 25,
  "hearing_500hz_left": 35,  "hearing_500hz_right": 40,
  "hearing_1000hz_left": 45, "hearing_1000hz_right": 50,
  "hearing_2000hz_left": 55, "hearing_2000hz_right": 60,
  "hearing_4000hz_left": 65, "hearing_4000hz_right": 70,
  "hearing_8000hz_left": 75, "hearing_8000hz_right": 80
}
```
**Response**
```json
{
  "input_source": "manual",
  "severity_class": 2,
  "severity_label": "Moderate",
  "confidence": 0.84,
  "probabilities": {"Normal": 0.02, "Mild": 0.10, "Moderate": 0.84, "Severe": 0.04}
}
```

### Predict from audio file (librosa)
```
POST /api/predict/audio
Content-Type: multipart/form-data
file=<your_audio.wav>
```

### Patient management
```
POST /api/patients          ← add patient
GET  /api/patients          ← list all
GET  /api/patients/{id}/predictions
```

---

## Dataset

The project ships with a synthetic audiometric dataset generator.  
To use real data, download from Kaggle and place in `data/raw/audiometric_data.csv`:

> https://www.kaggle.com/datasets/paultimothymooney/hearing-test-results-dataset

Features: age, gender, noise exposure, tinnitus flag, hearing thresholds at
250 / 500 / 1000 / 2000 / 4000 / 8000 Hz for both ears (dB HL).

---

## Model Performance (synthetic data)

Typical results on the generated dataset:

| Metric   | Score   |
|----------|---------|
| Accuracy | ~95 %   |
| F1 (weighted) | ~0.95 |

Actual performance will vary on real audiometric data.

---


