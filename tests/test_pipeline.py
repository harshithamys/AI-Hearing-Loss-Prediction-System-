"""
tests/test_pipeline.py
======================
Smoke tests for the core pipeline components.
Run with:  pytest tests/
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

class TestDataGeneration:
    def test_generate_returns_dataframe(self):
        from data.generate_data import generate
        df = generate(n=200, seed=0)
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self):
        from data.generate_data import generate
        df = generate(n=100, seed=0)
        assert "hearing_loss_severity" in df.columns
        assert "hearing_1000hz_left" in df.columns

    def test_label_range(self):
        from data.generate_data import generate
        df = generate(n=500, seed=0)
        assert set(df["hearing_loss_severity"].unique()).issubset({0, 1, 2, 3})

    def test_audiogram_values_in_range(self):
        from data.generate_data import generate
        df = generate(n=200, seed=0)
        from config import FREQ_COLS_LEFT, FREQ_COLS_RIGHT
        for col in FREQ_COLS_LEFT + FREQ_COLS_RIGHT:
            assert df[col].between(0, 120).all(), f"{col} out of range"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_clean_removes_duplicates(self):
        from data.generate_data import generate
        from src.preprocess import clean
        df = generate(n=100, seed=1)
        df = pd.concat([df, df.iloc[:5]])  # add 5 duplicates
        cleaned = clean(df)
        assert len(cleaned) == 100

    def test_add_features_creates_pta(self):
        from data.generate_data import generate
        from src.preprocess import add_features
        df = generate(n=100, seed=1)
        df = add_features(df)
        assert "pta_better" in df.columns
        assert "hf_avg_left" in df.columns

    def test_pta_non_negative(self):
        from data.generate_data import generate
        from src.preprocess import add_features
        df = generate(n=200, seed=2)
        df = add_features(df)
        assert (df["pta_better"] >= 0).all()


# ---------------------------------------------------------------------------
# Audio features
# ---------------------------------------------------------------------------

class TestAudioFeatures:
    def test_feature_names_length(self):
        from src.audio_features import feature_names
        names = feature_names()
        assert len(names) == 48

    def test_extract_features_with_synthetic_audio(self, tmp_path):
        """Generate a short sine wave and extract features."""
        pytest.importorskip("librosa")
        import soundfile as sf
        from src.audio_features import extract_features

        sr   = 22050
        tone = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
        wav  = str(tmp_path / "test.wav")
        sf.write(wav, tone, sr)

        feats = extract_features(wav)
        assert feats.shape == (48,)
        assert not np.isnan(feats).any()


# ---------------------------------------------------------------------------
# Model (integration — requires trained model)
# ---------------------------------------------------------------------------

class TestModel:
    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                        "models", "rf_model.pkl")),
        reason="Model not trained yet"
    )
    def test_predict_shape(self):
        import joblib
        from config import MODEL_PATH, SCALER_PATH
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        x = np.random.rand(1, 26)
        x_s = scaler.transform(x)
        pred = model.predict(x_s)
        assert pred.shape == (1,)
        assert pred[0] in {0, 1, 2, 3}
