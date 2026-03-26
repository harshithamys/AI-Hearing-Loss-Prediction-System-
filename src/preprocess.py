"""
src/preprocess.py
=================
Load raw audiometric CSV, clean it, engineer features, and return
train/test splits ready for model training.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, SCALER_PATH,
    FREQ_COLS_LEFT, FREQ_COLS_RIGHT, DEMO_COLS, TARGET_COL,
    TEST_SIZE, RANDOM_STATE,
)


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load CSV dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run  python data/generate_data.py  first."
        )
    df = pd.read_csv(path)
    print(f"[preprocess] Loaded {len(df)} rows from {path}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, clip audiometric values to valid range, fill NaNs."""
    df = df.drop_duplicates()
    hearing_cols = FREQ_COLS_LEFT + FREQ_COLS_RIGHT
    df[hearing_cols] = df[hearing_cols].clip(0, 120).fillna(df[hearing_cols].median())
    df["age"]            = df["age"].clip(0, 120).fillna(df["age"].median())
    df["noise_exposure"] = df["noise_exposure"].clip(0, 60).fillna(0)
    df["tinnitus"]       = df["tinnitus"].fillna(0).astype(int)
    df["gender"]         = df["gender"].fillna(0).astype(int)
    df = df.dropna(subset=[TARGET_COL])
    print(f"[preprocess] After cleaning: {len(df)} rows")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer clinically meaningful features from raw audiograms."""
    pta_freqs_l = ["hearing_500hz_left",  "hearing_1000hz_left",
                   "hearing_2000hz_left", "hearing_4000hz_left"]
    pta_freqs_r = ["hearing_500hz_right", "hearing_1000hz_right",
                   "hearing_2000hz_right","hearing_4000hz_right"]

    df["pta_left"]   = df[pta_freqs_l].mean(axis=1)
    df["pta_right"]  = df[pta_freqs_r].mean(axis=1)
    df["pta_better"] = df[["pta_left", "pta_right"]].min(axis=1)   # better ear
    df["pta_worse"]  = df[["pta_left", "pta_right"]].max(axis=1)

    # High-frequency average (noise-induced loss indicator)
    df["hf_avg_left"]  = df[["hearing_4000hz_left",  "hearing_8000hz_left"]].mean(axis=1)
    df["hf_avg_right"] = df[["hearing_4000hz_right", "hearing_8000hz_right"]].mean(axis=1)

    # Inter-ear asymmetry at speech frequencies
    df["asym_1khz"] = (df["hearing_1000hz_left"] - df["hearing_1000hz_right"]).abs()
    df["asym_4khz"] = (df["hearing_4000hz_left"] - df["hearing_4000hz_right"]).abs()

    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return all feature column names (everything except the target)."""
    return [c for c in df.columns if c != TARGET_COL]


def preprocess(save: bool = True):
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    df = load_data()
    df = clean(df)
    df = add_features(df)

    features = get_feature_cols(df)
    X = df[features].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Persist
    if save:
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"[preprocess] Saved processed data → {PROCESSED_DATA_PATH}")
        print(f"[preprocess] Saved scaler         → {SCALER_PATH}")

    print(f"[preprocess] Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(features)}")
    return X_train, X_test, y_train, y_test, features


if __name__ == "__main__":
    preprocess()
