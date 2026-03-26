"""
data/generate_data.py
=====================
Generates a realistic synthetic audiometric dataset (3 000 samples).

Each row represents one patient's hearing test with:
  - Demographics : age, gender, noise_exposure (years), tinnitus (0/1)
  - Audiogram    : hearing thresholds (dB HL) at 6 frequencies for both ears
  - Label        : hearing_loss_severity  (0=Normal, 1=Mild, 2=Moderate, 3=Severe)

Classification follows WHO guidelines:
  PTA (500/1000/2000/4000 Hz) <= 25  → Normal
  PTA 26-40                           → Mild
  PTA 41-60                           → Moderate
  PTA > 60                            → Severe

Run:
    python data/generate_data.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_PATH

FREQS   = [250, 500, 1000, 2000, 4000, 8000]
N       = 3000
SEED    = 42

# Target class proportions
CLASS_DIST = {0: 0.40, 1: 0.30, 2: 0.20, 3: 0.10}   # Normal/Mild/Moderate/Severe

# PTA range (mean, std) for each class (dB HL)
PTA_PARAMS = {
    0: (15, 5),    # Normal   — PTA  5-25
    1: (33, 5),    # Mild     — PTA 26-40
    2: (50, 6),    # Moderate — PTA 41-60
    3: (72, 8),    # Severe   — PTA 61-90
}

# Audiometric slope: higher freqs have larger thresholds relative to PTA
# shape: (6,) — multipliers for [250, 500, 1000, 2000, 4000, 8000] Hz
FREQ_SLOPE = np.array([0.6, 0.8, 1.0, 1.1, 1.4, 1.8])


def classify_hearing(pta: float) -> int:
    """WHO hearing-loss classification based on Pure Tone Average."""
    if pta <= 25:  return 0  # Normal
    if pta <= 40:  return 1  # Mild
    if pta <= 60:  return 2  # Moderate
    return 3                  # Severe


def generate(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Sample class assignments ---
    counts = {cls: round(n * p) for cls, p in CLASS_DIST.items()}
    # adjust rounding to sum exactly to n
    counts[0] += n - sum(counts.values())

    # --- Build rows per class ---
    rows = []
    for cls, cnt in counts.items():
        pta_mean, pta_std = PTA_PARAMS[cls]

        # Sample PTA for each ear independently
        pta_l = rng.normal(pta_mean, pta_std, cnt).clip(0, 110)
        pta_r = rng.normal(pta_mean, pta_std, cnt).clip(0, 110)

        # Derive per-frequency thresholds from PTA via frequency slope
        # thresholds ≈ PTA * freq_multiplier + noise
        thresh_l = (np.outer(pta_l, FREQ_SLOPE) +
                    rng.normal(0, 3, (cnt, 6))).clip(0, 110).round(1)
        thresh_r = (np.outer(pta_r, FREQ_SLOPE) +
                    rng.normal(0, 3, (cnt, 6))).clip(0, 110).round(1)

        age            = rng.integers(18, 80, cnt)
        gender         = rng.integers(0, 2, cnt)
        noise_exposure = rng.exponential(4, cnt).clip(0, 40).round(1)
        tinnitus       = (rng.random(cnt) < 0.25).astype(int)

        for i in range(cnt):
            row = {
                "age": int(age[i]),
                "gender": int(gender[i]),
                "noise_exposure": float(noise_exposure[i]),
                "tinnitus": int(tinnitus[i]),
            }
            for j, f in enumerate(FREQS):
                row[f"hearing_{f}hz_left"]  = float(thresh_l[i, j])
                row[f"hearing_{f}hz_right"] = float(thresh_r[i, j])
            row["hearing_loss_severity"] = cls
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle
    return df


if __name__ == "__main__":
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df = generate()
    df.to_csv(RAW_DATA_PATH, index=False)

    print(f"Dataset saved  : {RAW_DATA_PATH}")
    print(f"Shape          : {df.shape}")
    print("\nClass distribution:")
    labels = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
    for k, v in df["hearing_loss_severity"].value_counts().sort_index().items():
        print(f"  {k} - {labels[k]:<10} : {v} samples ({v/len(df)*100:.1f}%)")
