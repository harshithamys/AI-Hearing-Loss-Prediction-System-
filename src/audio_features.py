"""
src/audio_features.py
=====================
Extract audio features from a recording using librosa.

Used by the API when a patient uploads an audio file instead of
entering manual audiometric values.

Features extracted:
  - 13 MFCCs (mean + std)         → 26 values
  - 12 Chroma features (mean)     → 12 values
  - 7  Spectral contrast (mean)   →  7 values
  - Spectral bandwidth (mean)     →  1 value
  - Spectral rolloff   (mean)     →  1 value
  - Zero-crossing rate (mean)     →  1 value
  Total: 48 features
"""

import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def extract_features(audio_path: str, sr: int = 22050, n_mfcc: int = 13) -> np.ndarray:
    """
    Extract a fixed-length feature vector from an audio file.

    Parameters
    ----------
    audio_path : str  — path to .wav / .mp3 / .ogg / .flac file
    sr         : int  — target sample rate
    n_mfcc     : int  — number of MFCC coefficients

    Returns
    -------
    np.ndarray of shape (48,)
    """
    if not LIBROSA_AVAILABLE:
        raise RuntimeError("librosa is not installed. Run: pip install librosa")

    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    # --- MFCCs ---
    mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean  = mfcc.mean(axis=1)          # (13,)
    mfcc_std   = mfcc.std(axis=1)           # (13,)

    # --- Chroma ---
    chroma     = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean= chroma.mean(axis=1)        # (12,)

    # --- Spectral contrast ---
    contrast   = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = contrast.mean(axis=1)   # (7,)

    # --- Scalar features ---
    bw         = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    rolloff    = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    zcr        = librosa.feature.zero_crossing_rate(y).mean()

    features = np.concatenate([
        mfcc_mean, mfcc_std, chroma_mean, contrast_mean,
        [bw, rolloff, zcr]
    ])                                       # (48,)

    return features.astype(np.float32)


def feature_names() -> list:
    """Return human-readable names for each extracted feature dimension."""
    names = (
        [f"mfcc_{i}_mean" for i in range(13)] +
        [f"mfcc_{i}_std"  for i in range(13)] +
        [f"chroma_{i}"    for i in range(12)] +
        [f"contrast_{i}"  for i in range(7)]  +
        ["spectral_bandwidth", "spectral_rolloff", "zero_crossing_rate"]
    )
    return names
