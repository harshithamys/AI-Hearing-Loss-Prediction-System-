"""
src/train.py
============
Train a Random Forest classifier on the preprocessed audiometric data.

Usage:
    python src/train.py          # fast training (default params)
    python src/train.py --tune   # GridSearchCV hyperparameter tuning
"""

import os
import sys
import json
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PATH, METRICS_PATH, MODELS_DIR, MODEL_PARAMS, RANDOM_STATE
from src.preprocess import preprocess

# Smaller grid for tuning (fast but still meaningful)
PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth":    [10, 20, None],
    "max_features": ["sqrt", "log2"],
}


def train(X_train, y_train, tune: bool = False) -> RandomForestClassifier:
    """Fit a Random Forest; optionally run GridSearchCV first."""
    if tune:
        print("[train] Running GridSearchCV (this may take a minute)…")
        rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        gs = GridSearchCV(rf, PARAM_GRID, cv=5, scoring="f1_weighted",
                          n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        print(f"[train] Best params : {gs.best_params_}")
        print(f"[train] Best CV F1  : {gs.best_score_:.4f}")
        model = gs.best_estimator_
    else:
        print("[train] Training with default parameters…")
        model = RandomForestClassifier(**MODEL_PARAMS)
        model.fit(X_train, y_train)

    return model


def cross_validate(model, X_train, y_train) -> dict:
    """5-fold cross-validation on training set."""
    scores = cross_val_score(model, X_train, y_train,
                             cv=5, scoring="f1_weighted", n_jobs=-1)
    result = {"cv_f1_mean": round(float(scores.mean()), 4),
              "cv_f1_std":  round(float(scores.std()),  4)}
    print(f"[train] CV F1 (weighted): {result['cv_f1_mean']} ± {result['cv_f1_std']}")
    return result


def save_model(model, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[train] Model saved → {path}")


def load_model(path: str = MODEL_PATH) -> RandomForestClassifier:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No trained model at {path}. Run train.py first.")
    return joblib.load(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true",
                        help="Run GridSearchCV hyperparameter tuning")
    args = parser.parse_args()

    # 1. Preprocess
    X_train, X_test, y_train, y_test, features = preprocess()

    # 2. Train
    model = train(X_train, y_train, tune=args.tune)

    # 3. Cross-validate
    cv_results = cross_validate(model, X_train, y_train)

    # 4. Save model
    save_model(model)

    # 5. Save feature list alongside the model (needed by API)
    feat_path = os.path.join(MODELS_DIR, "feature_names.json")
    with open(feat_path, "w") as f:
        json.dump(features, f)

    # 6. Cache test split for evaluate.py
    np.save(os.path.join(MODELS_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(MODELS_DIR, "y_test.npy"), y_test)

    print("\n[train] Done. Next → python src/evaluate.py")
