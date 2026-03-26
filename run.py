"""
run.py — End-to-End Pipeline Runner
=====================================
Runs every stage of the hearing-loss prediction pipeline in order.

Usage:
    python run.py            # full pipeline (generate → train → evaluate)
    python run.py --api      # start Flask API after pipeline
    python run.py --tune     # include GridSearchCV hyperparameter tuning
"""

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def banner(text: str) -> None:
    print("\n" + "=" * 55)
    print(f"  {text}")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Hearing Loss Prediction Pipeline")
    parser.add_argument("--api",  action="store_true", help="Start Flask API when done")
    parser.add_argument("--tune", action="store_true", help="GridSearchCV tuning")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Step 1 — Generate synthetic dataset
    # ------------------------------------------------------------------ #
    banner("STEP 1 / 4 — Generating dataset")
    from data.generate_data import generate
    from config import RAW_DATA_PATH
    import os, pandas as pd
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df = generate()
    df.to_csv(RAW_DATA_PATH, index=False)
    labels = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
    for k, v in df["hearing_loss_severity"].value_counts().sort_index().items():
        print(f"  {k} - {labels[k]:<10} : {v} ({v/len(df)*100:.1f}%)")

    # ------------------------------------------------------------------ #
    # Step 2 — Preprocess
    # ------------------------------------------------------------------ #
    banner("STEP 2 / 4 — Preprocessing")
    from src.preprocess import preprocess
    X_train, X_test, y_train, y_test, features = preprocess()

    # ------------------------------------------------------------------ #
    # Step 3 — Train
    # ------------------------------------------------------------------ #
    banner("STEP 3 / 4 — Training Random Forest")
    import json, numpy as np
    from src.train import train, cross_validate, save_model
    from config import MODELS_DIR
    model = train(X_train, y_train, tune=args.tune)
    cv    = cross_validate(model, X_train, y_train)
    save_model(model)

    feat_path = os.path.join(MODELS_DIR, "feature_names.json")
    with open(feat_path, "w") as f:
        json.dump(features, f)
    np.save(os.path.join(MODELS_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(MODELS_DIR, "y_test.npy"), y_test)

    # ------------------------------------------------------------------ #
    # Step 4 — Evaluate
    # ------------------------------------------------------------------ #
    banner("STEP 4 / 4 — Evaluating")
    from src.evaluate import evaluate
    metrics = evaluate()

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    banner("PIPELINE COMPLETE")
    print(f"  Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"  CV F1     : {cv['cv_f1_mean']} ± {cv['cv_f1_std']}")
    print(f"  Model     : models/rf_model.pkl")
    print(f"  Metrics   : models/metrics.json")
    if args.api:
        print("\n  Starting Flask API…")
        import subprocess
        subprocess.run([sys.executable, "api/app.py"])


if __name__ == "__main__":
    main()
