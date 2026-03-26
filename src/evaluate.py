"""
src/evaluate.py
===============
Evaluate the trained model on the held-out test set and save metrics.

Usage:
    python src/evaluate.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import METRICS_PATH, MODELS_DIR, LABELS
from src.train import load_model

CLASS_NAMES = [LABELS[i] for i in sorted(LABELS)]


def evaluate():
    # Load cached test split
    X_test = np.load(os.path.join(MODELS_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(MODELS_DIR, "y_test.npy"))

    model = load_model()
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    present = sorted(set(y_test))
    names   = [CLASS_NAMES[i] for i in present]
    report  = classification_report(y_test, y_pred, labels=present,
                                    target_names=names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=present)

    # --- Print results ---
    print("\n" + "="*55)
    print("  MODEL EVALUATION RESULTS")
    print("="*55)
    print(f"  Accuracy : {acc*100:.2f}%")
    print("\n" + classification_report(y_test, y_pred, labels=present, target_names=names))

    # --- Save metrics JSON ---
    metrics = {
        "accuracy":          round(acc, 4),
        "classification_report": report,
        "confusion_matrix":  cm.tolist(),
    }
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] Metrics saved → {METRICS_PATH}")

    # --- Feature importance plot ---
    feat_path = os.path.join(MODELS_DIR, "feature_names.json")
    if os.path.exists(feat_path):
        with open(feat_path) as f:
            feat_names = json.load(f)

        importances = model.feature_importances_
        top_n = 15
        idx = np.argsort(importances)[::-1][:top_n]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(range(top_n), importances[idx])
        ax.set_xticks(range(top_n))
        ax.set_xticklabels([feat_names[i] for i in idx], rotation=45, ha="right")
        ax.set_title("Top Feature Importances — Random Forest")
        ax.set_ylabel("Importance")
        plt.tight_layout()
        plot_path = os.path.join(MODELS_DIR, "feature_importance.png")
        plt.savefig(plot_path)
        print(f"[evaluate] Feature importance plot → {plot_path}")

    # --- Confusion matrix plot ---
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"[evaluate] Confusion matrix plot  → {cm_path}")

    return metrics


if __name__ == "__main__":
    evaluate()
