"""
train_pipeline.py — End-to-end training pipeline.

Run this once to:
  1. Download & load NSL-KDD
  2. Preprocess data
  3. Train & compare classifiers
  4. Train anomaly detector
  5. Generate evaluation plots
  6. Save all models

Usage:
    python train_pipeline.py
"""

import sys
import os
import time
import numpy as np

# Add project root to path
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
if _FILE_DIR not in sys.path:
    sys.path.insert(0, _FILE_DIR)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from src.data_loader import load_data
from src.preprocessing import preprocess
from src.classifier import train_classifiers
from src.anomaly_detector import train_anomaly_detector
from src.evaluate import generate_all_plots
from sklearn.metrics import classification_report


def main():
    # Ensure working directory is the project root so all relative paths resolve
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Cybersecurity Network Threat & Intrusion Profiler         ║")
    print("║  Training Pipeline                                         ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    start_time = time.time()

    # ── Phase 1: Load Data ───────────────────────────────────────────────
    train_df, test_df = load_data()

    # ── Phase 2: Preprocess ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test, label_encoder, scaler, feature_names = preprocess(
        train_df, test_df
    )

    # ── Phase 3: Classification ──────────────────────────────────────────
    label_names = list(label_encoder.classes_)
    clf_results = train_classifiers(
        X_train, y_train, X_test, y_test, label_names
    )

    # Print detailed report for best model
    best_name = clf_results['best_model_name']
    print(f"\n📋 Detailed Classification Report ({best_name})")
    print("=" * 40)
    print(classification_report(
        y_test, clf_results[best_name]['y_pred'],
        target_names=label_names
    ))

    # ── Phase 4: Anomaly Detection ───────────────────────────────────────
    # normal_label = index of 'normal' in sorted ATTACK_CATEGORIES
    normal_label = list(label_encoder.classes_).index('normal')
    anomaly_results = train_anomaly_detector(
        X_train, y_train, X_test, y_test,
        normal_label=normal_label
    )

    print(f"\n📋 Anomaly Detection Report")
    print("=" * 40)
    report = anomaly_results['report']
    print(f"  Precision: {report['Anomaly']['precision']:.4f}")
    print(f"  Recall:    {report['Anomaly']['recall']:.4f}")
    print(f"  F1-score:  {report['Anomaly']['f1-score']:.4f}")
    print(f"  Detection rate: {anomaly_results['anomaly_rate']:.1f}%")

    # ── Phase 5: Evaluation Plots ────────────────────────────────────────
    generate_all_plots(
        clf_results, anomaly_results,
        y_train, y_test,
        label_encoder, feature_names
    )

    # ── Done ─────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"✅ Pipeline complete in {elapsed:.1f}s")
    print(f"   Models saved to: models/")
    print(f"   Plots saved to:  reports/figures/")
    print(f"\n   Launch dashboard:")
    print(f"   streamlit run app/streamlit_app.py")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
