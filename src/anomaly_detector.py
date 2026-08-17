"""
anomaly_detector.py — Isolation Forest for anomaly detection.

Trained only on "normal" traffic samples, then used to flag
anomalous/unknown network behaviour at inference time.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix


def train_anomaly_detector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    normal_label: int = 0,
    save_dir: str = 'models'
) -> dict:
    """
    Train Isolation Forest on normal traffic only.

    The model learns what "normal" looks like, then flags anything
    that deviates as an anomaly.

    Args:
        X_train, y_train: training data
        X_test, y_test: testing data
        normal_label: encoded label for "normal" (default 0 from sorted ATTACK_CATEGORIES)
        save_dir: where to save the model

    Returns:
        results dict with predictions and metrics.
    """
    print("\n🔍 Training Anomaly Detector (Isolation Forest)")
    print("=" * 40)

    # Train only on normal samples
    normal_mask = (y_train == normal_label)
    X_normal = X_train[normal_mask]
    print(f"  Normal training samples: {len(X_normal):,}")

    # Fit Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(X_normal)

    # Predict on test set
    # IsolationForest returns: 1 = normal, -1 = anomaly
    raw_preds = iso_forest.predict(X_test)
    anomaly_scores = iso_forest.decision_function(X_test)

    # Convert to binary: 0 = normal, 1 = anomaly
    anomaly_preds = (raw_preds == -1).astype(int)

    # Ground truth: 0 = normal, 1 = attack (any non-normal)
    y_true_binary = (y_test != normal_label).astype(int)

    # Metrics
    report = classification_report(
        y_true_binary, anomaly_preds,
        target_names=['Normal', 'Anomaly'],
        output_dict=True
    )
    cm = confusion_matrix(y_true_binary, anomaly_preds)

    anomaly_rate = anomaly_preds.sum() / len(anomaly_preds) * 100

    print(f"  Anomaly detection rate: {anomaly_rate:.1f}%")
    print(f"  Precision: {report['Anomaly']['precision']:.4f}")
    print(f"  Recall:    {report['Anomaly']['recall']:.4f}")
    print(f"  F1-score:  {report['Anomaly']['f1-score']:.4f}")

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(iso_forest, os.path.join(save_dir, 'anomaly_detector.pkl'))
    print(f"  ✓ Saved anomaly detector → {save_dir}/anomaly_detector.pkl")

    return {
        'model': iso_forest,
        'anomaly_preds': anomaly_preds,
        'anomaly_scores': anomaly_scores,
        'y_true_binary': y_true_binary,
        'report': report,
        'confusion_matrix': cm,
        'anomaly_rate': anomaly_rate,
    }
