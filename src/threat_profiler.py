"""
threat_profiler.py — Unified threat profiling.

Combines classification (known attacks) and anomaly detection (unknown threats)
into a single risk assessment with a confidence score and risk level.
"""

import numpy as np
import joblib
import os


def load_models(model_dir: str = 'models') -> dict:
    """Load all trained models and transformers."""
    return {
        'classifier':     joblib.load(os.path.join(model_dir, 'classifier.pkl')),
        'anomaly_detector': joblib.load(os.path.join(model_dir, 'anomaly_detector.pkl')),
        'scaler':         joblib.load(os.path.join(model_dir, 'scaler.pkl')),
        'label_encoder':  joblib.load(os.path.join(model_dir, 'label_encoder.pkl')),
        'feature_names':  joblib.load(os.path.join(model_dir, 'feature_names.pkl')),
    }


def profile_threat(features: np.ndarray, models: dict) -> dict:
    """
    Analyse a single network connection sample.

    Args:
        features: 1D array of preprocessed, scaled features
        models: dict from load_models()

    Returns:
        dict with classification result, anomaly status, and risk level.
    """
    classifier     = models['classifier']
    anomaly_det    = models['anomaly_detector']
    label_encoder  = models['label_encoder']

    # Reshape for single sample
    X = features.reshape(1, -1)

    # ── Classification ───────────────────────────────────────────────────
    class_pred = classifier.predict(X)[0]
    class_proba = classifier.predict_proba(X)[0]
    class_label = label_encoder.inverse_transform([class_pred])[0]
    confidence = float(class_proba.max()) * 100

    # ── Anomaly Detection ────────────────────────────────────────────────
    anomaly_pred = anomaly_det.predict(X)[0]       # 1 = normal, -1 = anomaly
    anomaly_score = anomaly_det.decision_function(X)[0]
    is_anomaly = (anomaly_pred == -1)

    # ── Risk Level ───────────────────────────────────────────────────────
    is_attack = (class_label != 'normal')

    if is_attack and is_anomaly:
        risk_level = 'CRITICAL'
    elif is_attack:
        risk_level = 'HIGH'
    elif is_anomaly:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'LOW'

    return {
        'classification': class_label.upper(),
        'is_attack': is_attack,
        'attack_type': class_label.upper() if is_attack else 'NONE',
        'confidence': round(confidence, 1),
        'is_anomaly': is_anomaly,
        'anomaly_score': round(float(anomaly_score), 4),
        'risk_level': risk_level,
        'class_probabilities': {
            label_encoder.inverse_transform([i])[0]: round(float(p) * 100, 1)
            for i, p in enumerate(class_proba)
        },
    }
