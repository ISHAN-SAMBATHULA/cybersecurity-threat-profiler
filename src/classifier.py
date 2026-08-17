"""
classifier.py — Train and compare classification models.

Models:
  • Logistic Regression (baseline)
  • Decision Tree (interpretable)
  • Random Forest (primary model)

Target: Multi-class (normal, dos, probe, r2l, u2r)
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score


def train_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    save_dir: str = 'models'
) -> dict:
    """
    Train 3 classifiers, compare performance, and save the best model.

    Returns:
        results dict with model names, metrics, and the best model.
    """
    print("\n🎯 Training Classification Models")
    print("=" * 40)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42, n_jobs=-1),
        'Decision Tree':       DecisionTreeClassifier(random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }

    results = {}
    best_f1 = 0
    best_name = None
    best_model = None

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average='macro')
        report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)

        results[name] = {
            'model': model,
            'accuracy': acc,
            'f1_macro': f1,
            'y_pred': y_pred,
            'report': report,
        }

        print(f"    Accuracy: {acc:.4f}")
        print(f"    F1 (macro): {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    # Save best model
    print(f"\n  🏆 Best model: {best_name} (F1={best_f1:.4f})")
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(save_dir, 'classifier.pkl'))
    joblib.dump(best_name, os.path.join(save_dir, 'best_model_name.pkl'))
    print(f"  ✓ Saved classifier → {save_dir}/classifier.pkl")

    results['best_model_name'] = best_name
    return results
