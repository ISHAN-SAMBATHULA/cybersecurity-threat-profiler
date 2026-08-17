"""
preprocessing.py — Data preprocessing pipeline for NSL-KDD.

Handles:
  • Attack label mapping (40+ types → 5 categories)
  • One-Hot Encoding for categorical features
  • Standard scaling for numerical features
  • Saving/loading fitted transformers
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Attack type → category mapping ───────────────────────────────────────────

ATTACK_MAP = {
    'normal': 'normal',
    # DoS attacks
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
    'smurf': 'dos', 'teardrop': 'dos', 'apache2': 'dos',
    'udpstorm': 'dos', 'processtable': 'dos', 'mailbomb': 'dos',
    # Probe attacks
    'satan': 'probe', 'ipsweep': 'probe', 'nmap': 'probe',
    'portsweep': 'probe', 'mscan': 'probe', 'saint': 'probe',
    # R2L attacks
    'guess_passwd': 'r2l', 'ftp_write': 'r2l', 'imap': 'r2l',
    'phf': 'r2l', 'multihop': 'r2l', 'warezmaster': 'r2l',
    'warezclient': 'r2l', 'spy': 'r2l', 'xlock': 'r2l',
    'xsnoop': 'r2l', 'snmpguess': 'r2l', 'snmpgetattack': 'r2l',
    'httptunnel': 'r2l', 'sendmail': 'r2l', 'named': 'r2l',
    'worm': 'r2l',
    # U2R attacks
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'rootkit': 'u2r',
    'perl': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r', 'ps': 'u2r',
}

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']
ATTACK_CATEGORIES = ['normal', 'dos', 'probe', 'r2l', 'u2r']


def map_attack_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw attack labels to 5 categories: normal, dos, probe, r2l, u2r."""
    df = df.copy()
    df['attack_category'] = df['label'].map(ATTACK_MAP).fillna('unknown')
    # Remove any unknown attack types (if any)
    df = df[df['attack_category'] != 'unknown']
    df.drop('label', axis=1, inplace=True)
    return df


def preprocess(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_dir: str = 'models'
) -> tuple:
    """
    Full preprocessing pipeline.

    Steps:
      1. Map attack labels → 5 categories
      2. One-Hot Encode categorical features
      3. Standard scale numerical features
      4. Encode target labels
      5. Save fitted transformers

    Returns:
        (X_train, X_test, y_train, y_test, label_encoder, scaler, feature_names)
    """
    print("\n⚙️  Preprocessing Data")
    print("=" * 40)

    # Step 1: Map attack labels
    train_df = map_attack_labels(train_df)
    test_df  = map_attack_labels(test_df)

    print(f"  Attack categories: {ATTACK_CATEGORIES}")
    print(f"  Train distribution:")
    for cat, count in train_df['attack_category'].value_counts().items():
        print(f"    {cat:>8s}: {count:,}")

    # Step 2: Separate features and target
    X_train = train_df.drop('attack_category', axis=1)
    y_train_raw = train_df['attack_category']
    X_test  = test_df.drop('attack_category', axis=1)
    y_test_raw  = test_df['attack_category']

    # Step 3: One-Hot Encode categorical features
    X_train = pd.get_dummies(X_train, columns=CATEGORICAL_COLS, drop_first=False, dtype=int)
    X_test  = pd.get_dummies(X_test,  columns=CATEGORICAL_COLS, drop_first=False, dtype=int)

    # Align columns (test may have different categories)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    feature_names = list(X_train.columns)
    print(f"  Features after encoding: {len(feature_names)}")

    # Step 4: Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values.astype(np.float64))
    X_test_scaled  = scaler.transform(X_test.values.astype(np.float64))

    # Step 5: Encode target labels
    label_encoder = LabelEncoder()
    label_encoder.fit(ATTACK_CATEGORIES)
    y_train = label_encoder.transform(y_train_raw)
    y_test  = label_encoder.transform(y_test_raw)

    print(f"  Label classes: {list(label_encoder.classes_)}")

    # Step 6: Save transformers
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(save_dir, 'label_encoder.pkl'))
    joblib.dump(feature_names, os.path.join(save_dir, 'feature_names.pkl'))
    print(f"  ✓ Saved scaler, label_encoder, feature_names → {save_dir}/")

    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, scaler, feature_names
