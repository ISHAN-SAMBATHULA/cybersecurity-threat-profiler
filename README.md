# 🛡️ Cybersecurity Network Threat & Intrusion Profiler Using Machine Learning

An end-to-end machine learning pipeline for detecting network intrusions using the **NSL-KDD** dataset. Combines **supervised classification** (identifying known attack types) with **unsupervised anomaly detection** (flagging unknown/zero-day threats) into a unified threat profiling system with a **Streamlit dashboard**.

---

## 📋 Project Overview

| Component | Description |
|---|---|
| **Dataset** | NSL-KDD (refined KDD Cup 1999) |
| **Classification** | Random Forest, Decision Tree, Logistic Regression |
| **Anomaly Detection** | Isolation Forest |
| **Target Labels** | Normal, DoS, Probe, R2L, U2R |
| **Dashboard** | Streamlit web application |

### ML Pipeline

```
NSL-KDD → Preprocessing → Feature Engineering → Classification → Anomaly Detection → Threat Profiling → Dashboard
```

---

## 🗂️ Project Structure

```
IBM/
├── data/raw/                  ← NSL-KDD dataset (auto-downloaded)
├── models/                    ← Saved models & transformers
├── reports/figures/           ← Evaluation plots
├── src/
│   ├── data_loader.py         ← Dataset download & loading
│   ├── preprocessing.py       ← Encoding, scaling, label mapping
│   ├── classifier.py          ← Train & compare classifiers
│   ├── anomaly_detector.py    ← Isolation Forest training
│   ├── threat_profiler.py     ← Unified threat assessment
│   └── evaluate.py            ← Metrics & visualizations
├── app/
│   └── streamlit_app.py       ← Web dashboard
├── train_pipeline.py          ← End-to-end training script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
python train_pipeline.py
```

This will:
- Download the NSL-KDD dataset
- Preprocess and encode features
- Train and compare 3 classifiers
- Train the Isolation Forest anomaly detector
- Generate evaluation plots
- Save all models to `models/`

### 3. Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 🎯 Models

### Classification (Supervised)

Three models are trained and compared:

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline linear model |
| Decision Tree | Interpretable tree model |
| **Random Forest** | Primary ensemble model |

**Target categories:** Normal, DoS, Probe, R2L, U2R

### Anomaly Detection (Unsupervised)

**Isolation Forest** trained exclusively on normal traffic to learn baseline network behaviour. At inference, it flags traffic that deviates from the learned normal patterns as anomalous.

### Threat Profiler

Combines both models into a unified risk assessment:

| Risk Level | Condition |
|---|---|
| **CRITICAL** | Attack classified AND anomaly detected |
| **HIGH** | Attack classified |
| **MEDIUM** | Anomaly detected only |
| **LOW** | Normal traffic, no anomaly |

---

## 📊 Evaluation Metrics

### Classification
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- Model Comparison Chart
- Feature Importance Plot

### Anomaly Detection
- Precision, Recall, F1-score
- Confusion Matrix
- Anomaly Score Distribution
- Detection Rate

---

## 💻 Dashboard

The Streamlit dashboard has two pages:

1. **🛡️ Threat Analyzer** — Input network features and get real-time classification, anomaly detection, and risk assessment
2. **📊 Analytics Dashboard** — View attack distributions, model comparisons, confusion matrices, and feature importance

---

## 🛠️ Technologies

- **Python 3.10+**
- **scikit-learn** — Classification & anomaly detection
- **pandas / numpy** — Data processing
- **matplotlib / seaborn** — Static plots
- **plotly** — Interactive charts
- **Streamlit** — Web dashboard

---

## 📚 Dataset

**NSL-KDD** — A benchmark dataset for network intrusion detection, containing 41 network traffic features and labels for normal traffic and 4 attack categories (DoS, Probe, R2L, U2R).

- Train set: ~125,000 samples
- Test set: ~22,000 samples

---

*IBM SkillsBuild Project Submission*
