"""
Cybersecurity Network Threat & Intrusion Profiler
Streamlit Dashboard

Pages:
  🛡️ Threat Analyzer  — Input features → get classification + anomaly detection results
  📊 Analytics         — Pre-computed charts and model performance
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys

# Add project root to path
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(_APP_DIR, '..')
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import CATEGORICAL_COLS, ATTACK_CATEGORIES
from src.threat_profiler import load_models, profile_threat

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cyber Threat Profiler | AI-Powered NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Global ────────────────────────────── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Animated Header ───────────────────── */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.15); }
        50% { box-shadow: 0 0 40px rgba(99, 102, 241, 0.3); }
    }

    .main-header {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1a0a2e, #0d1b3e);
        background-size: 400% 400%;
        animation: gradientShift 12s ease infinite;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        border: 1px solid rgba(99, 102, 241, 0.25);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
                    radial-gradient(circle at 80% 50%, rgba(168, 85, 247, 0.08) 0%, transparent 50%);
        pointer-events: none;
    }
    .main-header h1 {
        color: #fff;
        font-size: 2.2rem;
        margin: 0;
        font-weight: 800;
        letter-spacing: -0.5px;
        animation: fadeInUp 0.8s ease-out;
    }
    .main-header .subtitle {
        color: rgba(165, 180, 252, 0.9);
        margin: 0.6rem 0 0 0;
        font-size: 1rem;
        font-weight: 400;
        letter-spacing: 0.5px;
        animation: fadeInUp 1s ease-out;
    }
    .main-header .badge-row {
        margin-top: 1rem;
        display: flex;
        justify-content: center;
        gap: 0.75rem;
        flex-wrap: wrap;
        animation: fadeInUp 1.2s ease-out;
    }
    .main-header .tech-badge {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        color: rgba(200, 210, 255, 0.85);
        font-weight: 500;
        backdrop-filter: blur(10px);
    }

    /* ── Glassmorphism Metric Cards ────────── */
    .metric-card {
        background: linear-gradient(145deg, rgba(26, 26, 46, 0.9), rgba(22, 33, 62, 0.9));
        backdrop-filter: blur(20px);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        border: 1px solid rgba(99, 102, 241, 0.15);
        margin-bottom: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: pulseGlow 4s ease-in-out infinite;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
    }
    .metric-card .label {
        color: rgba(148, 163, 184, 0.9);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    .metric-card .value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0.6rem 0 0.2rem 0;
        color: #e2e8f0;
    }

    /* ── Risk Level Colors ─────────────────── */
    .risk-critical {
        color: #ff1744;
        text-shadow: 0 0 15px rgba(255,23,68,0.6), 0 0 30px rgba(255,23,68,0.3);
    }
    .risk-high {
        color: #ff5252;
        text-shadow: 0 0 12px rgba(255,82,82,0.4);
    }
    .risk-medium {
        color: #fbbf24;
        text-shadow: 0 0 8px rgba(251,191,36,0.3);
    }
    .risk-low {
        color: #34d399;
        text-shadow: 0 0 8px rgba(52,211,153,0.3);
    }

    /* ── Sidebar ───────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c0a1f 0%, #131136 50%, #0f0c29 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
    section[data-testid="stSidebar"] .stRadio > label {
        color: rgba(165, 180, 252, 0.7) !important;
        font-weight: 500;
    }

    /* ── Buttons ───────────────────────────── */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }

    /* ── Tabs ──────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
    }

    /* ── Info Boxes ────────────────────────── */
    .info-box {
        background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(168,85,247,0.05));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        color: rgba(200, 210, 255, 0.85);
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* ── Section Headers ──────────────────── */
    .section-header {
        background: linear-gradient(90deg, rgba(99,102,241,0.15), transparent);
        padding: 0.6rem 1rem;
        border-radius: 8px;
        border-left: 3px solid #6366f1;
        margin: 1rem 0;
        font-weight: 600;
        font-size: 1.1rem;
    }

    /* ── Footer ────────────────────────────── */
    .pro-footer {
        background: linear-gradient(145deg, rgba(15,12,41,0.6), rgba(26,26,46,0.6));
        border: 1px solid rgba(99, 102, 241, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 2rem;
    }
    .pro-footer .footer-title {
        color: rgba(148, 163, 184, 0.7);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }
    .pro-footer .footer-badges {
        margin-top: 0.75rem;
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .pro-footer .footer-badge {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.7rem;
        color: rgba(165, 180, 252, 0.7);
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Dataframe styling ─────────────────── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ─────────────────────────────────────────────────────────

@st.cache_resource
def get_models():
    """Load trained models (cached)."""
    try:
        model_dir = os.path.join(PROJECT_ROOT, 'models')
        return load_models(model_dir=model_dir)
    except Exception:
        return None


def risk_badge(level):
    """Return HTML for a colored risk level badge."""
    css_class = f"risk-{level.lower()}"
    return f'<span class="{css_class}">{level}</span>'


# ── Sidebar Navigation ──────────────────────────────────────────────────────

st.sidebar.markdown("""
<div style="text-align:center; padding: 1rem 0 0.5rem 0;">
    <div style="font-size: 2.5rem;">🛡️</div>
    <div style="font-size: 1.1rem; font-weight: 700; color: #a5b4fc; letter-spacing: 0.5px;">CyberThreat AI</div>
    <div style="font-size: 0.7rem; color: rgba(148,163,184,0.6); margin-top: 0.2rem;">Intelligent NIDS v2.0</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🛡️ Threat Analyzer", "📊 Analytics Dashboard"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="padding: 0.5rem; font-size: 0.78rem; color: rgba(148,163,184,0.6);">
    <div style="font-weight:600; color: rgba(165,180,252,0.7); margin-bottom: 0.4rem;">🧠 ML Models</div>
    <div>• Random Forest Classifier</div>
    <div>• Decision Tree Classifier</div>
    <div>• Logistic Regression</div>
    <div>• Isolation Forest (Anomaly)</div>
    <div style="margin-top: 0.8rem; font-weight:600; color: rgba(165,180,252,0.7); margin-bottom: 0.4rem;">📡 Dataset</div>
    <div>• NSL-KDD Benchmark</div>
    <div>• 5 Attack Categories</div>
    <div>• 41 Network Features</div>
</div>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🛡️ Cybersecurity Network Threat & Intrusion Profiler</h1>
    <p class="subtitle">AI-Powered Network Intrusion Detection System · Classification + Anomaly Detection</p>
    <div class="badge-row">
        <span class="tech-badge">🤖 Random Forest</span>
        <span class="tech-badge">🔍 Isolation Forest</span>
        <span class="tech-badge">📊 NSL-KDD Dataset</span>
        <span class="tech-badge">⚡ Real-Time Analysis</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: THREAT ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

if page == "🛡️ Threat Analyzer":

    models = get_models()
    if models is None:
        st.error("⚠️ Models not found. Please run `python train_pipeline.py` first.")
        st.stop()

    feature_names = models['feature_names']
    scaler = models['scaler']
    label_encoder = models['label_encoder']

    # ── Helper: process a raw dataframe row and return result ─────────
    def _analyze_row(row_series):
        """Take a raw NSL-KDD row (with original columns), preprocess, and profile."""
        from src.preprocessing import CATEGORICAL_COLS, map_attack_labels

        row_df = pd.DataFrame([row_series])
        true_label = row_series.get('label', 'unknown')

        # Drop label & difficulty_level
        for col in ['label', 'difficulty_level', 'attack_category']:
            if col in row_df.columns:
                row_df = row_df.drop(columns=[col])

        # One-hot encode
        row_df = pd.get_dummies(row_df, columns=[c for c in CATEGORICAL_COLS if c in row_df.columns], drop_first=False, dtype=int)

        # Align to training feature set
        for fname in feature_names:
            if fname not in row_df.columns:
                row_df[fname] = 0
        row_df = row_df[feature_names]

        X = row_df.values.astype(np.float64)
        X_scaled = scaler.transform(X)
        result = profile_threat(X_scaled[0], models)
        result['true_label'] = true_label
        return result

    # ── Load test dataset ────────────────────────────────────────────
    @st.cache_data
    def load_test_data():
        test_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'KDDTest+.txt')
        if not os.path.exists(test_path):
            return None
        from src.data_loader import COLUMN_NAMES
        df = pd.read_csv(test_path, header=None, names=COLUMN_NAMES)
        return df

    test_df = load_test_data()

    # ── Two tabs: Quick Scan vs Advanced ─────────────────────────────
    tab1, tab2 = st.tabs(["🎯 Quick Scan (Recommended)", "⚙️ Advanced Manual Input"])

    # ══════════════════════════════════════════════════════════════════
    # TAB 1: QUICK SCAN — no knowledge needed
    # ══════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("""
        **How it works:** Click a button to load a real network traffic sample from the NSL-KDD test dataset.
        The AI models will instantly classify it and assess the threat level — no technical knowledge required!
        """)

        if test_df is None:
            st.error("Test dataset not found. Run `python train_pipeline.py` first.")
        else:
            col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
            with col_btn1:
                scan_random = st.button("🎲 Random Sample", use_container_width=True)
            with col_btn2:
                scan_normal = st.button("🟢 Normal Traffic", use_container_width=True)
            with col_btn3:
                scan_attack = st.button("🔴 Attack Traffic", use_container_width=True)
            with col_btn4:
                scan_batch = st.button("📊 Batch Scan (10)", use_container_width=True)

            sample = None
            batch_mode = False

            if scan_random:
                sample = test_df.sample(1, random_state=np.random.randint(0, 100000)).iloc[0]
            elif scan_normal:
                normals = test_df[test_df['label'] == 'normal']
                if len(normals) > 0:
                    sample = normals.sample(1, random_state=np.random.randint(0, 100000)).iloc[0]
            elif scan_attack:
                attacks = test_df[test_df['label'] != 'normal']
                if len(attacks) > 0:
                    sample = attacks.sample(1, random_state=np.random.randint(0, 100000)).iloc[0]
            elif scan_batch:
                batch_mode = True

            # ── Single sample result ─────────────────────────────────
            if sample is not None:
                result = _analyze_row(sample)

                st.markdown("---")
                st.markdown("### 🔎 Analysis Results")

                # Show what was loaded
                st.markdown(f"**Loaded sample** — Protocol: `{sample.get('protocol_type', '?')}` · "
                            f"Service: `{sample.get('service', '?')}` · Flag: `{sample.get('flag', '?')}` · "
                            f"True Label: `{result['true_label']}`")

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    status_icon = "🔴" if result['is_attack'] else "🟢"
                    status_text = "ATTACK DETECTED" if result['is_attack'] else "NORMAL"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">Classification</div>
                        <div class="value">{status_icon} {status_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">Attack Type</div>
                        <div class="value">{result['attack_type']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">Confidence</div>
                        <div class="value">{result['confidence']}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c4:
                    risk_class = f"risk-{result['risk_level'].lower()}"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">Risk Level</div>
                        <div class="value {risk_class}">{result['risk_level']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    anomaly_icon = "⚠️ ANOMALOUS" if result['is_anomaly'] else "✅ NORMAL"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">Anomaly Detection</div>
                        <div class="value">{anomaly_icon}</div>
                        <div style="color:#8888aa; font-size:0.8rem;">Score: {result['anomaly_score']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_b:
                    probs = result['class_probabilities']
                    fig = go.Figure(go.Bar(
                        x=list(probs.values()),
                        y=list(probs.keys()),
                        orientation='h',
                        marker_color=['#69f0ae' if k == 'normal' else '#ff5252' for k in probs.keys()]
                    ))
                    fig.update_layout(
                        title="Class Probabilities (%)",
                        xaxis_title="Probability %",
                        height=250,
                        margin=dict(l=0, r=0, t=40, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#ccc'),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # ── Batch scan result ────────────────────────────────────
            if batch_mode:
                st.markdown("---")
                st.markdown("### 📊 Batch Threat Scan — 10 Random Samples")

                batch_samples = test_df.sample(10, random_state=np.random.randint(0, 100000))
                batch_results = []
                for _, row in batch_samples.iterrows():
                    r = _analyze_row(row)
                    batch_results.append({
                        'Protocol': row.get('protocol_type', '?'),
                        'Service': row.get('service', '?'),
                        'True Label': r['true_label'],
                        'Predicted': r['classification'],
                        'Confidence': f"{r['confidence']}%",
                        'Anomaly': '⚠️ Yes' if r['is_anomaly'] else '✅ No',
                        'Risk Level': r['risk_level'],
                    })

                result_df = pd.DataFrame(batch_results)
                st.dataframe(result_df, use_container_width=True, hide_index=True)

                attacks_found = sum(1 for r in batch_results if r['Predicted'] != 'NORMAL')
                anomalies = sum(1 for r in batch_results if '⚠️' in r['Anomaly'])
                col_s1, col_s2, col_s3 = st.columns(3)
                col_s1.metric("Samples Scanned", 10)
                col_s2.metric("Attacks Detected", attacks_found)
                col_s3.metric("Anomalies Flagged", anomalies)

    # ══════════════════════════════════════════════════════════════════
    # TAB 2: ADVANCED MANUAL INPUT
    # ══════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("Enter network connection features manually for expert-level analysis.")

        with st.form("threat_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**🔌 Connection**")
                protocol_type = st.selectbox("Protocol Type", ["tcp", "udp", "icmp"])
                service = st.selectbox("Service", ["http", "smtp", "ftp", "ftp_data", "ssh", "telnet", "domain_u", "private", "pop_3", "finger", "other"])
                flag = st.selectbox("Flag", ["SF", "S0", "REJ", "RSTR", "RSTO", "SH", "S1", "S2", "RSTOS0", "S3", "OTH"])
                duration = st.number_input("Duration (sec)", min_value=0, value=0, step=1)

            with col2:
                st.markdown("**📊 Traffic**")
                src_bytes = st.number_input("Source Bytes", min_value=0, value=181, step=10)
                dst_bytes = st.number_input("Destination Bytes", min_value=0, value=5450, step=10)
                count = st.number_input("Count", min_value=0, value=8, step=1)
                srv_count = st.number_input("Srv Count", min_value=0, value=8, step=1)
                same_srv_rate = st.slider("Same Srv Rate", 0.0, 1.0, 1.0)
                diff_srv_rate = st.slider("Diff Srv Rate", 0.0, 1.0, 0.0)

            with col3:
                st.markdown("**🖥️ Host**")
                dst_host_count = st.number_input("Dst Host Count", min_value=0, max_value=255, value=9)
                dst_host_srv_count = st.number_input("Dst Host Srv Count", min_value=0, max_value=255, value=9)
                dst_host_same_srv_rate = st.slider("Dst Host Same Srv Rate", 0.0, 1.0, 1.0)
                dst_host_diff_srv_rate = st.slider("Dst Host Diff Srv Rate", 0.0, 1.0, 0.0)
                dst_host_serror_rate = st.slider("Dst Host SError Rate", 0.0, 1.0, 0.0)
                dst_host_rerror_rate = st.slider("Dst Host RError Rate", 0.0, 1.0, 0.0)

            submitted = st.form_submit_button("🔍  Analyze Threat", use_container_width=True)

        if submitted:
            with st.spinner("Analyzing network traffic..."):
                raw_features = {
                    'duration': duration, 'src_bytes': src_bytes, 'dst_bytes': dst_bytes,
                    'land': 0, 'wrong_fragment': 0, 'urgent': 0, 'hot': 0,
                    'num_failed_logins': 0, 'logged_in': 1, 'num_compromised': 0,
                    'root_shell': 0, 'su_attempted': 0, 'num_root': 0,
                    'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0,
                    'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0,
                    'count': count, 'srv_count': srv_count,
                    'serror_rate': 0.0, 'srv_serror_rate': 0.0,
                    'rerror_rate': 0.0, 'srv_rerror_rate': 0.0,
                    'same_srv_rate': same_srv_rate, 'diff_srv_rate': diff_srv_rate,
                    'srv_diff_host_rate': 0.0,
                    'dst_host_count': dst_host_count,
                    'dst_host_srv_count': dst_host_srv_count,
                    'dst_host_same_srv_rate': dst_host_same_srv_rate,
                    'dst_host_diff_srv_rate': dst_host_diff_srv_rate,
                    'dst_host_same_src_port_rate': 0.0,
                    'dst_host_srv_diff_host_rate': 0.0,
                    'dst_host_serror_rate': dst_host_serror_rate,
                    'dst_host_srv_serror_rate': 0.0,
                    'dst_host_rerror_rate': dst_host_rerror_rate,
                    'dst_host_srv_rerror_rate': 0.0,
                }

                feature_vector = {}
                for fname in feature_names:
                    if fname in raw_features:
                        feature_vector[fname] = raw_features[fname]
                    elif fname.startswith('protocol_type_'):
                        feature_vector[fname] = 1 if fname == f'protocol_type_{protocol_type}' else 0
                    elif fname.startswith('service_'):
                        feature_vector[fname] = 1 if fname == f'service_{service}' else 0
                    elif fname.startswith('flag_'):
                        feature_vector[fname] = 1 if fname == f'flag_{flag}' else 0
                    else:
                        feature_vector[fname] = 0

                X = np.array([feature_vector[f] for f in feature_names], dtype=np.float64).reshape(1, -1)
                X_scaled = scaler.transform(X)
                result = profile_threat(X_scaled[0], models)

            st.markdown("---")
            st.markdown("### 🔎 Analysis Results")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                status_icon = "🔴" if result['is_attack'] else "🟢"
                status_text = "ATTACK DETECTED" if result['is_attack'] else "NORMAL"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Classification</div>
                    <div class="value">{status_icon} {status_text}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Attack Type</div>
                    <div class="value">{result['attack_type']}</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Confidence</div>
                    <div class="value">{result['confidence']}%</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                risk_class = f"risk-{result['risk_level'].lower()}"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Risk Level</div>
                    <div class="value {risk_class}">{result['risk_level']}</div>
                </div>
                """, unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                anomaly_icon = "⚠️ ANOMALOUS" if result['is_anomaly'] else "✅ NORMAL"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Anomaly Detection</div>
                    <div class="value">{anomaly_icon}</div>
                    <div style="color:#8888aa; font-size:0.8rem;">Score: {result['anomaly_score']}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                probs = result['class_probabilities']
                fig = go.Figure(go.Bar(
                    x=list(probs.values()),
                    y=list(probs.keys()),
                    orientation='h',
                    marker_color=['#69f0ae' if k == 'normal' else '#ff5252' for k in probs.keys()]
                ))
                fig.update_layout(
                    title="Class Probabilities (%)",
                    xaxis_title="Probability %",
                    height=250,
                    margin=dict(l=0, r=0, t=40, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ccc'),
                )
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Analytics Dashboard":

    figures_dir = os.path.join(PROJECT_ROOT, 'reports', 'figures')

    if not os.path.exists(figures_dir):
        st.error("⚠️ No reports found. Please run `python train_pipeline.py` first.")
        st.stop()

    st.markdown('<div class="section-header">📈 Model Performance & Data Analytics</div>', unsafe_allow_html=True)

    # ── Row 1: Attack Distribution + Model Comparison ────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 Attack Category Distribution")
        img_path = os.path.join(figures_dir, 'attack_distribution.png')
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.info("Plot not available.")

    with col2:
        st.markdown("#### 🏆 Model Comparison")
        img_path = os.path.join(figures_dir, 'model_comparison.png')
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.info("Plot not available.")

    st.markdown("---")

    # ── Row 2: Confusion Matrices ────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### 🔢 Classification Confusion Matrix")
        img_path = os.path.join(figures_dir, 'clf_confusion_matrix.png')
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.info("Plot not available.")

    with col4:
        st.markdown("#### 🔍 Anomaly Detection Confusion Matrix")
        img_path = os.path.join(figures_dir, 'anomaly_confusion_matrix.png')
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.info("Plot not available.")

    st.markdown("---")

    # ── Row 3: Feature Importance + Anomaly Scores ───────────────────────
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("#### 📊 Top Feature Importances")
        img_path = os.path.join(figures_dir, 'feature_importance.png')
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.info("Plot not available.")

    with col6:
        st.markdown("#### 📉 Anomaly Score Distribution")
        img_path = os.path.join(figures_dir, 'anomaly_scores.png')
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.info("Plot not available.")

    # ── Model Info ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📋 Model Information</div>', unsafe_allow_html=True)

    try:
        best_name = joblib.load(os.path.join(PROJECT_ROOT, 'models', 'best_model_name.pkl'))
        le = joblib.load(os.path.join(PROJECT_ROOT, 'models', 'label_encoder.pkl'))

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Best Classifier</div>
                <div class="value" style="color: #6366f1;">🏆 {best_name}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Anomaly Detector</div>
                <div class="value" style="color: #8b5cf6;">🔍 Isolation Forest</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box">
            <strong>Target Classes:</strong> {', '.join(le.classes_)}<br>
            <strong>Dataset:</strong> NSL-KDD (Network Security Laboratory - Knowledge Discovery in Databases)<br>
            <strong>Features:</strong> 41 network traffic attributes · 5 attack categories
        </div>
        """, unsafe_allow_html=True)

    except Exception:
        st.info("Model info not available. Run training pipeline first.")


# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div class="pro-footer">
    <div class="footer-title">Cybersecurity Network Threat & Intrusion Profiler Using Machine Learning</div>
    <div class="footer-badges">
        <span class="footer-badge">Python 3.11</span>
        <span class="footer-badge">scikit-learn</span>
        <span class="footer-badge">Streamlit</span>
        <span class="footer-badge">Plotly</span>
        <span class="footer-badge">Pandas</span>
        <span class="footer-badge">NSL-KDD</span>
    </div>
    <div style="margin-top: 0.75rem; color: rgba(148,163,184,0.5); font-size: 0.72rem;">
        IBM SkillsBuild Project Submission · Built with ❤️ using Machine Learning
    </div>
</div>
""", unsafe_allow_html=True)
