"""
FatigueDetect v3 — Streamlit Frontend
Joint: Fatigue (PPG) + Activity (IMU both earbuds)
"""

import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io, json

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="FatigueDetect",
    page_icon="🧠",
    layout="wide",
)

# ── Constants ────────────────────────────────────────────────
BACKEND_URL = "https://fatigue-backend-2.onrender.com"
FS          = 50
PPG_SAMPLES = 1000   # 20 s
IMU_SAMPLES = 200    #  4 s

ACTIVITY_LABELS = [
    "ideal", "beard_pulling", "face_itching", "hair_pulling", "nail_biting"
]
ACTIVITY_COLORS = {
    "ideal":         "#78909C",
    "beard_pulling": "#2196F3",
    "face_itching":  "#FF9800",
    "hair_pulling":  "#F44336",
    "nail_biting":   "#4CAF50",
}
ACTIVITY_ICONS = {
    "ideal":         "😌",
    "beard_pulling": "🧔",
    "face_itching":  "😣",
    "hair_pulling":  "💇",
    "nail_biting":   "💅",
}

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
.result-card {
    border-radius: 12px; padding: 20px; margin: 8px 0;
    text-align: center; font-size: 1.1em;
}
.fatigue-high  { background: #ffebee; border-left: 5px solid #f44336; }
.fatigue-low   { background: #e8f5e9; border-left: 5px solid #4caf50; }
.activity-card { background: #e3f2fd; border-left: 5px solid #2196f3; }
.joint-card    { background: #f3e5f5; border-left: 5px solid #9c27b0; }
.metric-big    { font-size: 2em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Title ────────────────────────────────────────────────────
st.title("🧠 FatigueDetect v3")
st.caption("Joint Fatigue Detection (PPG) + Activity Recognition (IMU, both earbuds)")

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    backend = st.text_input("Backend URL", value=BACKEND_URL)

    st.subheader("Backend Status")
    if st.button("🔄 Check Status"):
        try:
            r = requests.get(f"{backend}/health", timeout=10)
            h = r.json()
            if h.get("models_loaded"):
                st.success("✅ Both models loaded")
                st.json(h)
            else:
                st.warning("⚠️ Models not fully loaded")
                st.json(h)
        except Exception as e:
            st.error(f"❌ {e}")

    st.markdown("---")
    st.subheader("📋 Column Mapping")
    st.caption("Map CSV columns to sensors")
    ppg_col    = st.text_input("PPG column",       "Value 3")
    acc_x_col  = st.text_input("ACC X column",     "Value 1")
    acc_y_col  = st.text_input("ACC Y column",     "Value 2")
    acc_z_col  = st.text_input("ACC Z column",     "Value 3")
    gyro_x_col = st.text_input("GYRO X column",    "Value 1")
    gyro_y_col = st.text_input("GYRO Y column",    "Value 2")
    gyro_z_col = st.text_input("GYRO Z column",    "Value 3")
    ts_col     = st.text_input("Timestamp column", "Timestamp")

    st.markdown("---")
    st.caption("Peripheral ID filter")
    ppg_pid  = st.number_input("PPG PID",  value=5, step=1)
    acc_pid  = st.number_input("ACC PID",  value=0, step=1)
    gyro_pid = st.number_input("GYRO PID", value=1, step=1)

# ── Tabs ─────────────────────────────────────────────────────

def _show_results(res):
    st.markdown("---")
    st.subheader("📊 Results")

    # ── Top cards ────────────────────────────────────────────
    col_f, col_a, col_j = st.columns(3)

    fat_cls   = "fatigue-high" if res["fatigue_binary"] else "fatigue-low"
    fat_emoji = "🔴" if res["fatigue_binary"] else "🟢"

    with col_f:
        st.markdown(f"""
        <div class="result-card {fat_cls}">
            <div style="font-size:1.5em">{fat_emoji} Fatigue</div>
            <div class="metric-big">{res['fatigue_label']}</div>
            <div>Probability: {res['fatigue_prob']:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    act_col = ACTIVITY_COLORS.get(res["activity_label"], "#607D8B")
    act_ico = ACTIVITY_ICONS.get(res["activity_label"], "🤔")
    with col_a:
        st.markdown(f"""
        <div class="result-card activity-card"
             style="border-left-color:{act_col}">
            <div style="font-size:1.5em">{act_ico} Activity</div>
            <div class="metric-big">{res['activity_label'].replace('_',' ').title()}</div>
            <div>Confidence: {res['activity_conf']:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_j:
        st.markdown(f"""
        <div class="result-card joint-card">
            <div style="font-size:1.5em">🔗 Joint</div>
            <div class="metric-big" style="font-size:1.3em">{res['joint_label']}</div>
            <div>&nbsp;</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Activity probability bar chart ───────────────────────
    st.markdown("#### Activity Probabilities")
    act_probs = res.get("activity_probs", {})
    if act_probs:
        df_ap = pd.DataFrame({
            "Activity": [k.replace("_"," ").title() for k in act_probs.keys()],
            "Probability": list(act_probs.values()),
            "Color": [ACTIVITY_COLORS.get(k,"#607D8B") for k in act_probs.keys()],
        }).sort_values("Probability", ascending=True)

        fig = go.Figure(go.Bar(
            x=df_ap["Probability"],
            y=df_ap["Activity"],
            orientation="h",
            marker_color=df_ap["Color"],
            text=[f"{v:.1%}" for v in df_ap["Probability"]],
            textposition="outside",
        ))
        fig.update_layout(height=250, margin=dict(l=10,r=60,t=10,b=10),
                          xaxis=dict(range=[0,1.05], tickformat=".0%"),
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # ── Raw JSON ─────────────────────────────────────────────
    with st.expander("Raw JSON response"):
        st.json(res)

tab_upload, tab_manual, tab_about = st.tabs(
    ["📁 Upload CSV", "🔢 Manual Input", "ℹ️ About"]
)

# ============================================================
# TAB 1: CSV UPLOAD
# ============================================================
with tab_upload:
    st.subheader("Upload sensor CSV file")
    st.caption(
        "Upload your merged labeled CSV. Must contain both left & right earbud "
        "rows with PPG (PID=5 or 2), Accelerometer (PID=0), Gyroscope (PID=1)."
    )

    uploaded = st.file_uploader("Choose CSV", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        df_raw.columns = df_raw.columns.str.strip()

        st.success(f"Loaded {len(df_raw):,} rows, {len(df_raw.columns)} columns")
        with st.expander("Preview raw data"):
            st.dataframe(df_raw.head(50))

        # ── Extraction helper ────────────────────────────────
        def extract_sensor(df, pid, earbud=None, val_col="Value 3", n=None):
            mask = df["Peripheral ID"] == pid
            if earbud:
                mask &= df["Earbud"].str.strip().str.lower() == earbud
            sub = df[mask].sort_values(ts_col)
            arr = pd.to_numeric(sub[val_col], errors="coerce").ffill().dropna().values
            arr = arr[:n] if n and len(arr)>=n else arr
            return arr

        def extract_imu_3ax(df, pid, earbud, n=None):
            mask = (df["Peripheral ID"] == pid) & \
                   (df["Earbud"].str.strip().str.lower() == earbud)
            sub  = df[mask].sort_values(ts_col)
            x = pd.to_numeric(sub[acc_x_col],  errors="coerce").ffill().values
            y = pd.to_numeric(sub[acc_y_col],  errors="coerce").ffill().values
            z = pd.to_numeric(sub[acc_z_col],  errors="coerce").ffill().values
            min_n = min(len(x),len(y),len(z))
            if n: min_n = min(min_n, n)
            return np.stack([x[:min_n],y[:min_n],z[:min_n]],axis=1)

        # ── Validate & extract ───────────────────────────────
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**PPG**")
            ppg_all = extract_sensor(df_raw, ppg_pid, val_col=ppg_col)
            st.write(f"Samples: {len(ppg_all)}")
            ok_ppg = len(ppg_all) >= PPG_SAMPLES
            st.write("✅ Enough" if ok_ppg else f"⚠️ Need {PPG_SAMPLES}")

        with col_b:
            st.markdown("**Left IMU**")
            la = extract_imu_3ax(df_raw, acc_pid,  "left",  IMU_SAMPLES)
            lg = extract_imu_3ax(df_raw, gyro_pid, "left",  IMU_SAMPLES)
            st.write(f"ACC: {la.shape}  GYRO: {lg.shape}")
            ok_l = la.shape[0]>=IMU_SAMPLES and lg.shape[0]>=IMU_SAMPLES
            st.write("✅ OK" if ok_l else "⚠️ Insufficient")

        with col_c:
            st.markdown("**Right IMU**")
            ra = extract_imu_3ax(df_raw, acc_pid,  "right", IMU_SAMPLES)
            rg = extract_imu_3ax(df_raw, gyro_pid, "right", IMU_SAMPLES)
            st.write(f"ACC: {ra.shape}  GYRO: {rg.shape}")
            ok_r = ra.shape[0]>=IMU_SAMPLES and rg.shape[0]>=IMU_SAMPLES
            st.write("✅ OK" if ok_r else "⚠️ Insufficient")

        # ── Signal visualisation ─────────────────────────────
        with st.expander("📈 View signals"):
            fig = make_subplots(rows=3, cols=1, shared_xaxes=False,
                                subplot_titles=["PPG (first 20s)",
                                                "Left ACC magnitude",
                                                "Right ACC magnitude"])
            t_ppg = np.arange(min(len(ppg_all),PPG_SAMPLES)) / FS
            fig.add_trace(go.Scatter(x=t_ppg, y=ppg_all[:len(t_ppg)],
                                     line=dict(color="#E91E63",width=1),
                                     name="PPG"), row=1,col=1)
            if la.shape[0]>0:
                t_imu = np.arange(min(la.shape[0],IMU_SAMPLES)) / FS
                lam = np.sqrt(np.sum(la[:len(t_imu)]**2,axis=1))
                ram = np.sqrt(np.sum(ra[:len(t_imu)]**2,axis=1))
                fig.add_trace(go.Scatter(x=t_imu,y=lam,
                                         line=dict(color="#1565C0",width=1),
                                         name="Left ACC"),  row=2,col=1)
                fig.add_trace(go.Scatter(x=t_imu,y=ram,
                                         line=dict(color="#C62828",width=1),
                                         name="Right ACC"), row=3,col=1)
            fig.update_layout(height=500, showlegend=True,
                              template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        # ── Predict button ───────────────────────────────────
        st.markdown("---")
        can_predict = ok_ppg and ok_l and ok_r

        if not can_predict:
            st.warning("⚠️ Not enough data in one or more sensors to run prediction.")

        if st.button("🚀 Analyse", disabled=not can_predict, type="primary"):
            payload = {
                "ppg":        ppg_all[:PPG_SAMPLES].tolist(),
                "left_acc":   la[:IMU_SAMPLES].tolist(),
                "left_gyro":  lg[:IMU_SAMPLES].tolist(),
                "right_acc":  ra[:IMU_SAMPLES].tolist(),
                "right_gyro": rg[:IMU_SAMPLES].tolist(),
            }

            with st.spinner("Running joint analysis..."):
                try:
                    resp = requests.post(f"{backend}/predict",
                                         json=payload, timeout=60)
                    if resp.status_code == 200:
                        res = resp.json()
                        _show_results(res)
                    else:
                        st.error(f"Backend error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")


# ============================================================
# TAB 2: MANUAL INPUT
# ============================================================
with tab_manual:
    st.subheader("Generate synthetic signals for testing")
    st.caption("This uses random signals with configurable patterns to test the API.")

    c1, c2, c3 = st.columns(3)
    with c1:
        hr_bpm  = st.slider("Simulated HR (bpm)", 50, 110, 70)
        fat_lvl = st.select_slider("Fatigue level",
                                    options=["Low","Medium","High"], value="Low")
    with c2:
        noise   = st.slider("Signal noise", 0.0, 0.5, 0.05, step=0.01)
        sim_act = st.selectbox("Simulate activity",
                                ["ideal","beard_pulling","face_itching",
                                 "hair_pulling","nail_biting"])
    with c3:
        seed = st.number_input("Random seed", value=42)

    if st.button("🎲 Generate & Predict", type="primary"):
        rng = np.random.default_rng(int(seed))
        t_ppg = np.arange(PPG_SAMPLES) / FS
        hr_hz = hr_bpm / 60
        ppg   = (np.sin(2*np.pi*hr_hz*t_ppg) * 1000 +
                 rng.normal(0, noise*200, PPG_SAMPLES)).tolist()

        # Simulate IMU — different amplitude per activity
        amp_map = {"ideal":0.1, "beard_pulling":0.8, "face_itching":0.4,
                   "hair_pulling":0.6,  "nail_biting":0.5}
        amp = amp_map[sim_act]
        t_imu = np.arange(IMU_SAMPLES) / FS

        def imu3(amp, freq=1.0):
            return np.stack([
                amp*np.sin(2*np.pi*freq*t_imu) + rng.normal(0,noise,IMU_SAMPLES),
                amp*np.cos(2*np.pi*freq*t_imu) + rng.normal(0,noise,IMU_SAMPLES),
                rng.normal(0,amp*0.3+noise,   IMU_SAMPLES),
            ], axis=1).tolist()

        payload = {
            "ppg":        ppg,
            "left_acc":   imu3(amp, 1.2),
            "left_gyro":  imu3(amp*0.5, 1.2),
            "right_acc":  imu3(amp*0.9, 1.1),
            "right_gyro": imu3(amp*0.4, 1.1),
        }

        with st.spinner("Running prediction..."):
            try:
                resp = requests.post(f"{backend}/predict",
                                      json=payload, timeout=60)
                if resp.status_code == 200:
                    _show_results(resp.json())
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Failed: {e}")


# ============================================================
# SHARED RESULT DISPLAY
# ============================================================
# Move function def before the tabs (Python requires it defined before call)
# ============================================================
# TAB 3: ABOUT
# ============================================================
with tab_about:
    st.subheader("About FatigueDetect v3")
    st.markdown("""
    ### Pipeline

    ```
    CSV Upload
        │
        ├─ PPG signal (20s @ 50Hz)
        │       └── HRV features (time + freq + morphology)
        │               └── Ensemble (RF + XGB + GB)
        │                       └── 🔴/🟢 Fatigue (Not Fatigued / Fatigued)
        │
        └─ IMU both earbuds (4s @ 50Hz each, 12 channels)
                └── 280 features (stat + freq + mag + cross-earbud)
                        └── XGBoost + Signal Augmentation  ← best model
                                └── 🔵 Activity class (5-class)
                                    ideal / beard_pulling /
                                    face_itching / hair_pulling /
                                    nail_biting
    ```

    > **Note:** Annotation labels `face_scratching` and `face itching` both
    > map to the `face_itching` class automatically.

    ### Models
    | Model | Task | Validation | Result |
    |-------|------|-----------|--------|
    | Ensemble RF+XGB+GB | Fatigue (binary) | LOGO-CV, 14 subjects | ~85.7% acc |
    | XGBoost + Signal Augmentation | Activity (5-class) | Stratified 5-Fold | **F1 = 0.958** |

    ### Full classifier comparison (your experimental results)
    | Classifier | Strategy | F1 | ΔF1 |
    |---|---|---|---|
    | Random Forest | Standard (1s step) | 0.734 | baseline |
    | Random Forest | Dense (0.5s step) | 0.890 | +0.156 ✅ |
    | Random Forest | Signal Augmentation | 0.946 | +0.212 ✅ |
    | XGBoost | Standard (1s step) | 0.700 | -0.034 |
    | XGBoost | Dense (0.5s step) | 0.901 | +0.167 ✅ |
    | **XGBoost** | **Signal Augmentation** | **0.958** | **+0.224 ✅ BEST** |
    | SVM (RBF) | Standard (1s step) | 0.720 | -0.014 |
    | SVM (RBF) | Dense (0.5s step) | 0.875 | +0.141 ✅ |
    | SVM (RBF) | Signal Augmentation | 0.929 | +0.195 ✅ |
    | Gradient Boosting | Standard (1s step) | 0.726 | -0.008 |
    | Gradient Boosting | Dense (0.5s step) | 0.909 | +0.175 ✅ |
    | Gradient Boosting | Signal Augmentation | 0.955 | +0.221 ✅ |

    ### Activity classes
    | Class | Annotation variants recognised |
    |-------|-------------------------------|
    | `ideal` | ideal, idle, rest |
    | `beard_pulling` | beard pulling, beard_pulling |
    | `face_itching` | face itching, face_itching, face scratching, face_scratching |
    | `hair_pulling` | hair pulling, hair_pulling |
    | `nail_biting` | nail biting, nail_biting |

    ### API input format
    | Field | Shape | Description |
    |-------|-------|-------------|
    | `ppg` | (1000,) | 20 s PPG @ 50Hz |
    | `left_acc` | (200,3) | 4 s left ACC @ 50Hz |
    | `left_gyro` | (200,3) | 4 s left GYRO @ 50Hz |
    | `right_acc` | (200,3) | 4 s right ACC @ 50Hz |
    | `right_gyro` | (200,3) | 4 s right GYRO @ 50Hz |
    """)
