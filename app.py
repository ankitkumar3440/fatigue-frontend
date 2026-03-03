"""
FatigueDetect — Streamlit frontend
Deploy on Streamlit Cloud (free)
"""

import streamlit as st
import requests, io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FatigueDetect",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');

* { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: #080c14;
    color: #cbd5e1;
}
.stApp { background: #080c14; }

/* ── Hero ── */
.hero {
    padding: 36px 44px 28px;
    margin-bottom: 28px;
    background: linear-gradient(135deg,#0c1221 0%,#0f1a35 60%,#0c1221 100%);
    border: 1px solid #1e2d4a;
    border-radius: 20px;
    position: relative; overflow: hidden;
}
.hero::after {
    content:'';
    position:absolute; top:0; right:0; width:320px; height:100%;
    background: radial-gradient(ellipse at 80% 50%, rgba(56,189,248,.06) 0%, transparent 70%);
    pointer-events:none;
}
.hero-eyebrow {
    font-family:'IBM Plex Mono',monospace;
    font-size:.7rem; letter-spacing:.2em; text-transform:uppercase;
    color:#38bdf8; margin-bottom:10px;
}
.hero-title {
    font-size:2.4rem; font-weight:700; line-height:1.15;
    color:#f1f5f9; margin:0 0 12px;
}
.hero-title span { color:#38bdf8; }
.hero-desc { font-size:.95rem; color:#64748b; font-weight:300; max-width:560px; }

/* ── Stat cards ── */
.stat-row { display:flex; gap:14px; margin-bottom:24px; }
.stat {
    flex:1; background:#0c1524; border:1px solid #1e2d4a;
    border-radius:14px; padding:18px 22px;
}
.stat-val {
    font-family:'IBM Plex Mono',monospace;
    font-size:1.6rem; font-weight:600; color:#38bdf8;
}
.stat-lbl { font-size:.72rem; color:#475569; text-transform:uppercase; letter-spacing:.12em; margin-top:4px; }

/* ── Upload zone ── */
[data-testid="stFileUploader"] > div {
    background:#0c1524 !important;
    border:1.5px dashed #1e3a5f !important;
    border-radius:14px !important;
    transition: border-color .2s;
}
[data-testid="stFileUploader"] > div:hover {
    border-color:#38bdf8 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    font-family:'IBM Plex Mono',monospace;
    font-size:.8rem; letter-spacing:.06em;
    color:#64748b; border-bottom:2px solid transparent;
    padding:10px 18px;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color:#38bdf8 !important;
    border-bottom-color:#38bdf8 !important;
}

/* ── Button ── */
.stButton > button {
    background:linear-gradient(135deg,#0ea5e9,#2563eb) !important;
    color:#fff !important; border:none !important;
    border-radius:10px !important;
    font-family:'IBM Plex Mono',monospace !important;
    font-size:.85rem !important; letter-spacing:.08em !important;
    padding:14px 36px !important; width:100% !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 12px 28px rgba(14,165,233,.35) !important;
}

/* ── Result card ── */
.result-card {
    border-radius:16px; padding:28px 36px; margin:16px 0;
    position:relative; overflow:hidden;
}
.result-ok  { background:rgba(20,83,45,.15);  border:1px solid rgba(34,197,94,.25); }
.result-bad { background:rgba(127,29,29,.15); border:1px solid rgba(239,68,68,.25); }
.result-status {
    font-family:'IBM Plex Mono',monospace;
    font-size:1.9rem; font-weight:600; margin-bottom:8px;
}
.result-msg { font-size:.9rem; color:#94a3b8; line-height:1.65; }

/* ── Activity cards ── */
.act {
    background:#0c1524; border:1px solid #1e2d4a;
    border-radius:10px; padding:13px 16px; margin-bottom:9px;
    display:flex; align-items:center; gap:12px;
}
.act-name { font-size:.9rem; font-weight:500; flex:1; }
.badge {
    background:#0f172a; border:1px solid #1e3a5f;
    border-radius:6px; padding:2px 9px;
    font-family:'IBM Plex Mono',monospace;
    font-size:.7rem; color:#64748b; white-space:nowrap;
}
.avoid-item {
    background:rgba(239,68,68,.05);
    border:1px solid rgba(239,68,68,.18);
    border-radius:8px; padding:10px 14px;
    margin-bottom:8px; color:#fca5a5; font-size:.88rem;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background:#0c1524 !important;
    border:1px solid #1e2d4a !important;
    border-radius:10px !important;
    padding:14px 18px !important;
}
[data-testid="stMetricValue"] {
    font-family:'IBM Plex Mono',monospace !important;
    color:#38bdf8 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background:#060a12 !important; border-right:1px solid #0f1e33; }
[data-testid="stSidebar"] * { color:#64748b !important; }
[data-testid="stSidebar"] h3 { color:#94a3b8 !important; }
.api-badge {
    padding:7px 12px; border-radius:8px; font-size:.78rem;
    font-family:'IBM Plex Mono',monospace; margin-top:6px;
}
.online  { background:rgba(20,83,45,.3);  border:1px solid rgba(34,197,94,.3);  color:#4ade80 !important; }
.offline { background:rgba(127,29,29,.3); border:1px solid rgba(239,68,68,.3);  color:#f87171 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
API = st.secrets.get("API_BASE", "http://localhost:8000")

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0a0f1e",
    font=dict(color="#64748b", family="IBM Plex Mono, monospace", size=11),
    margin=dict(l=10, r=10, t=16, b=40),
)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎧 FatigueDetect")
    st.markdown("---")

    # API health
    online = False
    try:
        h = requests.get(f"{API}/health", timeout=5)
        if h.ok:
            d = h.json()
            online = True
            st.markdown(
                f'<div class="api-badge online">● API Online &nbsp;|&nbsp; '
                f'{d.get("features","?")} features</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="api-badge offline">● API Error</div>',
                        unsafe_allow_html=True)
    except:
        st.markdown('<div class="api-badge offline">● API Offline</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown("- OmniBuds-418B (50 Hz)")
    st.markdown("- PPG bandpass 0.5–4 Hz")
    st.markdown("- 10s windows, 5s step")
    st.markdown("- Nonlinear HRV: SD1/SD2/ApEn")
    st.markdown("- Ensemble: RF+XGB+GB+SVM")
    st.markdown("- LOGO-CV validated (14 subjects)")
    st.markdown("---")
    st.markdown("**CSV columns needed**")
    st.code("Peripheral ID\nValue 1 / Value 2 / Value 3\nTimestamp")
    st.caption("PID 0=Acc  1=Gyro  5=PPG")

# ─────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Ear-worn biometric analysis</div>
  <div class="hero-title">🎧 Fatigue<span>Detect</span></div>
  <div class="hero-desc">
    Upload an OmniBuds CSV recording to analyse fatigue state via
    nonlinear HRV &amp; IMU features. Get personalised activity recommendations.
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# File upload
# ─────────────────────────────────────────────
uploaded = st.file_uploader("Upload OmniBuds CSV recording", type=["csv"])

if uploaded is None:
    st.markdown("""
    <div class="stat-row">
      <div class="stat"><div class="stat-val">10s</div><div class="stat-lbl">Window Size</div></div>
      <div class="stat"><div class="stat-val">17</div><div class="stat-lbl">HRV Features</div></div>
      <div class="stat"><div class="stat-val">4×</div><div class="stat-lbl">Model Ensemble</div></div>
      <div class="stat"><div class="stat-val">85.7%</div><div class="stat-lbl">LOGO-CV Accuracy</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

file_bytes = uploaded.read()

# ─────────────────────────────────────────────
# Fetch signal stats (for metrics bar)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_stats(fb):
    try:
        r = requests.post(f"{API}/signal-stats",
                          files={"file": ("data.csv", fb, "text/csv")},
                          timeout=30)
        return r.json() if r.ok else None
    except:
        return None

stats = get_stats(file_bytes)

if stats:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Duration",    f"{stats['duration_sec']}s")
    c2.metric("PPG Samples", f"{stats['ppg_samples']:,}")
    c3.metric("Acc Samples", f"{stats['acc_samples']:,}")
    c4.metric("Motion",      f"{stats['motion_pct']}%")
    st.markdown("")

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
TAB_VIZ, TAB_PRED, TAB_ACT = st.tabs(
    ["📈  Signal View", "🧠  Fatigue Analysis", "🏃  Activities"])

# ══════════════════════════
# TAB 1 — SIGNAL VIEW
# ══════════════════════════
with TAB_VIZ:
    st.markdown("#### Raw Sensor Signals")

    # ── PPG chart ──
    if stats and "ppg_chart" in stats:
        t = np.array(stats["ppg_chart"]["t"])
        v = np.array(stats["ppg_chart"]["v"])

        # Smoothed overlay via rolling mean
        v_s = pd.Series(v).rolling(15, center=True, min_periods=1).mean().values

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=v, name="PPG raw",
            line=dict(color="rgba(56,189,248,.25)", width=0.7)))
        fig.add_trace(go.Scatter(x=t, y=v_s, name="PPG smoothed",
            line=dict(color="#38bdf8", width=1.6)))
        fig.update_layout(**CHART_THEME, height=260,
            xaxis=dict(title="Time (s)", gridcolor="#111827"),
            yaxis=dict(title="ADC value", gridcolor="#111827"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            title=dict(text="PPG Signal — OmniBuds-418B (PID 5)",
                       font=dict(size=12, color="#94a3b8")))
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Local parse fallback
        try:
            df_v = pd.read_csv(io.BytesIO(file_bytes))
            df_v.columns = df_v.columns.str.strip()
            pid_col = next((c for c in df_v.columns if "peripheral" in c.lower()), None)
            if pid_col:
                df_v[pid_col] = pd.to_numeric(df_v[pid_col], errors="coerce")
                ppg_rows = df_v[df_v[pid_col].isin([2,5])]
                v3 = next((c for c in df_v.columns if "value 3" in c.lower() or c.lower()=="value3"), None)
                if v3 and not ppg_rows.empty:
                    sig = pd.to_numeric(ppg_rows[v3], errors="coerce").dropna().values
                    step = max(1, len(sig)//1500)
                    t_p = np.arange(len(sig))[::step]/50
                    fig2 = go.Figure(go.Scatter(x=t_p, y=sig[::step],
                        line=dict(color="#38bdf8", width=1.2)))
                    fig2.update_layout(**CHART_THEME, height=250,
                        xaxis=dict(title="Time (s)", gridcolor="#111827"),
                        yaxis=dict(title="PPG (ADC)", gridcolor="#111827"))
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("PPG data not found. Check PID column.")
        except Exception as ex:
            st.warning(f"Could not render chart locally: {ex}")

    # ── Sensor breakdown ──
    st.markdown("#### Sensor Row Counts")
    try:
        df_s = pd.read_csv(io.BytesIO(file_bytes))
        df_s.columns = df_s.columns.str.strip()
        pc = next((c for c in df_s.columns if "peripheral" in c.lower()), None)
        if pc:
            df_s[pc] = pd.to_numeric(df_s[pc], errors="coerce")
            cnts = df_s.groupby(pc).size().reset_index(name="Rows")
            pid_lbl = {0:"Accelerometer",1:"Gyroscope",2:"PPG",5:"PPG"}
            cnts["Sensor"] = cnts[pc].map(pid_lbl).fillna("Unknown")
            st.dataframe(cnts.rename(columns={pc:"PID"})[["Sensor","PID","Rows"]],
                         use_container_width=True, hide_index=True)
    except: pass

# ══════════════════════════
# TAB 2 — FATIGUE ANALYSIS
# ══════════════════════════
with TAB_PRED:
    st.markdown("#### Run Fatigue Analysis")
    st.caption("Runs ensemble model on every 10-second window, then averages "
               "probability across windows for a subject-level decision.")

    if st.button("🔬  Analyse Now", type="primary"):
        with st.spinner("Running ensemble model …"):
            try:
                resp = requests.post(
                    f"{API}/predict",
                    files={"file": ("data.csv", file_bytes, "text/csv")},
                    timeout=120)
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach API at **{API}**\n\n"
                         "Check `.streamlit/secrets.toml` → `API_BASE`")
                st.stop()
            except requests.exceptions.Timeout:
                st.error("Timeout (>120s). Render free tier may be cold-starting. "
                         "Wait 30s and try again.")
                st.stop()

        if resp.status_code != 200:
            st.error(f"API {resp.status_code}: {resp.text}")
            st.stop()

        res = resp.json()
        st.session_state["result"] = res

    if "result" not in st.session_state:
        st.stop()

    res  = st.session_state["result"]
    pred = res["prediction"]
    prob = res["probability"]
    conf = res["confidence"]
    clr  = res["color"]
    cls  = "result-ok" if pred == 0 else "result-bad"

    # Result card
    st.markdown(f"""
    <div class="result-card {cls}">
      <div class="result-status" style="color:{clr}">{res['emoji']} {res['status']}</div>
      <div class="result-msg">{res['message']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Fatigue Probability", f"{prob*100:.1f}%")
    c2.metric("Confidence",          f"{conf:.1f}%")
    c3.metric("Recording Duration",  f"{res['duration_sec']}s")
    c4.metric("Windows Analysed",    res["n_windows"])

    st.markdown("")
    left, right = st.columns([1, 1])

    # Gauge
    with left:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob*100, 1),
            title=dict(text="P(Fatigue) %",
                       font=dict(color="#64748b", size=13)),
            number=dict(font=dict(color=clr, size=42,
                                  family="IBM Plex Mono, monospace")),
            gauge=dict(
                axis=dict(range=[0,100], tickfont=dict(color="#475569"), tickcolor="#1e2d4a"),
                bar=dict(color=clr, thickness=0.22),
                bgcolor="#0a0f1e", bordercolor="#1e2d4a",
                steps=[dict(range=[0,45],  color="rgba(34,197,94,.10)"),
                       dict(range=[45,60], color="rgba(250,204,21,.07)"),
                       dict(range=[60,100],color="rgba(239,68,68,.10)")],
                threshold=dict(line=dict(color="#f1f5f9",width=2),
                               thickness=0.65, value=45)
            )
        ))
        fig_g.update_layout(**CHART_THEME, height=260)
        st.plotly_chart(fig_g, use_container_width=True)

    # Window timeline
    with right:
        wins = res.get("windows",[])
        if wins:
            t_w = [w["t"]    for w in wins]
            p_w = [w["prob"] for w in wins]
            bar_clr = ["#ef4444" if p >= 0.45 else "#22c55e" for p in p_w]

            fig_t = go.Figure()
            fig_t.add_hline(y=0.45, line_color="#f59e0b",
                             line_dash="dot", line_width=1.2)
            fig_t.add_trace(go.Bar(x=t_w, y=p_w, marker_color=bar_clr,
                name="P(Fatigue) per window",
                hovertemplate="t=%{x:.1f}s<br>prob=%{y:.3f}<extra></extra>"))
            fig_t.update_layout(**CHART_THEME, height=260,
                xaxis=dict(title="Time (s)", gridcolor="#111827"),
                yaxis=dict(title="P(Fatigue)", range=[0,1.05], gridcolor="#111827"),
                showlegend=False,
                title=dict(text="Per-window Fatigue Probability",
                           font=dict(size=12,color="#94a3b8")))
            st.plotly_chart(fig_t, use_container_width=True)

# ══════════════════════════
# TAB 3 — ACTIVITIES
# ══════════════════════════
with TAB_ACT:
    st.markdown("#### Personalised Activity Recommendations")

    if "result" not in st.session_state:
        st.info("Run the analysis in **Fatigue Analysis** tab first.")
        st.stop()

    res  = st.session_state["result"]
    pred = res["prediction"]
    clr  = res["color"]
    cls  = "result-ok" if pred == 0 else "result-bad"

    st.markdown(f"""
    <div class="result-card {cls}">
      <div class="result-status" style="color:{clr};font-size:1.4rem">
        {res['emoji']} {res['status']}
      </div>
      <div class="result-msg">{res['message']}</div>
    </div>
    """, unsafe_allow_html=True)

    col_rec, col_avo = st.columns([3, 2])

    with col_rec:
        st.markdown("##### ✅ Recommended")
        for a in res.get("recommended",[]):
            st.markdown(f"""
            <div class="act">
              <span class="act-name">{a['name']}</span>
              <span class="badge">{a['intensity']}</span>
              <span class="badge">{a['duration']}</span>
            </div>""", unsafe_allow_html=True)

    with col_avo:
        st.markdown("##### ⚠️ Avoid")
        for item in res.get("avoid",[]):
            st.markdown(f'<div class="avoid-item">— {item}</div>',
                        unsafe_allow_html=True)

    # Suitability bar chart
    st.markdown("")
    st.markdown("##### Activity Suitability")

    acts   = ["Intense Training","Moderate Exercise","Light Walk","Desk Work","Rest / Sleep"]
    scores = ([0.93, 0.82, 0.60, 0.50, 0.20] if pred==0
               else [0.05, 0.18, 0.75, 0.55, 0.95])
    bar_c  = ["#22c55e" if s>=.7 else "#f59e0b" if s>=.4 else "#ef4444" for s in scores]

    fig_s = go.Figure(go.Bar(
        x=scores, y=acts, orientation="h",
        marker_color=bar_c,
        text=[f"{s*100:.0f}%" for s in scores],
        textposition="outside",
        textfont=dict(color="#cbd5e1", size=12,
                      family="IBM Plex Mono, monospace"),
        hovertemplate="%{y}: %{x:.0%}<extra></extra>"
    ))
    fig_s.update_layout(**CHART_THEME, height=260,
        xaxis=dict(range=[0,1.2], title="Suitability Score", gridcolor="#111827",
                   tickformat=".0%"),
        yaxis=dict(gridcolor="#111827", tickfont=dict(color="#94a3b8")),
        title=dict(text="Activity Suitability Based on Fatigue State",
                   font=dict(size=12,color="#94a3b8")))
    st.plotly_chart(fig_s, use_container_width=True)
