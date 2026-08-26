from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yaml
from streamlit_autorefresh import st_autorefresh

CONFIG_PATH = os.environ.get("PINGPONG_CONFIG", "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

API = CFG["dashboard"]["api_base_url"].rstrip("/")
REFRESH_MS = int(CFG["dashboard"]["refresh_ms"])
HISTORY_POINTS = int(CFG["dashboard"]["history_points"])
MIN_USABLE = int(CFG["model"]["min_rounds"])
WARMUP = max(int(CFG["features"]["max_lag"]), max(CFG["features"]["rolling_windows"]))
TARGET_ROUNDS = MIN_USABLE + WARMUP

st.set_page_config(page_title="PULSE | Ping Pong Intelligence", page_icon="◉", layout="wide", initial_sidebar_state="expanded")
st_autorefresh(interval=REFRESH_MS, key="live_refresh")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
:root { --ink:#e8edf5; --muted:#8995a8; --panel:#111827; --line:#263244; --cyan:#25d0b1; --amber:#ffb547; --red:#ff647c; }
.stApp { background: radial-gradient(circle at 75% -20%, #173044 0, #090e17 42%, #070b12 100%); color:var(--ink); }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.block-container { padding-top:1.4rem; padding-bottom:4rem; max-width:1480px; }
#MainMenu, footer { visibility:hidden; }
[data-testid="stSidebar"] { background:#0b111c; border-right:1px solid var(--line); }
.brand { font:700 1.25rem 'Space Mono'; letter-spacing:.12em; color:#fff; }
.eyebrow { font:700 .7rem 'Space Mono'; color:var(--cyan); letter-spacing:.15em; text-transform:uppercase; }
.hero { padding:.35rem 0 1rem; }
.hero h1 { font:600 clamp(2rem,4vw,3.7rem) 'DM Sans'; letter-spacing:-.055em; margin:.25rem 0; line-height:1; }
.hero p { color:var(--muted); max-width:780px; font-size:1rem; }
.live-dot { width:8px;height:8px;background:var(--cyan);border-radius:50%;display:inline-block;margin-right:8px;box-shadow:0 0 14px var(--cyan); }
.status-strip { border:1px solid var(--line);background:rgba(17,24,39,.76);padding:.8rem 1rem;border-radius:12px;color:var(--muted); }
.status-strip strong { color:var(--ink); }
[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(22,31,46,.94),rgba(13,20,32,.94)); border:1px solid var(--line); padding:1rem 1.1rem; border-radius:14px; min-height:112px; }
[data-testid="stMetricLabel"] { color:var(--muted);font-size:.76rem;text-transform:uppercase;letter-spacing:.08em; }
[data-testid="stMetricValue"] { font:700 1.75rem 'Space Mono'; color:#f4f7fb; }
.section-title { margin-top:1rem;font:600 1.45rem 'DM Sans';letter-spacing:-.025em; }
.section-copy { color:var(--muted); margin-top:-.45rem; }
.pred-card { border:1px solid var(--line); background:linear-gradient(140deg,rgba(20,31,46,.97),rgba(9,16,27,.96));border-radius:18px;padding:1.25rem; }
.readiness { display:flex;justify-content:space-between;color:var(--muted);font:500 .78rem 'Space Mono';margin-bottom:.45rem; }
.progress-track { height:9px;background:#1d2939;border-radius:99px;overflow:hidden; }
.progress-fill { height:100%;background:linear-gradient(90deg,#17a88d,#35e2bf);border-radius:99px;box-shadow:0 0 15px rgba(37,208,177,.35); }
.callout { border-left:3px solid var(--amber);background:rgba(255,181,71,.08);padding:.8rem 1rem;border-radius:0 10px 10px 0;color:#d7deea; }
.edge-ok { border-left-color:var(--cyan);background:rgba(37,208,177,.08); }
.edge-no { border-left-color:var(--amber); }
div[data-testid="stTabs"] button { font-weight:600; }
[data-testid="stDataFrame"] { border:1px solid var(--line);border-radius:12px;overflow:hidden; }
hr { border-color:var(--line); }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=max(1, REFRESH_MS / 1000 * .8), show_spinner=False)
def get_json(path: str, params=None):
    r = requests.get(f"{API}{path}", params=params, timeout=6)
    r.raise_for_status()
    return r.json()

def fmt_x(value):
    return f"{value:.2f}x" if isinstance(value, (int, float)) else "—"

def plot_layout(fig, height=370):
    fig.update_layout(height=height, margin=dict(l=18,r=18,t=35,b=15), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#aeb8c7",family="DM Sans"), hoverlabel=dict(bgcolor="#101827"), legend=dict(orientation="h",y=1.12))
    fig.update_xaxes(gridcolor="rgba(120,140,165,.10)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(120,140,165,.10)", zeroline=False)
    return fig

with st.sidebar:
    st.markdown('<div class="brand">◉ PULSE</div>', unsafe_allow_html=True)
    st.caption("RESEARCH TERMINAL · v0.2")
    st.divider()
    st.markdown("**System**")
    st.caption(f"API · {API}")
    st.caption(f"Refresh · {REFRESH_MS/1000:.0f}s")
    st.caption(f"Training target · {TARGET_ROUNDS:,} rounds")
    st.divider()
    st.markdown("**Research boundary**")
    st.caption("Read-only analysis. No staking, betting, cash-out, or guaranteed outcomes. Predictions count as evidence only when future-only gates pass.")

try:
    health = get_json("/health")
except Exception as e:
    st.error(f"Research API unavailable at {API}: {e}")
    st.stop()

rounds = int(health.get("rounds", 0))
latest = health.get("latest_round") or {}
model_ready = bool(health.get("model_exists"))
progress = min(1.0, rounds / max(TARGET_ROUNDS, 1))
remaining = max(0, TARGET_ROUNDS - rounds)

st.markdown('<div class="hero"><div class="eyebrow"><span class="live-dot"></span>Live intelligence</div><h1>Next-round research,<br>without false certainty.</h1><p>A real-time view of completed rounds, model readiness, calibrated threshold probabilities, and the held-out evidence required to trust them.</p></div>', unsafe_allow_html=True)

latest_time = latest.get("observed_at", "—")
st.markdown(f'<div class="status-strip"><span class="live-dot"></span><strong>Collector online</strong> &nbsp;·&nbsp; Round {latest.get("round_id", "—")} &nbsp;·&nbsp; Last update {latest_time}</div>', unsafe_allow_html=True)

m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("Completed rounds", f"{rounds:,}")
m2.metric("Latest result", fmt_x(latest.get("multiplier")))
m3.metric("Model state", "READY" if model_ready else "COLLECTING")
m4.metric("Training progress", f"{progress*100:.1f}%")
m5.metric("Rounds remaining", f"{remaining:,}" if not model_ready else "0")

prediction = None
if model_ready:
    try: prediction = get_json("/prediction")
    except Exception as e: st.warning(f"Prediction unavailable: {e}")

nav = st.tabs(["Overview", "Next Prediction", "Live Rounds", "Model Evidence", "Diagnostics"])

with nav[0]:
    st.markdown('<div class="section-title">Operational overview</div><div class="section-copy">Collection health, training readiness, and current distribution at a glance.</div>', unsafe_allow_html=True)
    left,right = st.columns([1.05,1])
    with left:
        st.markdown('<div class="pred-card">', unsafe_allow_html=True)
        st.markdown("#### Model readiness")
        st.markdown(f'<div class="readiness"><span>{rounds:,} collected</span><span>{TARGET_ROUNDS:,} target</span></div><div class="progress-track"><div class="progress-fill" style="width:{progress*100:.2f}%"></div></div>', unsafe_allow_html=True)
        if model_ready:
            st.success("Production ensemble is trained and available.")
        else:
            st.info(f"The live trainer is waiting for approximately {remaining:,} more completed rounds. Training starts automatically.")
        st.caption(f"The first {WARMUP} rounds establish lag and rolling-window features; {MIN_USABLE:,} usable rows are then split chronologically.")
        st.markdown('</div>', unsafe_allow_html=True)
    try: diag = get_json("/diagnostics")
    except Exception: diag = None
    with right:
        st.markdown("#### Distribution snapshot")
        if diag and diag.get("status") == "OK":
            rates = pd.DataFrame([{"Threshold":f"≥ {k}x","Observed rate":v*100} for k,v in diag.get("threshold_rates",{}).items()])
            fig=px.bar(rates,x="Threshold",y="Observed rate",color="Observed rate",color_continuous_scale=[[0,"#263244"],[1,"#25d0b1"]],text_auto=".1f")
            fig.update_coloraxes(showscale=False); st.plotly_chart(plot_layout(fig,320),use_container_width=True)
        else: st.info("Waiting for completed rounds.")

with nav[1]:
    st.markdown('<div class="section-title">Next-round probability workspace</div><div class="section-copy">The model predicts threshold probabilities, not an exact guaranteed crash point.</div>', unsafe_allow_html=True)
    if not prediction:
        st.markdown(f'<div class="pred-card"><div class="eyebrow">Prediction locked</div><h2>Collecting evidence before forecasting</h2><p style="color:#8995a8">{remaining:,} additional rounds are currently required. This panel will activate automatically after training and held-out evaluation complete.</p><div class="readiness"><span>{rounds:,} rounds</span><span>{progress*100:.1f}%</span></div><div class="progress-track"><div class="progress-fill" style="width:{progress*100:.2f}%"></div></div></div>',unsafe_allow_html=True)
    else:
        status=prediction.get("status")
        good=status=="MODEL_HAS_HELD_OUT_SIGNAL"
        cls="edge-ok" if good else "edge-no"
        msg="At least one threshold passed all held-out gates. This is evidence, not a guarantee." if good else "NO PREDICTIVE EDGE — probabilities are descriptive estimates and should not be treated as actionable forecasts."
        st.markdown(f'<div class="callout {cls}"><strong>{status.replace("_"," ")}</strong><br>{msg}</div>',unsafe_allow_html=True)
        summary=prediction.get("multiplier_summary",{})
        q1,q2,q3=st.columns(3)
        q1.metric("Estimated median",fmt_x(summary.get("estimated_p50_multiplier")))
        q2.metric("Estimated 75th percentile",fmt_x(summary.get("estimated_p75_multiplier")))
        q3.metric("Estimated 90th percentile",fmt_x(summary.get("estimated_p90_multiplier")))
        pdf=pd.DataFrame(prediction.get("predictions",[])).sort_values("threshold")
        pdf["Threshold"]=pdf["threshold"].map(lambda x:f"{x:g}x")
        pdf["Probability"]=pdf["probability"]*100
        fig=go.Figure(go.Scatter(x=pdf["threshold"],y=pdf["Probability"],mode="lines+markers",line=dict(color="#25d0b1",width=4),marker=dict(size=9,color="#08121b",line=dict(width=3,color="#25d0b1")),fill="tozeroy",fillcolor="rgba(37,208,177,.09)",hovertemplate="P(next ≥ %{x:g}x) = %{y:.1f}%<extra></extra>"))
        fig.update_yaxes(range=[0,100],title="Estimated probability (%)");fig.update_xaxes(title="Multiplier threshold",type="log")
        st.plotly_chart(plot_layout(fig,430),use_container_width=True)
        view=pdf[["Threshold","Probability","heldout_auc","relative_logloss_improvement","brier_improvement","edge_detected_for_threshold"]].copy()
        view["Probability"]=view["Probability"].map(lambda x:f"{x:.1f}%")
        view["relative_logloss_improvement"]=view["relative_logloss_improvement"].map(lambda x:f"{x*100:.3f}%")
        st.dataframe(view,use_container_width=True,hide_index=True)

with nav[2]:
    st.markdown('<div class="section-title">Live completed rounds</div><div class="section-copy">Real outcomes captured from the completed-round stream.</div>',unsafe_allow_html=True)
    try: hdf=pd.DataFrame(get_json("/history",{"limit":HISTORY_POINTS}).get("rows",[]))
    except Exception as e: st.warning(str(e));hdf=pd.DataFrame()
    if not hdf.empty:
        hdf["sequence"]=range(1,len(hdf)+1);hdf["observed_at"]=pd.to_datetime(hdf["observed_at"],errors="coerce")
        top,bottom=st.columns([1.7,1])
        with top:
            cap=float(hdf["multiplier"].quantile(.98)); chart=hdf.copy();chart["display_multiplier"]=chart["multiplier"].clip(upper=cap)
            fig=px.area(chart,x="observed_at",y="display_multiplier",markers=True,labels={"observed_at":"Time","display_multiplier":"Multiplier (98% display cap)"})
            fig.update_traces(line_color="#25d0b1",fillcolor="rgba(37,208,177,.10)");st.plotly_chart(plot_layout(fig,390),use_container_width=True)
        with bottom:
            fig=px.histogram(hdf,x="multiplier",nbins=35,color_discrete_sequence=["#ffb547"],labels={"multiplier":"Multiplier"});st.plotly_chart(plot_layout(fig,390),use_container_width=True)
        recent=hdf.sort_values("observed_at",ascending=False).head(20)[["round_id","observed_at","multiplier","source"]]
        st.dataframe(recent,use_container_width=True,hide_index=True,column_config={"round_id":"Round","observed_at":"Completed at","multiplier":st.column_config.NumberColumn("Result",format="%.2fx"),"source":"Source"})

with nav[3]:
    st.markdown('<div class="section-title">Held-out model evidence</div><div class="section-copy">Chronological future-only tests decide whether model output counts as signal.</div>',unsafe_allow_html=True)
    if not model_ready: st.info("Evidence tables appear after the first production training run.")
    else:
        try: met=get_json("/model/metrics")
        except Exception as e: st.warning(str(e));met=None
        if met:
            rows=[]
            for rep in met["reports"].values():
                rows.append({"Threshold":f"{rep['threshold']:g}x","Gate":"PASS" if rep["edge_detected"] else "NO EDGE","AUC":rep["ensemble_test"]["auc"],"Model log loss":rep["ensemble_test"]["logloss"],"Baseline log loss":rep["baseline_test"]["logloss"],"Log-loss gain %":100*rep["relative_logloss_improvement"],"Brier gain":rep["brier_improvement"],"Test rows":rep["n_test"]})
            edf=pd.DataFrame(rows);st.dataframe(edf,use_container_width=True,hide_index=True)
            fig=px.bar(edf,x="Threshold",y="Log-loss gain %",color="Gate",color_discrete_map={"PASS":"#25d0b1","NO EDGE":"#ff647c"});st.plotly_chart(plot_layout(fig,350),use_container_width=True)

with nav[4]:
    st.markdown('<div class="section-title">Randomness and drift laboratory</div><div class="section-copy">Dependence tests help detect structure; they do not prove next-round predictability.</div>',unsafe_allow_html=True)
    if not diag or diag.get("status")!="OK": st.info("Diagnostics need completed rounds.")
    else:
        d1,d2,d3,d4=st.columns(4)
        d1.metric("Median",fmt_x(diag.get("median_multiplier")));d2.metric("Observed max",fmt_x(diag.get("max_multiplier")))
        rp=diag.get("runs_test_ge_2x",{}).get("p_value");d3.metric("Runs-test p",f"{rp:.4f}" if isinstance(rp,(int,float)) else "—")
        drift=diag.get("drift",{});d4.metric("Distribution drift",("FLAG" if drift.get("drift_flag") else "No flag") if drift.get("available") else "Need more data")
        a,b=st.columns(2)
        acf=pd.DataFrame(diag.get("autocorrelation",[]));mi=pd.DataFrame(diag.get("mutual_information",[]))
        with a:
            if not acf.empty:
                fig=px.bar(acf,x="lag",y="corr",color="corr",color_continuous_scale=[[0,"#ff647c"],[.5,"#263244"],[1,"#25d0b1"]]);fig.update_coloraxes(showscale=False);st.plotly_chart(plot_layout(fig,370),use_container_width=True)
        with b:
            if not mi.empty:
                fig=px.line(mi,x="lag",y="mutual_information",markers=True,color_discrete_sequence=["#ffb547"]);st.plotly_chart(plot_layout(fig,370),use_container_width=True)
        with st.expander("Ljung–Box results and distribution quantiles"):
            st.dataframe(pd.DataFrame(diag.get("ljung_box",[])),use_container_width=True,hide_index=True)
            st.json(diag.get("quantiles",{}))

st.divider()
st.caption(f"PULSE Research Terminal · Live data · Updated {datetime.utcnow().strftime('%H:%M:%S')} UTC · Read-only statistical analysis")
