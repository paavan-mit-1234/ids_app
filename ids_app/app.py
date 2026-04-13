import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
import os

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="NIDS Adversarial Robustness",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data ─────────────────────────────────────────────────────
CLASS_NAMES = ["Analysis","Backdoor","DoS","Exploits","Fuzzers",
               "Generic","Normal","Reconnaissance","Shellcode","Worms"]
EPSILONS    = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]
FEATURE_NAMES = [
    'dur','proto','service','state','spkts','dpkts','sbytes','dbytes',
    'rate','sttl','dttl','sload','dload','sloss','dloss','sinpkt','dinpkt',
    'sjit','djit','swin','stcpb','dtcpb','dwin','tcprtt','synack','ackdat',
    'smean','dmean','trans_depth','response_body_len','ct_srv_src',
    'ct_state_ttl','ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm',
    'ct_dst_src_ltm','is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd',
    'ct_src_ltm','ct_srv_dst','is_sm_ips_ports'
]

# Macro F1 across epsilons
MACRO = {
    "Standard": {
        "FGSM": [0.4056,0.2754,0.1965,0.1559,0.1362,0.1041],
        "PGD":  [0.4056,0.2549,0.1560,0.0742,0.0724,0.0706],
    },
    "Robust": {
        "FGSM": [0.3373,0.3305,0.2844,0.2852,0.2351,0.1140],
        "PGD":  [0.3373,0.3104,0.2669,0.2401,0.1261,0.0396],
    },
}

# Per-class F1 at each epsilon — shape [6 epsilons][10 classes]
PERCLASS = {
    "Standard": {
        "FGSM": [
            [0.003,0.097,0.275,0.628,0.333,0.979,0.732,0.731,0.219,0.060],
            [0.011,0.085,0.212,0.523,0.238,0.843,0.730,0.061,0.030,0.021],
            [0.039,0.064,0.107,0.341,0.188,0.550,0.660,0.011,0.003,0.000],
            [0.037,0.021,0.067,0.128,0.163,0.507,0.624,0.009,0.002,0.000],
            [0.033,0.031,0.043,0.057,0.152,0.430,0.603,0.011,0.001,0.000],
            [0.042,0.023,0.025,0.052,0.140,0.186,0.563,0.009,0.001,0.000],
        ],
        "PGD": [
            [0.003,0.097,0.275,0.628,0.333,0.979,0.732,0.731,0.219,0.060],
            [0.014,0.087,0.186,0.482,0.206,0.801,0.730,0.027,0.012,0.004],
            [0.003,0.043,0.034,0.182,0.141,0.527,0.614,0.011,0.005,0.000],
            [0.003,0.011,0.007,0.068,0.103,0.010,0.529,0.008,0.003,0.000],
            [0.017,0.003,0.005,0.056,0.092,0.006,0.538,0.006,0.002,0.000],
            [0.002,0.008,0.001,0.063,0.088,0.005,0.533,0.005,0.000,0.000],
        ],
    },
    "Robust": {
        "FGSM": [
            [0.028,0.097,0.204,0.565,0.354,0.933,0.732,0.347,0.088,0.025],
            [0.034,0.084,0.166,0.577,0.345,0.896,0.732,0.377,0.068,0.026],
            [0.054,0.041,0.171,0.531,0.276,0.785,0.733,0.205,0.022,0.026],
            [0.032,0.032,0.230,0.451,0.317,0.751,0.712,0.294,0.007,0.026],
            [0.021,0.030,0.093,0.390,0.248,0.522,0.718,0.306,0.002,0.020],
            [0.017,0.020,0.052,0.323,0.088,0.017,0.596,0.020,0.000,0.007],
        ],
        "PGD": [
            [0.028,0.097,0.204,0.565,0.354,0.933,0.732,0.347,0.088,0.025],
            [0.032,0.085,0.165,0.577,0.344,0.894,0.732,0.184,0.065,0.026],
            [0.047,0.046,0.093,0.517,0.267,0.783,0.733,0.145,0.016,0.023],
            [0.030,0.029,0.095,0.361,0.226,0.721,0.685,0.223,0.007,0.023],
            [0.026,0.014,0.038,0.253,0.108,0.220,0.567,0.025,0.004,0.004],
            [0.007,0.001,0.011,0.066,0.065,0.109,0.123,0.011,0.003,0.000],
        ],
    },
}

# SHAP global importance
SHAP_STD = dict(zip(FEATURE_NAMES,[
    0.2319,0.1092,0.1204,0.1226,0.1760,0.3303,0.3733,0.2267,0.5977,0.7076,
    0.0891,0.0431,0.0334,0.0182,0.0158,0.0604,0.0481,0.0303,0.0236,0.0547,
    0.0447,0.0399,0.0520,0.0315,0.0281,0.0306,0.0876,0.0720,0.0143,0.0089,
    0.0520,0.0621,0.0742,0.0811,0.1027,0.0753,0.0062,0.0059,0.0062,0.0618,
    0.0816,0.0060
]))
SHAP_ROB = dict(zip(FEATURE_NAMES,[
    0.0871,0.0300,0.0415,0.0489,0.0712,0.0938,0.1450,0.0566,0.0834,0.1023,
    0.0367,0.0179,0.0145,0.0081,0.0073,0.0267,0.0213,0.0132,0.0103,0.0237,
    0.0193,0.0172,0.0224,0.0136,0.0121,0.0132,0.0379,0.0311,0.0062,0.0039,
    0.0224,0.0268,0.0320,0.0350,0.0443,0.0325,0.0027,0.0026,0.0027,0.0267,
    0.0353,0.0026
]))

ASSETS = os.path.join(os.path.dirname(__file__), "assets")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ NIDS Robustness")
    st.markdown("**UNSW-NB15 · DNN · ART**")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Overview", "Attack Analysis", "Robustness Comparison",
         "SHAP Explainability", "Representation Space"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Paavan Sejpal · KLE Tech · 2025")

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Adversarial Robustness of DNN-Based NIDS")
    st.markdown("**Dataset:** UNSW-NB15 &nbsp;|&nbsp; **Model:** IDS-DNN (256→128→64→32→10) &nbsp;|&nbsp; **Attacks:** FGSM, PGD via IBM ART")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Standard — Clean Macro F1", "0.406")
    c2.metric("Standard — PGD @ ε=0.10", "0.074", delta="-81.7%", delta_color="inverse")
    c3.metric("Robust — Clean Macro F1", "0.337", delta="-16.9% vs std", delta_color="inverse")
    c4.metric("Robust — PGD @ ε=0.10", "0.240", delta="+223.8% vs std")

    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Pipeline")
        st.markdown("""
| Phase | What |
|-------|------|
| **1 — Baseline** | DNN trained with SMOTE + class-weighted loss |
| **2 — Attack** | FGSM & PGD sweep over ε ∈ {0, 0.01, 0.05, 0.10, 0.20, 0.30} |
| **3 — Defence** | ε-randomised adversarial training (PGD, λ=0.5) |
| **4 — SHAP** | DeepExplainer feature attribution + t-SNE |
""")

    with col_r:
        st.subheader("Dataset Class Distribution")
        dist = {
            "Normal":677+37000-677, "Generic":18871, "Exploits":11132,
            "Fuzzers":6062, "DoS":4089, "Reconnaissance":3496,
            "Analysis":677, "Backdoor":583, "Shellcode":378, "Worms":44,
        }
        # override with actual
        dist = dict(zip(CLASS_NAMES,[677,583,4089,11132,6062,18871,37000,3496,378,44]))
        fig = px.bar(
            x=list(dist.values()), y=list(dist.keys()),
            orientation='h', color=list(dist.values()),
            color_continuous_scale='Blues',
            labels={'x':'Test samples','y':''},
        )
        fig.update_layout(height=320, margin=dict(l=0,r=0,t=0,b=0),
                          coloraxis_showscale=False,
                          plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Per-Class F1 — Clean Data")
    df_clean = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Standard": PERCLASS["Standard"]["FGSM"][0],
        "Robust":   PERCLASS["Robust"]["FGSM"][0],
    })
    fig = go.Figure()
    fig.add_bar(name="Standard", x=df_clean["Class"], y=df_clean["Standard"],
                marker_color="#4C78A8")
    fig.add_bar(name="Robust",   x=df_clean["Class"], y=df_clean["Robust"],
                marker_color="#E45756")
    fig.update_layout(
        barmode='group', height=340,
        yaxis_title="F1 Score", yaxis_range=[0,1.05],
        legend=dict(orientation='h', y=1.1),
        margin=dict(l=0,r=0,t=10,b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.15)')
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 2 — ATTACK ANALYSIS
# ═══════════════════════════════════════════════════════════════
elif page == "Attack Analysis":
    st.title("Phase 2 — Adversarial Attack Analysis")
    st.markdown("Standard model evaluated under FGSM and PGD at six perturbation budgets.")
    st.divider()

    st.subheader("Macro F1 Degradation")
    img = Image.open(os.path.join(ASSETS, "phase2_macro_degradation.png"))
    st.image(img, use_container_width=True)

    st.divider()
    st.subheader("Interactive Macro F1 Curve")

    fig = go.Figure()
    fig.add_scatter(x=EPSILONS, y=MACRO["Standard"]["FGSM"], mode='lines+markers',
                    name='FGSM', line=dict(color='#4C78A8', width=2.5),
                    marker=dict(size=8))
    fig.add_scatter(x=EPSILONS, y=MACRO["Standard"]["PGD"], mode='lines+markers',
                    name='PGD', line=dict(color='#E45756', dash='dash', width=2.5),
                    marker=dict(symbol='square', size=8))
    fig.add_hline(y=0.4056, line_dash='dot', line_color='gray',
                  annotation_text='Clean baseline (0.406)',
                  annotation_position='top right')
    fig.update_layout(
        xaxis_title='Perturbation budget ε', yaxis_title='Macro F1',
        yaxis_range=[0,0.5], height=380,
        legend=dict(orientation='h', y=1.1),
        margin=dict(l=0,r=0,t=10,b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.15)')
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.15)')
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Per-Class F1 Degradation")
    img2 = Image.open(os.path.join(ASSETS, "phase2_degradation_curves.png"))
    st.image(img2, use_container_width=True)

    st.divider()
    st.subheader("Interactive Per-Class Explorer")
    col1, col2 = st.columns(2)
    attack = col1.selectbox("Attack", ["FGSM", "PGD"], key="atk_p2")
    classes = col2.multiselect("Classes", CLASS_NAMES,
                                default=["Generic","Normal","Reconnaissance","DoS"],
                                key="cls_p2")

    if classes:
        fig = go.Figure()
        palette = px.colors.qualitative.Plotly
        for i, cls in enumerate(classes):
            idx = CLASS_NAMES.index(cls)
            vals = [PERCLASS["Standard"][attack][e][idx] for e in range(len(EPSILONS))]
            fig.add_scatter(x=EPSILONS, y=vals, mode='lines+markers',
                            name=cls, line=dict(color=palette[i%len(palette)], width=2),
                            marker=dict(size=7))
        fig.update_layout(
            xaxis_title='ε', yaxis_title='F1', yaxis_range=[-0.05,1.05],
            height=380, legend=dict(orientation='h', y=1.12),
            margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        fig.update_xaxes(gridcolor='rgba(128,128,128,0.15)')
        fig.update_yaxes(gridcolor='rgba(128,128,128,0.15)')
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 3 — ROBUSTNESS COMPARISON
# ═══════════════════════════════════════════════════════════════
elif page == "Robustness Comparison":
    st.title("Phase 3 — Standard vs Robust Model")
    st.markdown("Adversarial training with ε-randomised PGD (ε ∈ [0.01, 0.15]) and balanced loss (λ=0.5).")
    st.divider()

    attack = st.radio("Attack type", ["FGSM", "PGD"], horizontal=True, key="atk_p3")

    fig = go.Figure()
    fig.add_scatter(x=EPSILONS, y=MACRO["Standard"][attack], mode='lines+markers',
                    name='Standard', line=dict(color='#4C78A8', width=2.5),
                    marker=dict(size=8))
    fig.add_scatter(x=EPSILONS, y=MACRO["Robust"][attack], mode='lines+markers',
                    name='Robust', line=dict(color='#E45756', dash='dash', width=2.5),
                    marker=dict(symbol='square', size=8))
    fig.update_layout(
        xaxis_title='Perturbation budget ε', yaxis_title='Macro F1',
        yaxis_range=[0, 0.5], height=380,
        legend=dict(orientation='h', y=1.1),
        margin=dict(l=0,r=0,t=10,b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.15)')
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.15)')
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Per-Class Comparison at Selected ε")
    eps_idx = st.select_slider(
        "Perturbation budget ε",
        options=EPSILONS,
        value=0.1,
        key="eps_slider",
        format_func=lambda x: str(x),
    )
    ei = EPSILONS.index(eps_idx)

    std_vals = PERCLASS["Standard"][attack][ei]
    rob_vals = PERCLASS["Robust"][attack][ei]
    delta    = [round(r-s, 3) for r,s in zip(rob_vals, std_vals)]

    df = pd.DataFrame({
        "Class":    CLASS_NAMES,
        "Standard": std_vals,
        "Robust":   rob_vals,
        "Δ (Rob−Std)": delta,
    })

    fig2 = go.Figure()
    fig2.add_bar(name="Standard", x=df["Class"], y=df["Standard"],
                 marker_color="#4C78A8")
    fig2.add_bar(name="Robust",   x=df["Class"], y=df["Robust"],
                 marker_color="#E45756")
    fig2.update_layout(
        barmode='group', height=360,
        yaxis_title='F1 Score', yaxis_range=[0,1.05],
        legend=dict(orientation='h', y=1.1),
        margin=dict(l=0,r=0,t=10,b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    fig2.update_xaxes(showgrid=False)
    fig2.update_yaxes(gridcolor='rgba(128,128,128,0.15)')
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        df.style
          .format({"Standard":"{:.3f}","Robust":"{:.3f}","Δ (Rob−Std)":"{:+.3f}"})
          .background_gradient(subset=["Δ (Rob−Std)"], cmap="RdYlGn", vmin=-0.4, vmax=0.4),
        use_container_width=True, hide_index=True,
    )

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    std_m  = MACRO["Standard"][attack][ei]
    rob_m  = MACRO["Robust"][attack][ei]
    pct    = ((rob_m - std_m) / max(std_m, 1e-9)) * 100
    m1.metric("Standard Macro F1", f"{std_m:.4f}")
    m2.metric("Robust Macro F1",   f"{rob_m:.4f}")
    m3.metric("Absolute Gain",     f"{rob_m-std_m:+.4f}")
    m4.metric("Relative Change",   f"{pct:+.1f}%")

# ═══════════════════════════════════════════════════════════════
# PAGE 4 — SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════
elif page == "SHAP Explainability":
    st.title("Phase 4 — SHAP Feature Attribution")
    st.markdown("SHAP DeepExplainer on ~1,000 stratified test samples. Background: ~500 samples.")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Global Importance", "Beeswarm", "Rank Change"])

    with tab1:
        st.subheader("Global Feature Importance — Standard vs Robust")
        img = Image.open(os.path.join(ASSETS, "phase4_global_importance.png"))
        st.image(img, use_container_width=True)

        st.divider()
        st.subheader("Interactive Top-N Comparison")
        top_n = st.slider("Show top N features", 5, 42, 15)

        std_sorted = sorted(SHAP_STD.items(), key=lambda x: -x[1])[:top_n]
        rob_sorted = sorted(SHAP_ROB.items(), key=lambda x: -x[1])[:top_n]

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Standard model**")
            fig = go.Figure(go.Bar(
                x=[v for _,v in std_sorted],
                y=[f for f,_ in std_sorted],
                orientation='h', marker_color='#4C78A8',
            ))
            fig.update_layout(height=40*top_n+60, margin=dict(l=0,r=0,t=10,b=0),
                              xaxis_title='Mean |SHAP|',
                              plot_bgcolor='rgba(0,0,0,0)',
                              paper_bgcolor='rgba(0,0,0,0)')
            fig.update_xaxes(gridcolor='rgba(128,128,128,0.15)')
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown("**Robust model**")
            fig = go.Figure(go.Bar(
                x=[v for _,v in rob_sorted],
                y=[f for f,_ in rob_sorted],
                orientation='h', marker_color='#E45756',
            ))
            fig.update_layout(height=40*top_n+60, margin=dict(l=0,r=0,t=10,b=0),
                              xaxis_title='Mean |SHAP|',
                              plot_bgcolor='rgba(0,0,0,0)',
                              paper_bgcolor='rgba(0,0,0,0)')
            fig.update_xaxes(gridcolor='rgba(128,128,128,0.15)')
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("SHAP Beeswarm — Standard vs Robust")
        st.caption("Colour = normalised feature value (red = high, blue = low). "
                   "Note the compressed scale in the robust model — a known effect of adversarial training.")
        img = Image.open(os.path.join(ASSETS, "phase4_beeswarm.png"))
        st.image(img, use_container_width=True)

    with tab3:
        st.subheader("Feature Rank Change: Standard → Robust")
        img = Image.open(os.path.join(ASSETS, "phase4_rank_change.png"))
        st.image(img, use_container_width=True)

        st.divider()
        st.subheader("Interactive Rank Table")
        ranks_std = {f: r+1 for r,(f,_) in enumerate(
            sorted(SHAP_STD.items(), key=lambda x:-x[1]))}
        ranks_rob = {f: r+1 for r,(f,_) in enumerate(
            sorted(SHAP_ROB.items(), key=lambda x:-x[1]))}

        rank_df = pd.DataFrame({
            "Feature":       FEATURE_NAMES,
            "Std Rank":      [ranks_std[f] for f in FEATURE_NAMES],
            "Rob Rank":      [ranks_rob[f] for f in FEATURE_NAMES],
            "Std |SHAP|":    [SHAP_STD[f] for f in FEATURE_NAMES],
            "Rob |SHAP|":    [SHAP_ROB[f] for f in FEATURE_NAMES],
        })
        rank_df["Δ Rank"] = rank_df["Std Rank"] - rank_df["Rob Rank"]
        rank_df = rank_df.sort_values("Std Rank").reset_index(drop=True)

        st.dataframe(
            rank_df.style
              .format({"Std |SHAP|":"{:.4f}","Rob |SHAP|":"{:.4f}","Δ Rank":"{:+d}"})
              .background_gradient(subset=["Δ Rank"], cmap="RdYlGn", vmin=-5, vmax=5),
            use_container_width=True, hide_index=True,
        )

# ═══════════════════════════════════════════════════════════════
# PAGE 5 — REPRESENTATION SPACE
# ═══════════════════════════════════════════════════════════════
elif page == "Representation Space":
    st.title("Phase 4 — Penultimate Layer t-SNE")
    st.markdown(
        "32-dimensional activations from the layer before the classifier head, "
        "projected to 2D via t-SNE (perplexity=40, 1000 iter). "
        "Each point is one of ~1,000 stratified test samples coloured by true class."
    )
    st.divider()

    img = Image.open(os.path.join(ASSETS, "phase4_tsne.png"))
    st.image(img, use_container_width=True)

    st.divider()
    st.subheader("What this tells you")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Standard model**")
        st.markdown("""
- Elongated, partially overlapping manifolds
- Backdoor, DoS, Exploits bleed into each other
- No clear separation for minority classes
- Large spread → gradient attacks can easily push samples across boundaries
""")
    with col2:
        st.markdown("**Robust model**")
        st.markdown("""
- Tighter, more separated clusters
- Generic, Normal, Reconnaissance form compact blobs
- More whitespace between classes
- Compact clusters → harder to cross decision boundaries with bounded ε
""")
