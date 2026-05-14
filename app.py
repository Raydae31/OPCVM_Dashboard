"""
OPCVM Portfolio Dashboard  
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="OPCVM Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "dark_green":  "#1A3C2E",
    "mid_green":   "#2C5F2D",
    "light_green": "#97BC62",
    "mint":        "#D4EDDA",
    "navy":        "#1E3A5F",
    "red":         "#C0392B",
    "gold":        "#F0C040",
    "gray":        "#64748B",
    "purple":      "#6B3FA0",
}

st.markdown(f"""
<style>
    .stApp {{ background-color: #F5F9F6; }}
    [data-testid="stSidebar"] {{ background-color: {COLORS['dark_green']}; }}
    [data-testid="stSidebar"] * {{ color: white !important; }}
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {{ color: white !important; }}
    [data-testid="stSidebar"] .stSlider label {{ color: {COLORS['light_green']} !important; }}
    [data-testid="stSidebar"] .stSelectbox label {{ color: {COLORS['light_green']} !important; }}
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] {{
        background-color: {COLORS['mid_green']} !important;
        border: 1px solid {COLORS['light_green']} !important;
    }}
    [data-testid="stSidebar"] .stButton > button {{
        background-color: {COLORS['mid_green']} !important;
        color: white !important;
        border: 2px solid {COLORS['light_green']} !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        width: 100% !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background-color: {COLORS['light_green']} !important;
        color: {COLORS['dark_green']} !important;
    }}
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        background-color: {COLORS['light_green']} !important;
        color: {COLORS['dark_green']} !important;
        border: 2px solid white !important;
    }}
    [data-testid="stSidebar"] .streamlit-expanderHeader {{
        background-color: {COLORS['mid_green']} !important;
        border-radius: 6px !important;
        color: white !important;
    }}
    [data-testid="stSidebar"] .streamlit-expanderContent {{
        background-color: rgba(255,255,255,0.05) !important;
        border-left: 2px solid {COLORS['light_green']} !important;
    }}
    .main-header {{
        background: linear-gradient(135deg, {COLORS['dark_green']}, {COLORS['mid_green']});
        padding: 18px 24px; border-radius: 10px; margin-bottom: 20px;
        border-left: 6px solid {COLORS['light_green']};
    }}
    .main-header h1 {{ color: white; margin: 0; font-size: 1.6rem; }}
    .main-header p  {{ color: {COLORS['mint']}; margin: 4px 0 0; font-size: 0.9rem; }}
    .metric-card {{
        background: white; border-radius: 10px; padding: 14px 16px;
        border-left: 5px solid {COLORS['mid_green']};
        box-shadow: 0 2px 8px rgba(0,0,0,0.07); margin-bottom: 10px;
    }}
    .metric-card.red   {{ border-left-color: {COLORS['red']}; }}
    .metric-card.gold  {{ border-left-color: {COLORS['gold']}; }}
    .metric-card.navy  {{ border-left-color: {COLORS['navy']}; }}
    .metric-card h3 {{ margin: 0; font-size: 1.6rem; color: {COLORS['dark_green']}; font-weight: 700; }}
    .metric-card p  {{ margin: 2px 0 0; font-size: 0.8rem; color: {COLORS['gray']}; }}
    .metric-card .delta {{ font-size: 0.75rem; font-weight: 600; margin-top: 4px; }}
    .delta-pos {{ color: {COLORS['mid_green']}; }}
    .delta-neg {{ color: {COLORS['red']}; }}
    .section-title {{
        background: {COLORS['mid_green']}; color: white; padding: 8px 16px;
        border-radius: 6px; font-weight: 600; font-size: 0.95rem; margin: 16px 0 10px;
    }}
    .sum-warning {{
        background: #FFF3CD; border-left: 4px solid {COLORS['gold']};
        padding: 8px 12px; border-radius: 6px; font-size: 0.85rem; color: #856404;
    }}
    .sum-ok {{
        background: {COLORS['mint']}; border-left: 4px solid {COLORS['mid_green']};
        padding: 8px 12px; border-radius: 6px; font-size: 0.85rem;
        color: {COLORS['dark_green']};
    }}
    .stTabs [data-baseweb="tab"] {{ background: white; border-radius: 6px 6px 0 0; }}
    .stTabs [aria-selected="true"] {{ background: {COLORS['mid_green']}; color: white !important; }}
    .badge-correct   {{ background:#D4EDDA; color:#1A3C2E; padding:2px 8px; border-radius:10px; font-size:0.75rem; font-weight:600; }}
    .badge-modere    {{ background:#FFF3CD; color:#856404; padding:2px 8px; border-radius:10px; font-size:0.75rem; font-weight:600; }}
    .badge-sousperf  {{ background:#FFE4E4; color:#C0392B; padding:2px 8px; border-radius:10px; font-size:0.75rem; font-weight:600; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DONNÉES — alignées exactement avec la présentation PPT (slide 6)
# ══════════════════════════════════════════════════════════════════════════════

POIDS_ACTUELS = {
    "AFG GOV BOND FUND":       8.44,
    "AD BALANCED FUND":        2.95,
    "AFG OPTIMAL FUND":        4.51,
    "CDG IZDIHAR":             4.51,
    "AD SELECT BANK":          3.25,
    "ALPHA BANQUES FUND":      7.11,
    "ALPHA SECURE FUND":       8.30,
    "CAM OBLIBANQUES":         8.40,
    "CDG RENDEMENT":           8.41,
    "OBLIG CT":                9.80,
    "CDG TAWFIR":              8.68,
    "EMERGENCE SERENITE":     17.26,
    "CAPITAL TRUST EQUILIBRE": 4.35,
    "AD YIELD FUND":           4.04,
}

# ─────────────────────────────────────────────────────────────────────────────
# TOUTES LES VALEURS CI-DESSOUS SONT CELLES DE LA PRÉSENTATION PPT (slide 6)
# Rf = 2.25% | Alpha = Jensen OLS | Sharpe & Sortino ajustés
# ─────────────────────────────────────────────────────────────────────────────
META = {
    "AFG GOV BOND FUND": {
        # PPT slide 6 : Perf 5.74% | Alpha brut +1.20% | Sharpe 1.63 | Sortino 2.31
        # Vol 2.36% | DD -2.71% | VaR -0.384% | TE 1.50% | Score 4.9 | Modéré
        # Alpha Jensen (classement) : +0.69%  Beta OLS : 1.23
        "perf": 5.74, "vol": 2.36, "sharpe": 1.63, "sortino": 2.31,
        "alpha_brut": 1.20, "alpha_j": 0.69, "beta": 1.23,
        "te": 1.50, "ir": 0.82, "dd": -2.71, "var99": -0.384, "cvar99": -0.412,
        "score": 4.9, "statut": "Modéré",    "cat": "OMLT",
        "bench": "MBI GLOBAL",
    },
    "AD BALANCED FUND": {
        # PPT slide 6 : Perf 9.58% | Alpha brut -2.33% | Sharpe 1.93 | Sortino 2.88
        # Vol 32.49% | DD -8.87% | VaR -1.906% | TE 10.52% | Score 5.1 | Modéré
        # Alpha Jensen : -49.55%  Beta : 1.35
        "perf": 9.58, "vol": 32.49, "sharpe": 1.93, "sortino": 2.88,
        "alpha_brut": -2.33, "alpha_j": -49.55, "beta": 1.35,
        "te": 10.52, "ir": -0.89, "dd": -8.87, "var99": -1.906, "cvar99": -2.145,
        "score": 5.1, "statut": "Modéré",    "cat": "Diversifié",
        "bench": "MASI 50% + MBI MLT 50%",
    },
    "AFG OPTIMAL FUND": {
        # PPT slide 6 : Perf 12.22% | Alpha brut +1.57% | Sharpe 1.23 | Sortino 1.46
        # Vol 8.79% | DD -7.75% | VaR -1.573% | TE 3.52% | Score 5.2 | Modéré
        # Alpha Jensen : -3.29%  Beta : 1.54
        "perf": 12.22, "vol": 8.79, "sharpe": 1.23, "sortino": 1.46,
        "alpha_brut": 1.57, "alpha_j": -3.29, "beta": 1.54,
        "te": 3.52, "ir": 0.49, "dd": -7.75, "var99": -1.573, "cvar99": -1.831,
        "score": 5.2, "statut": "Modéré",    "cat": "Diversifié",
        "bench": "MBI GLOBAL 70% + MASI 30%",
    },
    "CDG IZDIHAR": {
        # PPT slide 6 : Perf 8.98% | Alpha brut +4.39% | Sharpe 2.34 | Sortino 3.97
        # Vol 24.76% | DD -6.71% | VaR -1.293% | TE 22.95% | Score 6.1 | Correct
        # Alpha Jensen : -15.01%  Beta : 2.85
        "perf": 8.98, "vol": 24.76, "sharpe": 2.34, "sortino": 3.97,
        "alpha_brut": 4.39, "alpha_j": -15.01, "beta": 2.85,
        "te": 22.95, "ir": 1.13, "dd": -6.71, "var99": -1.293, "cvar99": -1.524,
        "score": 6.1, "statut": "Correct",    "cat": "Diversifié",
        "bench": "MBI GLOBAL 77%",
    },
    "AD SELECT BANK": {
        # PPT slide 6 : Perf 3.39% | Alpha brut +0.90% | Sharpe 2.44 | Sortino 3.23
        # Vol 0.55% | DD -0.19% | VaR -0.081% | TE 0.32% | Score 6.1 | Correct
        # Alpha Jensen : +0.90%  Beta : 1.19
        "perf": 3.39, "vol": 0.55, "sharpe": 2.44, "sortino": 3.23,
        "alpha_brut": 0.90, "alpha_j": 0.90, "beta": 1.19,
        "te": 0.32, "ir": 2.99, "dd": -0.19, "var99": -0.081, "cvar99": -0.094,
        "score": 6.1, "statut": "Correct",    "cat": "OCT",
        "bench": "MBI CT 100%",
    },
    "ALPHA BANQUES FUND": {
        # PPT slide 6 : Perf 3.12% | Alpha brut +0.66% | Sharpe 2.16 | Sortino 2.90
        # Vol 0.50% | DD -0.18% | VaR -0.078% | TE 0.28% | Score 5.6 | Modéré
        # Alpha Jensen : +0.66%  Beta : 1.06
        "perf": 3.12, "vol": 0.50, "sharpe": 2.16, "sortino": 2.90,
        "alpha_brut": 0.66, "alpha_j": 0.66, "beta": 1.06,
        "te": 0.28, "ir": 2.41, "dd": -0.18, "var99": -0.078, "cvar99": -0.091,
        "score": 5.6, "statut": "Modéré",    "cat": "OCT",
        "bench": "MBI CT 100%",
    },
    "ALPHA SECURE FUND": {
        # PPT slide 6 : Perf 4.89% | Alpha brut +0.35% | Sharpe 1.73 | Sortino 2.18
        # Vol 1.71% | DD -1.59% | VaR -0.274% | TE 0.95% | Score 4.5 | Modéré
        # Alpha Jensen : +0.52%  Beta : 0.94
        "perf": 4.89, "vol": 1.71, "sharpe": 1.73, "sortino": 2.18,
        "alpha_brut": 0.35, "alpha_j": 0.52, "beta": 0.94,
        "te": 0.95, "ir": 0.38, "dd": -1.59, "var99": -0.274, "cvar99": -0.315,
        "score": 4.5, "statut": "Modéré",    "cat": "OMLT",
        "bench": "MBI GLOBAL 100%",
    },
    "CAM OBLIBANQUES": {
        # PPT slide 6 : Perf 3.17% | Alpha brut +0.60% | Sharpe 2.04 | Sortino 2.84
        # Vol 0.55% | DD -0.17% | VaR -0.074% | TE 0.25% | Score 5.5 | Modéré
        # Alpha Jensen : +0.60%  Beta : 1.15
        "perf": 3.17, "vol": 0.55, "sharpe": 2.04, "sortino": 2.84,
        "alpha_brut": 0.60, "alpha_j": 0.60, "beta": 1.15,
        "te": 0.25, "ir": 2.58, "dd": -0.17, "var99": -0.074, "cvar99": -0.088,
        "score": 5.5, "statut": "Modéré",    "cat": "OMLT",
        "bench": "85% MBI CT + 15% MBI MT",
    },
    "CDG RENDEMENT": {
        # PPT slide 6 : Perf 5.06% | Alpha brut -0.28% | Sharpe 4.58 | Sortino 6.49
        # Vol 5.05% | DD -1.69% | VaR -0.283% | TE 3.48% | Score 7.5 | Correct
        # Alpha Jensen : +6.76%  Beta : 0.66
        "perf": 5.06, "vol": 5.05, "sharpe": 4.58, "sortino": 6.49,
        "alpha_brut": -0.28, "alpha_j": 6.76, "beta": 0.66,
        "te": 3.48, "ir": -0.38, "dd": -1.69, "var99": -0.283, "cvar99": -0.321,
        "score": 7.5, "statut": "Correct",    "cat": "OMLT",
        "bench": "95% MBI MLT + 5% MASI",
    },
    "OBLIG CT": {
        # PPT slide 6 : Perf 3.23% | Alpha brut +0.66% | Sharpe 2.18 | Sortino 2.69
        # Vol 0.55% | DD -0.19% | VaR -0.096% | TE 0.22% | Score 5.8 | Modéré
        # Alpha Jensen : +0.66%  Beta : 1.04
        "perf": 3.23, "vol": 0.55, "sharpe": 2.18, "sortino": 2.69,
        "alpha_brut": 0.66, "alpha_j": 0.66, "beta": 1.04,
        "te": 0.22, "ir": 3.02, "dd": -0.19, "var99": -0.096, "cvar99": -0.112,
        "score": 5.8, "statut": "Modéré",    "cat": "OMLT",
        "bench": "75% MBI CT + 25% MBI MT",
    },
    "CDG TAWFIR": {
        # PPT slide 6 : Perf 5.01% | Alpha brut +0.47% | Sharpe 1.62 | Sortino 2.08
        # Vol 1.91% | DD -1.69% | VaR -0.344% | TE 1.16% | Score 4.5 | Sous-perf.
        # Alpha Jensen : +0.52%  Beta : 0.99
        "perf": 5.01, "vol": 1.91, "sharpe": 1.62, "sortino": 2.08,
        "alpha_brut": 0.47, "alpha_j": 0.52, "beta": 0.99,
        "te": 1.16, "ir": 0.42, "dd": -1.69, "var99": -0.344, "cvar99": -0.388,
        "score": 4.5, "statut": "Sous-perf.", "cat": "OMLT",
        "bench": "MBI GLOBAL 100%",
    },
    "EMERGENCE SERENITE": {
        # PPT slide 6 : Perf 4.63% | Alpha brut +0.06% | Sharpe 4.31 | Sortino 5.79
        # Vol 4.83% | DD -1.66% | VaR -0.616% | TE 2.34% | Score 7.1 | Correct
        # Alpha Jensen : +3.17%  Beta : 1.10
        "perf": 4.63, "vol": 4.83, "sharpe": 4.31, "sortino": 5.79,
        "alpha_brut": 0.06, "alpha_j": 3.17, "beta": 1.10,
        "te": 2.34, "ir": 0.12, "dd": -1.66, "var99": -0.616, "cvar99": -0.684,
        "score": 7.1, "statut": "Correct",    "cat": "OMLT",
        "bench": "MBI GLOBAL 100%",
    },
    "CAPITAL TRUST EQUILIBRE": {
        # PPT slide 6 : Perf 11.49% | Alpha brut +0.84% | Sharpe 1.22 | Sortino 1.45
        # Vol 8.22% | DD -7.63% | VaR -1.725% | TE 4.27% | Score 5.0 | Modéré
        # Alpha Jensen : -1.79%  Beta : 1.30
        "perf": 11.49, "vol": 8.22, "sharpe": 1.22, "sortino": 1.45,
        "alpha_brut": 0.84, "alpha_j": -1.79, "beta": 1.30,
        "te": 4.27, "ir": 0.23, "dd": -7.63, "var99": -1.725, "cvar99": -1.965,
        "score": 5.0, "statut": "Modéré",    "cat": "Diversifié",
        "bench": "70% MBI GLOBAL + 30% MASI",
    },
    "AD YIELD FUND": {
        # PPT slide 6 : Perf 3.11% | Alpha brut +0.59% | Sharpe 1.81 | Sortino 2.45
        # Vol 0.59% | DD -0.21% | VaR -0.089% | TE 0.36% | Score 5.0 | Modéré
        # Alpha Jensen : +0.59%  Beta : 1.21
        "perf": 3.11, "vol": 0.59, "sharpe": 1.81, "sortino": 2.45,
        "alpha_brut": 0.59, "alpha_j": 0.59, "beta": 1.21,
        "te": 0.36, "ir": 1.83, "dd": -0.21, "var99": -0.089, "cvar99": -0.103,
        "score": 5.0, "statut": "Modéré",    "cat": "OCT",
        "bench": "MBI CT 100%",
    },
}

NOMS     = list(POIDS_ACTUELS.keys())
N        = len(NOMS)
RF       = 0.0225          # Rf = 2.25%/an (slide 4)
RF_DAILY = RF / 252        # 0.00893%/j
METHODES = ["Min Variance", "Max Sharpe", "Min CVaR"]

STATUT_BADGE = {
    "Correct":    "badge-correct",
    "Modéré":     "badge-modere",
    "Sous-perf.": "badge-sousperf",
}


# ══════════════════════════════════════════════════════════════════════════════
# RENDEMENTS SYNTHÉTIQUES
# Cohérents avec perf, vol et beta réels de la PPT
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def generer_rendements():
    np.random.seed(42)
    T    = 247
    mu_j = np.array([META[n]["perf"] / 100 / 252     for n in NOMS])
    sg_j = np.array([META[n]["vol"]  / 100 / np.sqrt(252) for n in NOMS])
    bet  = np.array([META[n]["beta"] for n in NOMS])
    r_mkt = np.random.normal(0.0003, 0.008, T)
    R = np.zeros((T, N))
    for i in range(N):
        idio = max(sg_j[i] * np.sqrt(max(1 - bet[i]**2 * 0.35, 0.05)), 1e-6)
        R[:, i] = mu_j[i] + bet[i] * (r_mkt - 0.0003) + np.random.normal(0, idio, T)
        # Recalage exact perf annuelle et vol
        R[:, i] = (R[:, i] - R[:, i].mean() + mu_j[i])
        cur_sig  = R[:, i].std() + 1e-12
        R[:, i]  = (R[:, i] - mu_j[i]) / cur_sig * sg_j[i] + mu_j[i]
    return R

R_GLOBAL = generer_rendements()


# ══════════════════════════════════════════════════════════════════════════════
# CALCULS STATS PORTEFEUILLE
# ══════════════════════════════════════════════════════════════════════════════
def calcul_stats(w_arr, R=R_GLOBAL):
    r_ptf   = R @ w_arr
    perf    = float(r_ptf.mean() * 252 * 100)
    vol     = float(r_ptf.std()  * np.sqrt(252) * 100)
    sharpe  = float((perf/100 - RF) / (vol/100)) if vol > 1e-8 else 0.0
    r_neg   = r_ptf[r_ptf < RF_DAILY]
    dv      = float(r_neg.std() * np.sqrt(252)) if len(r_neg) > 1 else vol/100
    sortino = float((perf/100 - RF) / dv)        if dv  > 1e-8 else 0.0
    # VaR 99% — méthode historique lower (alignée PPT)
    var99   = float(np.percentile(r_ptf, 1, method="lower") * 100)
    mask    = r_ptf <= np.percentile(r_ptf, 1, method="lower")
    cvar99  = float(r_ptf[mask].mean() * 100) if mask.any() else var99
    cum     = np.cumprod(1 + r_ptf)
    roll    = np.maximum.accumulate(cum)
    dd_max  = float(((cum - roll) / roll).min() * 100)
    return {"perf": perf, "vol": vol, "sharpe": sharpe, "sortino": sortino,
            "var99": var99, "cvar99": cvar99, "dd_max": dd_max}


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMISATION — bornes dynamiques par profil OPCVM
# ══════════════════════════════════════════════════════════════════════════════
def calculer_bornes(w_actuel):
    sharpe_ind = np.array([META[n]["sharpe"] for n in NOMS])
    vol_ind    = np.array([META[n]["vol"]    for n in NOMS])
    sh_pos     = np.clip(sharpe_ind, 0.1, None)
    p_sharpe   = sh_pos / sh_pos.sum()
    vol_inv    = 1.0 / (vol_ind + 0.1)
    p_vol      = vol_inv / vol_inv.sum()
    score      = 0.60 * p_sharpe + 0.40 * p_vol
    max_b      = np.clip(score * 3.5, 0.05, 0.35)
    min_b      = np.clip(w_actuel * 0.30, 0.005, 0.10)
    for i in range(N):
        if min_b[i] >= max_b[i]:
            min_b[i] = max(0.005, max_b[i] * 0.5)
    return list(zip(min_b, max_b))


def optimiser(methode, w_actuel, R=R_GLOBAL):
    mu     = R.mean(axis=0) * 252
    cov    = np.cov(R.T) * 252
    bounds = calculer_bornes(w_actuel)
    lo     = np.array([b[0] for b in bounds])
    hi     = np.array([b[1] for b in bounds])
    csts   = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]
    if methode != "Min Variance":
        ret_min = float(mu @ w_actuel) * 0.80
        csts.append({"type": "ineq", "fun": lambda w: float(mu @ w) - ret_min})

    def cvar_fn(w):
        r = R @ w
        v = np.percentile(r, 1, method="lower")
        t = r[r <= v]
        return float(-t.mean()) if len(t) > 0 else float(-v)

    objs = {
        "Min Variance": lambda w: float(w @ cov @ w),
        "Max Sharpe":   lambda w: float(-(mu @ w - RF) / max(np.sqrt(w @ cov @ w), 1e-9)),
        "Min CVaR":     cvar_fn,
    }

    rng    = np.random.default_rng(42)
    best   = None
    sh_pos = np.clip(np.array([META[n]["sharpe"] for n in NOMS]), 0.1, None)
    w_sh   = sh_pos / sh_pos.sum()
    starts = [np.clip(w_actuel, lo, hi)]
    for a in [1.0, 0.7, 0.4]:
        w0 = np.clip(a * w_sh + (1-a) * rng.dirichlet(np.ones(N)), lo, hi)
        starts.append(w0 / max(w0.sum(), 1e-8))
    for _ in range(8):
        w0 = np.clip(rng.dirichlet(np.ones(N)), lo, hi)
        starts.append(w0 / max(w0.sum(), 1e-8))

    for w0 in starts:
        try:
            res = minimize(objs[methode], w0, method="SLSQP",
                           bounds=bounds, constraints=csts,
                           options={"maxiter": 2000, "ftol": 1e-12})
            if res.success and (best is None or res.fun < best.fun):
                best = res
        except Exception:
            pass

    if best is not None and best.success:
        w = np.clip(best.x, lo, hi)
        return w / w.sum()
    return w_actuel


@st.cache_data
def calculer_frontiere(R=R_GLOBAL):
    mu   = R.mean(axis=0) * 252
    cov  = np.cov(R.T) * 252
    n    = R.shape[1]
    tgts = np.linspace(mu.min(), mu.max() * 0.85, 200)
    bds  = [(0.0, 0.35)] * n
    obj  = lambda w: float(w @ cov @ w)
    cst0 = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]
    vols, rets, sharpes, var99s, cvar99s = [], [], [], [], []
    rng = np.random.default_rng(0)
    for t in tgts:
        cst = cst0 + [{"type": "eq", "fun": lambda w, tt=t: float(mu @ w - tt)}]
        best = None
        for _ in range(8):
            w0 = rng.dirichlet(np.ones(n))
            try:
                res = minimize(obj, w0, method="SLSQP", bounds=bds,
                               constraints=cst, options={"maxiter": 800, "ftol": 1e-10})
                if res.success and (best is None or res.fun < best.fun):
                    best = res
            except Exception:
                pass
        if best and best.success:
            w     = best.x
            r_ptf = R @ w
            vol_p = float(np.sqrt(w @ cov @ w) * 100)
            ret_p = float(mu @ w * 100)
            sh_p  = float((ret_p/100 - RF) / (vol_p/100)) if vol_p > 1e-8 else 0.0
            v99   = float(np.percentile(r_ptf, 1, method="lower") * 100)
            mask  = r_ptf <= np.percentile(r_ptf, 1, method="lower")
            cv99  = float(r_ptf[mask].mean() * 100) if mask.any() else v99
            vols.append(vol_p); rets.append(ret_p); sharpes.append(sh_p)
            var99s.append(v99); cvar99s.append(cv99)
    return (np.array(vols), np.array(rets),
            np.array(sharpes), np.array(var99s), np.array(cvar99s))


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — Fix bug Streamlit (pas de __setitem__ sur widget actif)
# ══════════════════════════════════════════════════════════════════════════════
for nom in NOMS:
    if f"poids_cible_{nom}" not in st.session_state:
        st.session_state[f"poids_cible_{nom}"] = float(POIDS_ACTUELS[nom])

# Injecter les cibles dans les clés slider AVANT création des widgets
for nom in NOMS:
    st.session_state[f"slider_{nom}"] = st.session_state[f"poids_cible_{nom}"]


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown(f"""
<div style="background:{COLORS['mid_green']};padding:12px;border-radius:8px;
     margin-bottom:12px;border-left:4px solid {COLORS['light_green']}">
  <h2 style="color:white;margin:0;font-size:1.05rem">⚖️ Pondérations OPCVM</h2>
  <p style="color:{COLORS['mint']};font-size:0.78rem;margin:4px 0 0">
     Rf = 2.25% · 247 obs · 14 OPCVM · 1.267 Md MAD</p>
</div>
""", unsafe_allow_html=True)

CATEGORIES = {
    "🏦 Oblig. Long Terme (OMLT)": [
        "AFG GOV BOND FUND","ALPHA SECURE FUND","CAM OBLIBANQUES",
        "CDG RENDEMENT","OBLIG CT","CDG TAWFIR","EMERGENCE SERENITE"],
    "📈 Diversifié": [
        "AD BALANCED FUND","AFG OPTIMAL FUND","CDG IZDIHAR","CAPITAL TRUST EQUILIBRE"],
    "💰 Oblig. Court Terme (OCT)": [
        "AD SELECT BANK","ALPHA BANQUES FUND","AD YIELD FUND"],
}

poids_user = {}
for cat, fonds in CATEGORIES.items():
    with st.sidebar.expander(cat, expanded=True):
        for nom in fonds:
            poids_user[nom] = st.slider(
                nom[:24],
                min_value=0.0, max_value=35.0, step=0.1,
                key=f"slider_{nom}", format="%.1f%%",
            )
            st.session_state[f"poids_cible_{nom}"] = poids_user[nom]

total_poids = sum(poids_user.values())
delta_sum   = total_poids - 100.0
if abs(delta_sum) < 0.05:
    st.sidebar.markdown(
        f'<div class="sum-ok">✅ Somme = {total_poids:.1f}%</div>',
        unsafe_allow_html=True)
else:
    sg = "+" if delta_sum > 0 else ""
    st.sidebar.markdown(
        f'<div class="sum-warning">⚠️ Somme = {total_poids:.1f}% '
        f'({sg}{delta_sum:.1f}pp) — normalisé auto.</div>',
        unsafe_allow_html=True)

st.sidebar.markdown("---")

if st.sidebar.button("🔄 Réinitialiser les poids", use_container_width=True):
    for nom in NOMS:
        st.session_state[f"poids_cible_{nom}"] = float(POIDS_ACTUELS[nom])
    st.rerun()

methode_optim = st.sidebar.selectbox(
    "🎯 Méthode d'optimisation", METHODES, key="methode_select")

if st.sidebar.button(f"⚡ Appliquer : {methode_optim}",
                     use_container_width=True, type="primary"):
    w_cur = np.array([poids_user[n] / 100 for n in NOMS])
    w_cur = np.clip(w_cur, 0.005, 0.35); w_cur /= w_cur.sum()
    w_opt = optimiser(methode_optim, w_cur)
    for i, nom in enumerate(NOMS):
        st.session_state[f"poids_cible_{nom}"] = round(float(w_opt[i]) * 100, 1)
    st.rerun()

st.sidebar.markdown(f"""
<div style="margin-top:12px;padding:8px;background:{COLORS['dark_green']};
     border-radius:6px;font-size:0.72rem;color:{COLORS['gray']};text-align:center;">
  Méthode VaR : historique lower (alignée PPT)<br>
  Sharpe & Sortino : Rf = 2.25%/an<br>
  © 2026 · Ben said Raydae
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CALCULS PRINCIPAUX
# ══════════════════════════════════════════════════════════════════════════════
w_raw     = np.array([poids_user[n] / 100 for n in NOMS])
w_norm    = w_raw / w_raw.sum()
w_ref     = np.array([POIDS_ACTUELS[n] / 100 for n in NOMS])
stats_cur = calcul_stats(w_norm)
stats_ref = calcul_stats(w_ref)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="main-header">
  <h1>📊 OPCVM Portfolio Dashboard</h1>
  <p>Optimisation interactive · Frontière Efficiente · Rf = 2.25% ·
     14 OPCVM · 247 observations · Valeurs PPT alignées</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Tableau de Bord",
    "🎯 Frontière Efficiente",
    "📋 Analyse Détaillée",
    "⚖️ Comparaison Méthodes",
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — TABLEAU DE BORD
# ═══════════════════════════════════════════════════════════════════
with tab1:

    st.markdown('<div class="section-title">📌 Indicateurs du Portefeuille Courant</div>',
                unsafe_allow_html=True)

    def delta_html(val, ref, unit="%", inverse=False):
        diff = val - ref
        good = (diff < 0) if inverse else (diff > 0)
        sg   = "+" if diff > 0 else ""
        cls  = "delta-pos" if good else "delta-neg"
        return f'<span class="{cls}">{sg}{diff:.2f}{unit} vs actuel</span>'

    kpis = [
        ("Performance Annuelle",   f"{stats_cur['perf']:.2f}%",   stats_cur["perf"],    stats_ref["perf"],    "%", False, ""),
        ("Volatilité Annualisée",  f"{stats_cur['vol']:.2f}%",    stats_cur["vol"],     stats_ref["vol"],     "%", True,  "red"),
        ("Sharpe Ratio",           f"{stats_cur['sharpe']:.3f}",  stats_cur["sharpe"],  stats_ref["sharpe"],  "",  False, "gold"),
        ("Sortino Ratio",          f"{stats_cur['sortino']:.3f}", stats_cur["sortino"], stats_ref["sortino"], "",  False, "gold"),
        ("VaR 99% (j)",            f"{stats_cur['var99']:.4f}%",  stats_cur["var99"],   stats_ref["var99"],   "%", True,  "red"),
        ("CVaR 99% (j)",           f"{stats_cur['cvar99']:.4f}%", stats_cur["cvar99"],  stats_ref["cvar99"],  "%", True,  "red"),
        ("Drawdown Max",           f"{stats_cur['dd_max']:.2f}%", stats_cur["dd_max"],  stats_ref["dd_max"],  "%", True,  "navy"),
    ]
    cols = st.columns(len(kpis))
    for col, (lbl, vs, v, r_, u, inv, cls) in zip(cols, kpis):
        with col:
            st.markdown(f"""<div class="metric-card {cls}">
              <p>{lbl}</p><h3>{vs}</h3>
              <div class="delta">{delta_html(v,r_,u,inv)}</div>
            </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1.1, 1.9])
    with c1:
        st.markdown('<div class="section-title">🥧 Répartition courante</div>',
                    unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=[n[:20] for n in NOMS], values=w_norm*100,
            textinfo="label+percent", textfont_size=9,
            marker=dict(colors=px.colors.qualitative.Set3,
                        line=dict(color="white", width=1.5)),
            hole=0.35,
        ))
        fig_pie.update_layout(
            showlegend=False, margin=dict(t=10,b=10,l=10,r=10),
            height=370, paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text=f"<b>{total_poids:.0f}%</b>",
                              x=0.5, y=0.5, font_size=14, showarrow=False,
                              font=dict(color=COLORS["dark_green"]))])
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">📊 Courant vs Actuel</div>',
                    unsafe_allow_html=True)
        noms_s  = [n[:14] for n in NOMS]
        delta_p = w_norm*100 - np.array([POIDS_ACTUELS[n] for n in NOMS])
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Actuel", x=noms_s,
            y=[POIDS_ACTUELS[n] for n in NOMS],
            marker_color=COLORS["gray"], opacity=0.65))
        fig_bar.add_trace(go.Bar(
            name="Courant", x=noms_s, y=w_norm*100,
            marker_color=[COLORS["mid_green"] if d >= 0 else COLORS["red"]
                          for d in delta_p], opacity=0.85))
        fig_bar.update_layout(
            barmode="group", height=370,
            margin=dict(t=5,b=80,l=10,r=10),
            legend=dict(orientation="h",y=1.04,x=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(244,247,251,0.5)",
            xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
            yaxis=dict(title="Poids (%)", gridcolor="#E2E8F0", range=[0,40]))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-title">📈 Performance cumulée simulée</div>',
                unsafe_allow_html=True)
    rc = R_GLOBAL @ w_norm; rr = R_GLOBAL @ w_ref
    cc = (1+rc).cumprod()-1; cr = (1+rr).cumprod()-1
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=list(range(len(cr))), y=cr*100, name=f"Actuel ({cr[-1]*100:.2f}%)",
        line=dict(color=COLORS["gray"], dash="dash", width=2), opacity=0.8))
    fig_cum.add_trace(go.Scatter(
        x=list(range(len(cc))), y=cc*100, name=f"Courant ({cc[-1]*100:.2f}%)",
        line=dict(color=COLORS["mid_green"], width=2.5),
        fill="tonexty", fillcolor="rgba(44,95,45,0.08)"))
    fig_cum.add_hline(y=0, line_color=COLORS["gray"], line_width=0.8)
    fig_cum.update_layout(
        height=260, margin=dict(t=5,b=30,l=40,r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(244,247,251,0.5)",
        legend=dict(orientation="h",y=1.1,x=0),
        yaxis=dict(title="Perf. cumulée (%)", gridcolor="#E2E8F0"),
        xaxis=dict(title="Jours", gridcolor="#E2E8F0"), hovermode="x unified")
    st.plotly_chart(fig_cum, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — FRONTIÈRE EFFICIENTE
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">🎯 Frontière Efficiente de Markowitz</div>',
                unsafe_allow_html=True)

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        colorby = st.selectbox("Colorier par :",
                               ["Ratio de Sharpe","VaR 99%","CVaR 99%"])
    with ctrl2:
        show_indiv = st.checkbox("OPCVM individuels", True)
    with ctrl3:
        show_cml = st.checkbox("CML (Capital Market Line)", True)

    with st.spinner("Calcul frontière efficiente (200 pts)…"):
        fe_v, fe_r, fe_sh, fe_va, fe_cv = calculer_frontiere()

    optim_pts = {}
    with st.spinner("Calcul portefeuilles optimisés…"):
        for m in METHODES:
            w_opt = optimiser(m, w_ref)
            optim_pts[m] = calcul_stats(w_opt)

    col_opt = {"Min Variance":COLORS["mid_green"],
               "Max Sharpe":COLORS["navy"],"Min CVaR":COLORS["red"]}
    sym_opt = {"Min Variance":"square","Max Sharpe":"triangle-up","Min CVaR":"diamond"}

    cmap_cfg = {
        "Ratio de Sharpe": (fe_sh, "Sharpe",    "RdYlGn",   False),
        "VaR 99%":         (fe_va, "VaR %/j",   "RdYlGn_r", False),
        "CVaR 99%":        (fe_cv, "CVaR %/j",  "RdYlGn_r", False),
    }
    cvals, cbar_t, cscale, _ = cmap_cfg[colorby]

    fig_fe = go.Figure()

    # Zone sous frontière
    if len(fe_v) > 0:
        fig_fe.add_trace(go.Scatter(
            x=np.concatenate([fe_v, fe_v[::-1]]),
            y=np.concatenate([fe_r, np.full(len(fe_r), fe_r.min()-0.5)]),
            fill="toself", fillcolor="rgba(151,188,98,0.06)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip"))

    # Frontière colorée
    hover_fe = [
        f"Vol:{v:.2f}% | Perf:{r:.2f}% | Sharpe:{s:.3f} | VaR:{va:.4f}%"
        for v,r,s,va in zip(fe_v, fe_r, fe_sh, fe_va)
    ]
    fig_fe.add_trace(go.Scatter(
        x=fe_v, y=fe_r, mode="markers",
        marker=dict(color=cvals, colorscale=cscale, size=7, opacity=0.88,
                    colorbar=dict(title=cbar_t, thickness=13, len=0.55,
                                  y=0.75, yanchor="top"),
                    line=dict(width=0)),
        name="Frontière Efficiente",
        customdata=hover_fe,
        hovertemplate="%{customdata}<extra>Frontière</extra>"))

    # Contour
    if len(fe_v) > 0:
        si = np.argsort(fe_v)
        fig_fe.add_trace(go.Scatter(
            x=fe_v[si], y=fe_r[si], mode="lines",
            line=dict(color="rgba(44,95,45,0.25)", width=2),
            showlegend=False, hoverinfo="skip"))

    # Portefeuille tangent
    if len(fe_sh) > 0:
        ti = int(np.argmax(fe_sh))
        tv, tr, ts = float(fe_v[ti]), float(fe_r[ti]), float(fe_sh[ti])
        fig_fe.add_trace(go.Scatter(
            x=[tv], y=[tr], mode="markers+text",
            marker=dict(size=22, color=COLORS["gold"], symbol="star",
                        line=dict(color=COLORS["dark_green"], width=2)),
            text=["Tangent"], textposition="top right",
            textfont=dict(size=10, color=COLORS["gold"], family="Arial Black"),
            name=f"Tangent (Sh={ts:.3f})",
            hovertemplate=f"<b>Tangent</b><br>Vol:{tv:.2f}% Perf:{tr:.2f}% Sharpe:{ts:.3f}<extra></extra>"))

        if show_cml:
            slope = (tr/100 - RF) / (tv/100 + 1e-9)
            vr    = np.array([0.0, tv, tv*1.5])
            yr    = (RF + slope * vr/100) * 100
            fig_fe.add_trace(go.Scatter(
                x=vr[:2], y=yr[:2], mode="lines",
                line=dict(color=COLORS["gold"], width=2.2),
                name=f"CML (Rf={RF*100:.2f}%)", hoverinfo="skip"))
            fig_fe.add_trace(go.Scatter(
                x=vr[1:], y=yr[1:], mode="lines",
                line=dict(color=COLORS["gold"], width=1.4, dash="dot"),
                showlegend=False, hoverinfo="skip"))
            fig_fe.add_trace(go.Scatter(
                x=[0], y=[RF*100], mode="markers+text",
                marker=dict(size=10, color=COLORS["gold"], symbol="circle",
                            line=dict(color="white",width=1.5)),
                text=[f"Rf={RF*100:.2f}%"], textposition="top right",
                textfont=dict(size=9, color=COLORS["gold"]),
                showlegend=False, hoverinfo="skip"))

    # OPCVM individuels
    if show_indiv:
        for nom in NOMS:
            m   = META[nom]
            col = COLORS["mid_green"] if m["alpha_j"] > 0 else COLORS["red"]
            fig_fe.add_trace(go.Scatter(
                x=[m["vol"]], y=[m["perf"]], mode="markers+text",
                marker=dict(size=9, color=col, opacity=0.72,
                            line=dict(color="white", width=0.8)),
                text=[nom[:12]], textposition="top center",
                textfont=dict(size=7.5, color=COLORS["dark_green"]),
                showlegend=False,
                hovertemplate=(
                    f"<b>{nom}</b><br>"
                    f"Vol:{m['vol']:.2f}% Perf:{m['perf']:.2f}%<br>"
                    f"Sharpe:{m['sharpe']:.3f} α:{m['alpha_j']:+.2f}%<br>"
                    f"Score:{m['score']}/10 — {m['statut']}"
                    "<extra></extra>")))

    # Portefeuille actuel
    fig_fe.add_trace(go.Scatter(
        x=[stats_ref["vol"]], y=[stats_ref["perf"]], mode="markers+text",
        marker=dict(size=18, color=COLORS["gray"], symbol="square",
                    line=dict(color="white", width=2)),
        text=["Actuel"], textposition="top center",
        textfont=dict(size=10, color=COLORS["gray"], family="Arial Black"),
        name="Actuel",
        hovertemplate=(
            f"<b>Portefeuille Actuel</b><br>"
            f"Vol:{stats_ref['vol']:.2f}% Perf:{stats_ref['perf']:.2f}%<br>"
            f"Sharpe:{stats_ref['sharpe']:.3f} VaR:{stats_ref['var99']:.4f}%"
            "<extra></extra>")))

    # Portefeuille courant
    fig_fe.add_trace(go.Scatter(
        x=[stats_cur["vol"]], y=[stats_cur["perf"]], mode="markers+text",
        marker=dict(size=22, color=COLORS["light_green"], symbol="star",
                    line=dict(color=COLORS["dark_green"], width=2.5)),
        text=["Votre portefeuille"], textposition="top left",
        textfont=dict(size=10, color=COLORS["mid_green"], family="Arial Black"),
        name="Courant",
        hovertemplate=(
            f"<b>Votre Portefeuille</b><br>"
            f"Vol:{stats_cur['vol']:.2f}% Perf:{stats_cur['perf']:.2f}%<br>"
            f"Sharpe:{stats_cur['sharpe']:.3f} VaR:{stats_cur['var99']:.4f}%"
            "<extra></extra>")))

    # 3 méthodes
    for m_name, s in optim_pts.items():
        fig_fe.add_trace(go.Scatter(
            x=[s["vol"]], y=[s["perf"]], mode="markers+text",
            marker=dict(size=15, color=col_opt[m_name], symbol=sym_opt[m_name],
                        line=dict(color="white",width=1.5)),
            text=[m_name[:10]], textposition="bottom center",
            textfont=dict(size=8, color=col_opt[m_name], family="Arial Black"),
            name=m_name,
            hovertemplate=(
                f"<b>{m_name}</b><br>"
                f"Vol:{s['vol']:.2f}% Perf:{s['perf']:.2f}%<br>"
                f"Sharpe:{s['sharpe']:.3f} VaR:{s['var99']:.4f}%"
                "<extra></extra>")))

    # Ligne vers la frontière depuis actuel
    if len(fe_v) > 0:
        dist = np.sqrt((fe_v-stats_ref["vol"])**2 + (fe_r-stats_ref["perf"])**2)
        ni   = int(np.argmin(dist))
        fig_fe.add_trace(go.Scatter(
            x=[stats_ref["vol"], fe_v[ni]], y=[stats_ref["perf"], fe_r[ni]],
            mode="lines", line=dict(color=COLORS["gray"],dash="dot",width=1),
            showlegend=False, hoverinfo="skip"))

    fig_fe.update_layout(
        height=640, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(244,247,251,0.7)",
        xaxis=dict(title="Volatilité Annualisée (%)", gridcolor="#E2E8F0",
                   zeroline=False, showspikes=True, spikecolor="#CBD5E1"),
        yaxis=dict(title="Performance Annualisée (%)", gridcolor="#E2E8F0",
                   zeroline=False, showspikes=True, spikecolor="#CBD5E1"),
        legend=dict(orientation="v", x=1.12, y=1,
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor=COLORS["light_green"], borderwidth=1,
                    font=dict(size=9)),
        margin=dict(t=20,b=40,l=55,r=220),
        hovermode="closest")
    st.plotly_chart(fig_fe, use_container_width=True)

    # Légende
    lc = st.columns(5)
    for col, (lbl, brd) in zip(lc, [
        ("⭐ Tangent — Max Sharpe<br>sur la frontière",         COLORS["gold"]),
        ("🟩 Min Variance<br>Risque minimal",                   COLORS["mid_green"]),
        ("🔺 Max Sharpe<br>Meilleur ratio ajusté",              COLORS["navy"]),
        ("💎 Min CVaR<br>Pertes extrêmes minimales",            COLORS["red"]),
        ("⬛ Actuel<br>Portefeuille référence",                  COLORS["gray"]),
    ]):
        with col:
            st.markdown(f"""<div style="text-align:center;padding:8px;background:white;
            border-radius:8px;border:2px solid {brd};font-size:0.78rem;">{lbl}</div>""",
            unsafe_allow_html=True)

    # Tableau comparatif
    st.markdown('<div class="section-title">📊 Résumé des 3 méthodes</div>',
                unsafe_allow_html=True)
    rows_m = []
    if len(fe_sh) > 0:
        rows_m.append({
            "Méthode": "⭐ Tangent (frontière)",
            "Perf %/an": f"{fe_r[ti]:.2f}%",
            "Vol %/an":  f"{fe_v[ti]:.2f}%",
            "Sharpe":    f"{fe_sh[ti]:.3f}",
            "VaR 99%/j": f"{fe_va[ti]:.4f}%",
            "CVaR/j":    f"{fe_cv[ti]:.4f}%",
        })
    for m_name, s in optim_pts.items():
        rows_m.append({
            "Méthode": m_name,
            "Perf %/an": f"{s['perf']:.2f}%",
            "Vol %/an":  f"{s['vol']:.2f}%",
            "Sharpe":    f"{s['sharpe']:.3f}",
            "VaR 99%/j": f"{s['var99']:.4f}%",
            "CVaR/j":    f"{s['cvar99']:.4f}%",
        })
    st.dataframe(pd.DataFrame(rows_m).set_index("Méthode"),
                 use_container_width=True)

    # Pondérations optimisées
    st.markdown('<div class="section-title">📐 Pondérations optimisées par méthode</div>',
                unsafe_allow_html=True)
    w_par = {m: optimiser(m, w_ref) for m in METHODES}
    rows_w = []
    for i, nom in enumerate(NOMS):
        m = META[nom]
        row = {
            "OPCVM":        nom[:22],
            "Cat.":         m["cat"],
            "Score /10":    f"{m['score']:.1f}",
            "Statut":       m["statut"],
            "Actuel %":     f"{w_ref[i]*100:.1f}%",
        }
        for mth in METHODES:
            delta = (w_par[mth][i] - w_ref[i]) * 100
            row[f"{mth} %"] = f"{w_par[mth][i]*100:.1f}%"
            row[f"Δ {mth}"] = f"{delta:+.1f}pp"
        rows_w.append(row)
    df_w = pd.DataFrame(rows_w).set_index("OPCVM")
    st.dataframe(df_w, use_container_width=True, height=530)

    st.markdown(f"""
    <div style="background:{COLORS['mint']};border-left:4px solid {COLORS['mid_green']};
         padding:10px 14px;border-radius:6px;font-size:0.82rem;
         color:{COLORS['dark_green']};margin-top:8px;">
      <b>ℹ️ Bornes dynamiques :</b> chaque OPCVM reçoit un plafond calculé
      selon 60% Sharpe + 40% faible volatilité (plafond absolu 35%).
      Présence minimale = 30% du poids actuel. Méthode VaR : percentile historique
      lower (cohérente avec la PPT).
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — ANALYSE DÉTAILLÉE
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">📋 Tableau complet — valeurs PPT exactes</div>',
                unsafe_allow_html=True)

    rows_d = []
    for i, nom in enumerate(NOMS):
        m  = META[nom]
        wp = float(w_norm[i] * 100)
        wa = POIDS_ACTUELS[nom]
        rows_d.append({
            "OPCVM":              nom,
            "Catégorie":          m["cat"],
            "Benchmark":          m["bench"],
            "Score /10":          m["score"],
            "Statut":             m["statut"],
            "Poids actuel %":     round(wa, 2),
            "Poids courant %":    round(wp, 2),
            "Δ (pp)":             round(wp - wa, 2),
            "Perf %/an":          m["perf"],
            "Vol %/an":           m["vol"],
            "β":                  m["beta"],
            "α brut %":           m["alpha_brut"],
            "α Jensen %":         m["alpha_j"],
            "Sharpe":             m["sharpe"],
            "Sortino":            m["sortino"],
            "VaR 99% %/j":        m["var99"],
            "CVaR 99% %/j":       m["cvar99"],
            "TE %":               m["te"],
            "IR":                 m["ir"],
            "DD Max %":           m["dd"],
        })

    df_d = pd.DataFrame(rows_d).set_index("OPCVM")
    st.dataframe(df_d, use_container_width=True, height=520)

    # Scatter alpha Jensen vs Sharpe
    st.markdown('<div class="section-title">🔵 Alpha Jensen vs Sharpe (taille = poids courant)</div>',
                unsafe_allow_html=True)
    fig_sc = go.Figure()
    for i, nom in enumerate(NOMS):
        m  = META[nom]
        wp = float(w_norm[i] * 100)
        col = COLORS["mid_green"] if m["alpha_j"] > 0 else COLORS["red"]
        fig_sc.add_trace(go.Scatter(
            x=[m["alpha_j"]], y=[m["sharpe"]],
            mode="markers+text",
            marker=dict(size=max(wp/1.8, 8), color=col, opacity=0.72,
                        line=dict(color="white",width=1)),
            text=[nom[:13]], textposition="top center",
            textfont=dict(size=8, color=COLORS["dark_green"]),
            showlegend=False,
            hovertemplate=(
                f"<b>{nom}</b><br>"
                f"α Jensen:{m['alpha_j']:+.2f}% Sharpe:{m['sharpe']:.3f}<br>"
                f"Poids:{wp:.1f}% Score:{m['score']}/10 — {m['statut']}"
                "<extra></extra>")))
    fig_sc.add_vline(x=0,  line_dash="dot", line_color=COLORS["red"],  line_width=1.5)
    fig_sc.add_hline(y=2,  line_dash="dot", line_color=COLORS["navy"], line_width=1.5)
    fig_sc.add_hline(y=4,  line_dash="dot", line_color=COLORS["mid_green"], line_width=1,
                     annotation_text="Sharpe=4 (excellent)")
    fig_sc.update_layout(
        height=450, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(244,247,251,0.7)",
        xaxis=dict(title="Alpha Jensen (%)", gridcolor="#E2E8F0"),
        yaxis=dict(title="Sharpe Ratio",     gridcolor="#E2E8F0"),
        margin=dict(t=20,b=40,l=50,r=10))
    st.plotly_chart(fig_sc, use_container_width=True)

    # VaR vs Perf
    st.markdown('<div class="section-title">📉 VaR 99% vs Performance individuelle</div>',
                unsafe_allow_html=True)
    fig_vp = go.Figure()
    for i, nom in enumerate(NOMS):
        m   = META[nom]
        wp  = float(w_norm[i] * 100)
        col = COLORS["mid_green"] if m["alpha_j"] > 0 else COLORS["red"]
        fig_vp.add_trace(go.Scatter(
            x=[m["var99"]], y=[m["perf"]],
            mode="markers+text",
            marker=dict(size=max(wp/1.8, 8), color=col, opacity=0.72,
                        line=dict(color="white",width=1)),
            text=[nom[:12]], textposition="top center",
            textfont=dict(size=7.5, color=COLORS["dark_green"]),
            showlegend=False,
            hovertemplate=(
                f"<b>{nom}</b><br>"
                f"VaR:{m['var99']:.3f}% Perf:{m['perf']:.2f}%<br>"
                f"Score:{m['score']}/10 — {m['statut']}"
                "<extra></extra>")))
    fig_vp.update_layout(
        height=400, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(244,247,251,0.7)",
        xaxis=dict(title="VaR 99% (%/jour)", gridcolor="#E2E8F0"),
        yaxis=dict(title="Performance annuelle (%)", gridcolor="#E2E8F0"),
        margin=dict(t=20,b=40,l=50,r=10))
    st.plotly_chart(fig_vp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — COMPARAISON MÉTHODES
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">⚖️ Comparaison des 3 méthodes</div>',
                unsafe_allow_html=True)

    w_opt_all = {m: optimiser(m, w_ref) for m in METHODES}
    if not all(m in optim_pts for m in METHODES):
        optim_pts = {m: calcul_stats(w_opt_all[m]) for m in METHODES}

    c1, c2 = st.columns([1.3, 1.7])
    with c1:
        st.markdown("#### 🗺️ Heatmap des poids")
        noms_s  = [n[:14] for n in NOMS]
        mths    = ["Actuel"] + METHODES
        wmat    = np.vstack([w_ref*100, *[w_opt_all[m]*100 for m in METHODES]])
        fig_hm  = go.Figure(go.Heatmap(
            z=wmat, x=noms_s, y=mths, colorscale="Greens",
            text=np.round(wmat,1), texttemplate="%{text}%",
            textfont=dict(size=8, color="black"),
            hovertemplate="<b>%{y}</b> — %{x}<br>Poids: %{z:.1f}%<extra></extra>",
            colorbar=dict(title="Poids (%)", thickness=14), zmin=0, zmax=35))
        fig_hm.update_layout(
            height=420, paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
            yaxis=dict(tickfont=dict(size=9)),
            margin=dict(t=20,b=80,l=100,r=20))
        st.plotly_chart(fig_hm, use_container_width=True)

    with c2:
        st.markdown("#### 📈 Perf vs Vol par méthode")
        perf_m = [stats_ref["perf"]] + [optim_pts[m]["perf"] for m in METHODES]
        vol_m  = [stats_ref["vol"]]  + [optim_pts[m]["vol"]  for m in METHODES]
        noms_p = ["Actuel"] + METHODES
        c_bub  = [COLORS["gray"],COLORS["mid_green"],COLORS["navy"],COLORS["red"]]
        fig_pv = go.Figure(go.Scatter(
            x=vol_m, y=perf_m, mode="markers+text",
            marker=dict(size=30, color=c_bub, line=dict(color="white",width=2)),
            text=noms_p, textposition="middle center",
            textfont=dict(size=8, color="white", family="Arial Black"),
            hovertemplate="<b>%{text}</b><br>Vol:%{x:.2f}% Perf:%{y:.2f}%<extra></extra>"))
        fig_pv.update_layout(
            height=420, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(244,247,251,0.7)",
            xaxis=dict(title="Volatilité (%)", gridcolor="#E2E8F0"),
            yaxis=dict(title="Performance (%)", gridcolor="#E2E8F0"),
            margin=dict(t=20,b=30,l=50,r=30))
        st.plotly_chart(fig_pv, use_container_width=True)

    # VaR / CVaR
    st.markdown('<div class="section-title">📉 VaR & CVaR comparées</div>',
                unsafe_allow_html=True)
    nps  = ["Actuel"] + METHODES
    vars_  = [stats_ref["var99"]]  + [optim_pts[m]["var99"]  for m in METHODES]
    cvars_ = [stats_ref["cvar99"]] + [optim_pts[m]["cvar99"] for m in METHODES]
    cols4  = [COLORS["gray"],COLORS["mid_green"],COLORS["navy"],COLORS["red"]]
    fig_v  = go.Figure()
    fig_v.add_trace(go.Bar(name="VaR 99%",  x=nps, y=vars_,  marker_color=cols4,
                           opacity=0.85, text=[f"{v:.4f}%" for v in vars_],
                           textposition="inside", textfont=dict(color="white",size=10)))
    fig_v.add_trace(go.Bar(name="CVaR 99%", x=nps, y=cvars_, marker_color=cols4,
                           opacity=0.45, text=[f"{v:.4f}%" for v in cvars_],
                           textposition="inside", textfont=dict(color="white",size=9)))
    fig_v.update_layout(barmode="group", height=290,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(244,247,251,0.7)",
                        yaxis=dict(title="(%/jour)", gridcolor="#E2E8F0"),
                        legend=dict(orientation="h",x=0,y=1.12),
                        margin=dict(t=40,b=20,l=50,r=20))
    st.plotly_chart(fig_v, use_container_width=True)

    # Sharpe / Sortino
    st.markdown('<div class="section-title">⭐ Sharpe & Sortino comparés</div>',
                unsafe_allow_html=True)
    sharpes_  = [stats_ref["sharpe"]]  + [optim_pts[m]["sharpe"]  for m in METHODES]
    sortinos_ = [stats_ref["sortino"]] + [optim_pts[m]["sortino"] for m in METHODES]
    fig_s = go.Figure()
    fig_s.add_trace(go.Bar(name="Sharpe",  x=nps, y=sharpes_,  marker_color=cols4,
                           opacity=0.85, text=[f"{v:.3f}" for v in sharpes_],
                           textposition="outside"))
    fig_s.add_trace(go.Bar(name="Sortino", x=nps, y=sortinos_, marker_color=cols4,
                           opacity=0.45, text=[f"{v:.3f}" for v in sortinos_],
                           textposition="outside"))
    fig_s.update_layout(barmode="group", height=280,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(244,247,251,0.7)",
                        yaxis=dict(title="Ratio", gridcolor="#E2E8F0",
                                   range=[0, max(sharpes_+sortinos_)*1.15]),
                        legend=dict(orientation="h",x=0,y=1.12),
                        margin=dict(t=40,b=20,l=50,r=20))
    st.plotly_chart(fig_s, use_container_width=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;padding:18px;">
  <p style="color:{COLORS['gray']};font-size:0.75rem;margin:0;">
    <b>OPCVM Portfolio Dashboard</b> — Valeurs alignées avec la présentation PPT<br>
    Rf = 2.25% · Alpha de Jensen OLS · VaR historique lower · 247 jours · 14 OPCVM<br>
    3 méthodes : Min Variance · Max Sharpe · Min CVaR · Optimiseur SLSQP multi-start
  </p>
  <p style="color:{COLORS['light_green']};font-size:0.8rem;margin:8px 0 0;font-weight:600;">
    © 2026 · Ben said Raydae · Direction de la Gestion des Risques
  </p>
</div>
""", unsafe_allow_html=True)
