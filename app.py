"""
OPCVM Portfolio Dashboard — Streamlit v3
Inclut l'onglet « 🤖 Optimiseur IA » avec appel Anthropic API
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
# CONFIG GÉNÉRALE
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
    [data-testid="stSidebar"] [data-testid="stTickBarMin"],
    [data-testid="stSidebar"] [data-testid="stTickBarMax"] {{ color: {COLORS['mint']} !important; }}
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
        font-size: 0.85rem !important;
        padding: 8px 4px !important;
        width: 100% !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background-color: {COLORS['light_green']} !important;
        color: {COLORS['dark_green']} !important;
        border-color: {COLORS['light_green']} !important;
    }}
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        background-color: {COLORS['light_green']} !important;
        color: {COLORS['dark_green']} !important;
        border: 2px solid white !important;
        font-size: 0.9rem !important;
    }}
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
        background-color: white !important;
        color: {COLORS['dark_green']} !important;
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
        padding: 18px 24px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 6px solid {COLORS['light_green']};
    }}
    .main-header h1 {{ color: white; margin: 0; font-size: 1.6rem; }}
    .main-header p  {{ color: {COLORS['mint']}; margin: 4px 0 0; font-size: 0.9rem; }}
    .metric-card {{
        background: white;
        border-radius: 10px;
        padding: 14px 16px;
        border-left: 5px solid {COLORS['mid_green']};
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 10px;
        transition: transform 0.2s;
    }}
    .metric-card:hover {{ transform: translateY(-2px); }}
    .metric-card.red   {{ border-left-color: {COLORS['red']}; }}
    .metric-card.gold  {{ border-left-color: {COLORS['gold']}; }}
    .metric-card.navy  {{ border-left-color: {COLORS['navy']}; }}
    .metric-card h3 {{ margin: 0; font-size: 1.6rem; color: {COLORS['dark_green']}; font-weight: 700; }}
    .metric-card p  {{ margin: 2px 0 0; font-size: 0.8rem; color: {COLORS['gray']}; }}
    .metric-card .delta {{ font-size: 0.75rem; font-weight: 600; margin-top: 4px; }}
    .delta-pos {{ color: {COLORS['mid_green']}; }}
    .delta-neg {{ color: {COLORS['red']}; }}
    .section-title {{
        background: {COLORS['mid_green']};
        color: white;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
        margin: 16px 0 10px;
    }}
    .sum-warning {{
        background: #FFF3CD;
        border-left: 4px solid {COLORS['gold']};
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #856404;
    }}
    .sum-ok {{
        background: {COLORS['mint']};
        border-left: 4px solid {COLORS['mid_green']};
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: {COLORS['dark_green']};
    }}
    /* ── Onglet IA ── */
    .ai-tab-header {{
        background: linear-gradient(135deg, {COLORS['dark_green']}, {COLORS['navy']});
        padding: 20px 28px;
        border-radius: 12px;
        margin-bottom: 20px;
        border-left: 6px solid {COLORS['gold']};
        display: flex;
        align-items: center;
        gap: 16px;
    }}
    .ai-tab-header h2 {{ color: white; margin: 0; font-size: 1.3rem; }}
    .ai-tab-header p  {{ color: {COLORS['mint']}; margin: 4px 0 0; font-size: 0.85rem; }}
    .ai-info-box {{
        background: white;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid {COLORS['mint']};
        border-left: 5px solid {COLORS['gold']};
        margin-bottom: 16px;
        font-size: 0.88rem;
        color: {COLORS['dark_green']};
        line-height: 1.6;
    }}
    .ai-info-box b {{ color: {COLORS['mid_green']}; }}
    .stTabs [data-baseweb="tab"] {{ background: white; border-radius: 6px 6px 0 0; }}
    .stTabs [aria-selected="true"] {{ background: {COLORS['mid_green']}; color: white !important; }}
    div[data-testid="stNumberInput"] label {{ font-size: 0.8rem; color: {COLORS['dark_green']}; font-weight: 600; }}
    .stDataFrame {{ background: white; border-radius: 8px; padding: 8px; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DONNÉES OPCVM
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

META = {
    "AFG GOV BOND FUND":       {"perf": 6.41,  "vol": 3.61,  "sharpe": 1.774, "sortino": 2.524, "alpha_j": -0.82, "beta": 0.79,  "te": 1.638, "ir": -1.291, "dd": -2.71,  "var99": -0.384, "cvar99": -0.412},
    "AD BALANCED FUND":        {"perf": 33.23, "vol": 14.27, "sharpe": 2.210, "sortino": 3.242, "alpha_j": -1.99, "beta": 5.25,  "te": 12.287,"ir":  2.011, "dd": -17.80, "var99": -1.906, "cvar99": -2.145},
    "AFG OPTIMAL FUND":        {"perf": 12.66, "vol": 9.68,  "sharpe": 1.179, "sortino": 1.422, "alpha_j": -8.68, "beta": 3.04,  "te": 7.320, "ir":  0.565, "dd": -7.75,  "var99": -1.573, "cvar99": -1.831},
    "CDG IZDIHAR":             {"perf": 21.31, "vol": 10.42, "sharpe": 1.832, "sortino": 2.840, "alpha_j": -6.59, "beta": 4.09,  "te": 8.597, "ir":  1.488, "dd": -14.11, "var99": -1.293, "cvar99": -1.524},
    "AD SELECT BANK":          {"perf": 3.50,  "vol": 0.55,  "sharpe": 2.305, "sortino": 3.063, "alpha_j":  0.69, "beta": 0.09,  "te": 2.056, "ir": -2.442, "dd": -0.19,  "var99": -0.081, "cvar99": -0.094},
    "ALPHA BANQUES FUND":      {"perf": 3.23,  "vol": 0.48,  "sharpe": 2.024, "sortino": 2.706, "alpha_j":  0.46, "beta": 0.08,  "te": 2.056, "ir": -2.574, "dd": -0.18,  "var99": -0.078, "cvar99": -0.091},
    "ALPHA SECURE FUND":       {"perf": 5.44,  "vol": 1.70,  "sharpe": 1.871, "sortino": 2.393, "alpha_j":  0.24, "beta": 0.47,  "te": 1.787, "ir": -1.728, "dd": -1.59,  "var99": -0.274, "cvar99": -0.315},
    "CAM OBLIBANQUES":         {"perf": 3.32,  "vol": 0.63,  "sharpe": 1.965, "sortino": 2.741, "alpha_j":  0.53, "beta": 0.09,  "te": 2.063, "ir": -2.524, "dd": -0.17,  "var99": -0.074, "cvar99": -0.088},
    "CDG RENDEMENT":           {"perf": 12.85, "vol": 2.36,  "sharpe": 4.600, "sortino": 6.264, "alpha_j":  5.81, "beta": 0.76,  "te": 1.670, "ir":  2.590, "dd": -3.68,  "var99": -0.283, "cvar99": -0.321},
    "OBLIG CT":                {"perf": 3.40,  "vol": 0.54,  "sharpe": 2.141, "sortino": 2.652, "alpha_j":  0.61, "beta": 0.09,  "te": 2.063, "ir": -2.483, "dd": -0.19,  "var99": -0.096, "cvar99": -0.112},
    "CDG TAWFIR":              {"perf": 5.58,  "vol": 1.91,  "sharpe": 1.762, "sortino": 2.278, "alpha_j": -0.52, "beta": 0.61,  "te": 1.576, "ir": -1.868, "dd": -1.69,  "var99": -0.344, "cvar99": -0.388},
    "EMERGENCE SERENITE":      {"perf": 10.73, "vol": 2.27,  "sharpe": 3.739, "sortino": 5.365, "alpha_j":  3.17, "beta": 0.85,  "te": 1.353, "ir":  1.630, "dd": -3.60,  "var99": -0.616, "cvar99": -0.684},
    "CAPITAL TRUST EQUILIBRE": {"perf": 11.92, "vol": 9.11,  "sharpe": 1.178, "sortino": 1.420, "alpha_j": -8.53, "beta": 2.90,  "te": 6.662, "ir":  0.510, "dd": -7.63,  "var99": -1.710, "cvar99": -1.965},
    "AD YIELD FUND":           {"perf": 3.22,  "vol": 0.58,  "sharpe": 1.681, "sortino": 2.277, "alpha_j":  0.43, "beta": 0.09,  "te": 2.073, "ir": -2.561, "dd": -0.21,  "var99": -0.089, "cvar99": -0.103},
}

NOMS     = list(POIDS_ACTUELS.keys())
N        = len(NOMS)
RF       = 0.0225
RF_DAILY = RF / 252
METHODES = ["Min Variance", "Max Sharpe", "Min CVaR"]


# ── Rendements synthétiques ─────────────────────────────────────────────────
@st.cache_data
def generer_rendements_synthetiques():
    np.random.seed(42)
    T = 247
    mu_j  = np.array([META[n]["perf"] / 100 / 252     for n in NOMS])
    sig_j = np.array([META[n]["vol"]  / 100 / np.sqrt(252) for n in NOMS])
    betas = np.array([META[n]["beta"] for n in NOMS])
    r_market = np.random.normal(0.0003, 0.008, T)
    R = np.zeros((T, N))
    for i in range(N):
        alpha_j_daily = mu_j[i] - betas[i] * 0.0003
        idio_vol = max(sig_j[i] * np.sqrt(1 - min(betas[i]**2 * 0.35, 0.9)), 1e-5)
        eps = np.random.normal(0, idio_vol, T)
        R[:, i] = alpha_j_daily + betas[i] * r_market + eps
    for i in range(N):
        R[:, i] = R[:, i] - R[:, i].mean() + mu_j[i]
        scale    = sig_j[i] / (R[:, i].std() + 1e-12)
        R[:, i]  = (R[:, i] - mu_j[i]) * scale + mu_j[i]
    return R

R_GLOBAL = generer_rendements_synthetiques()


# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS CALCUL
# ══════════════════════════════════════════════════════════════════════════════

def calcul_stats(w_arr, R=R_GLOBAL):
    r_ptf   = R @ w_arr
    perf    = r_ptf.mean() * 252 * 100
    vol     = r_ptf.std()  * np.sqrt(252) * 100
    sharpe  = (perf/100 - RF) / (vol/100) if vol > 1e-8 else 0
    r_neg   = r_ptf[r_ptf < RF_DAILY]
    dv      = r_neg.std() * np.sqrt(252) if len(r_neg) > 1 else vol/100
    sortino = (perf/100 - RF) / dv if dv > 1e-8 else 0
    var99   = np.percentile(r_ptf, 1, method="linear") * 100
    cvar_mask = r_ptf <= np.percentile(r_ptf, 1)
    cvar99  = r_ptf[cvar_mask].mean() * 100 if cvar_mask.any() else var99
    cum     = np.cumprod(1 + r_ptf)
    roll    = np.maximum.accumulate(cum)
    dd_max  = ((cum - roll) / roll).min() * 100
    return {"perf": perf, "vol": vol, "sharpe": sharpe, "sortino": sortino,
            "var99": var99, "cvar99": cvar99, "dd_max": dd_max}


def optimiser(methode, w_actuel, R=R_GLOBAL):
    mu  = R.mean(axis=0) * 252
    cov = np.cov(R.T) * 252
    n   = len(w_actuel)
    bounds = [(0.01, 0.25)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    ret_min = (mu @ w_actuel) * 0.5
    if methode != "Min Variance":
        constraints.append({"type": "ineq", "fun": lambda w: (mu @ w) - ret_min})

    def cvar_fn(w):
        r = R @ w
        var_thresh = np.percentile(r, 1)
        tail = r[r <= var_thresh]
        return -tail.mean() if len(tail) > 0 else -var_thresh

    objectives = {
        "Min Variance": lambda w: w @ cov @ w,
        "Max Sharpe":   lambda w: -(mu @ w - RF) / (np.sqrt(w @ cov @ w + 1e-12)),
        "Min CVaR":     cvar_fn,
    }

    rng = np.random.default_rng(42)
    best = None
    starts = [w_actuel.copy()]
    for _ in range(8):
        w0 = rng.dirichlet(np.ones(n))
        w0 = np.clip(w0, 0.01, 0.25)
        w0 /= w0.sum()
        starts.append(w0)

    for w0 in starts:
        try:
            res = minimize(objectives[methode], w0, method="SLSQP",
                           bounds=bounds, constraints=constraints,
                           options={"maxiter": 1000, "ftol": 1e-10})
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            pass
    return best.x if best is not None and best.success else w_actuel


@st.cache_data
def calculer_frontiere(R=R_GLOBAL):
    mu  = R.mean(axis=0) * 252
    cov = np.cov(R.T) * 252
    n   = R.shape[1]
    ret_min = mu.min()
    ret_max = mu.max() * 0.85
    targets = np.linspace(ret_min, ret_max, 200)
    vols, rets, sharpes, var99s, cvar99s = [], [], [], [], []
    constraints_base = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.01, 0.25)] * n
    obj    = lambda w: w @ cov @ w
    for t in targets:
        cst = constraints_base + [{"type": "eq", "fun": lambda w, tt=t: mu @ w - tt}]
        rng = np.random.default_rng(0)
        best = None
        for _ in range(6):
            w0 = rng.dirichlet(np.ones(n))
            w0 = np.clip(w0, 0.01, 0.25)
            w0 /= w0.sum()
            try:
                res = minimize(obj, w0, method="SLSQP", bounds=bounds,
                               constraints=cst, options={"maxiter": 800, "ftol": 1e-10})
                if res.success and (best is None or res.fun < best.fun):
                    best = res
            except Exception:
                pass
        if best is not None and best.success:
            w     = best.x
            r_ptf = R @ w
            vol_p = np.sqrt(w @ cov @ w) * 100
            ret_p = mu @ w * 100
            sh_p  = (ret_p/100 - RF) / (vol_p/100) if vol_p > 1e-8 else 0
            v99   = np.percentile(r_ptf, 1, method="linear") * 100
            mask  = r_ptf <= np.percentile(r_ptf, 1)
            cv99  = r_ptf[mask].mean() * 100 if mask.any() else v99
            vols.append(vol_p); rets.append(ret_p); sharpes.append(sh_p)
            var99s.append(v99); cvar99s.append(cv99)
    return (np.array(vols), np.array(rets),
            np.array(sharpes), np.array(var99s), np.array(cvar99s))


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

for nom in NOMS:
    if f"poids_cible_{nom}" not in st.session_state:
        st.session_state[f"poids_cible_{nom}"] = float(POIDS_ACTUELS[nom])

for nom in NOMS:
    st.session_state[f"slider_{nom}"] = st.session_state[f"poids_cible_{nom}"]


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown(f"""
<div style="background:{COLORS['mid_green']};padding:12px;border-radius:8px;margin-bottom:12px;
            border-left:4px solid {COLORS['light_green']}">
  <h2 style="color:white;margin:0;font-size:1.1rem">⚖️ Pondérations OPCVM</h2>
  <p style="color:{COLORS['mint']};font-size:0.8rem;margin:4px 0 0">Ajustez les poids (1% – 25%)</p>
</div>
""", unsafe_allow_html=True)

CATEGORIES = {
    "🏦 Obligations Long Terme": [
        "AFG GOV BOND FUND", "ALPHA SECURE FUND", "CAM OBLIBANQUES",
        "CDG RENDEMENT", "OBLIG CT", "CDG TAWFIR", "EMERGENCE SERENITE"
    ],
    "📈 Fonds Diversifiés": [
        "AD BALANCED FUND", "AFG OPTIMAL FUND", "CDG IZDIHAR", "CAPITAL TRUST EQUILIBRE"
    ],
    "💰 Obligations Court Terme": [
        "AD SELECT BANK", "ALPHA BANQUES FUND", "AD YIELD FUND"
    ],
}

poids_user = {}
for cat, fonds in CATEGORIES.items():
    with st.sidebar.expander(cat, expanded=True):
        for nom in fonds:
            poids_user[nom] = st.slider(
                nom.replace("FCP ", "").replace("SICAV ", ""),
                min_value=0.0, max_value=25.0, step=0.1,
                key=f"slider_{nom}", format="%.1f%%"
            )
            st.session_state[f"poids_cible_{nom}"] = poids_user[nom]

total_poids = sum(poids_user.values())
delta_sum   = total_poids - 100.0

if abs(delta_sum) < 0.05:
    st.sidebar.markdown(f'<div class="sum-ok">✅ Somme = {total_poids:.1f}%</div>', unsafe_allow_html=True)
else:
    signe = "+" if delta_sum > 0 else ""
    st.sidebar.markdown(
        f'<div class="sum-warning">⚠️ Somme = {total_poids:.1f}% ({signe}{delta_sum:.1f}pp)<br>'
        f'Les résultats sont normalisés automatiquement.</div>',
        unsafe_allow_html=True
    )

st.sidebar.markdown("---")

if st.sidebar.button("🔄 Réinitialiser les poids", use_container_width=True):
    for nom in NOMS:
        st.session_state[f"poids_cible_{nom}"] = float(POIDS_ACTUELS[nom])
    st.rerun()

methode_optim = st.sidebar.selectbox("🎯 Méthode d'optimisation", METHODES, key="methode_select")

if st.sidebar.button(f"⚡ Appliquer : {methode_optim}", use_container_width=True, type="primary"):
    w_cur = np.array([poids_user[n] / 100 for n in NOMS])
    w_cur = np.clip(w_cur, 0.01, 0.25)
    w_cur /= w_cur.sum()
    w_opt = optimiser(methode_optim, w_cur)
    for i, nom in enumerate(NOMS):
        st.session_state[f"poids_cible_{nom}"] = round(float(w_opt[i]) * 100, 1)
    st.rerun()

st.sidebar.markdown(f"""
<div style="margin-top:12px;padding:8px;background:{COLORS['dark_green']};border-radius:6px;
            font-size:0.75rem;color:{COLORS['gray']};text-align:center;">
  📊 Rf = 2.25% · 247 obs · 14 OPCVM
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CALCULS PRINCIPAUX
# ══════════════════════════════════════════════════════════════════════════════

w_raw     = np.array([poids_user[n] / 100 for n in NOMS])
w_norm    = w_raw / w_raw.sum()
stats_cur = calcul_stats(w_norm)

w_ref     = np.array([POIDS_ACTUELS[n] / 100 for n in NOMS])
stats_ref = calcul_stats(w_ref)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="main-header">
  <h1>📊 OPCVM Portfolio Dashboard</h1>
  <p>Optimisation interactive · Frontière Efficiente · Rf = 2.25% · 14 OPCVM · 247 observations</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ONGLETS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Tableau de Bord",
    "🎯 Frontière Efficiente",
    "📋 Analyse Détaillée",
    "⚖️ Comparaison Méthodes",
    "🤖 Optimiseur IA",
    "📐 Markowitz Exact",         # ← NOUVEL ONGLET
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — TABLEAU DE BORD
# ═══════════════════════════════════════════════════════════════════
with tab1:

    st.markdown('<div class="section-title">📌 Indicateurs du Portefeuille Courant</div>', unsafe_allow_html=True)

    def delta_html(val, ref, unit="%", inverse=False):
        diff = val - ref
        good = diff > 0 if not inverse else diff < 0
        signe = "+" if diff > 0 else ""
        cls = "delta-pos" if good else "delta-neg"
        return f'<span class="{cls}">{signe}{diff:.2f}{unit} vs actuel</span>'

    kpis = [
        ("Performance Annuelle",  f"{stats_cur['perf']:.2f}%",   stats_cur["perf"],    stats_ref["perf"],    "%", False, ""),
        ("Volatilité Annualisée", f"{stats_cur['vol']:.2f}%",    stats_cur["vol"],     stats_ref["vol"],     "%", True,  "red"),
        ("Sharpe Ratio",          f"{stats_cur['sharpe']:.3f}",  stats_cur["sharpe"],  stats_ref["sharpe"],  "",  False, "gold"),
        ("Sortino Ratio",         f"{stats_cur['sortino']:.3f}", stats_cur["sortino"], stats_ref["sortino"], "",  False, "gold"),
        ("VaR 99% (journalière)", f"{stats_cur['var99']:.4f}%",  stats_cur["var99"],   stats_ref["var99"],   "%", True,  "red"),
        ("CVaR 99% (journalière)",f"{stats_cur['cvar99']:.4f}%", stats_cur["cvar99"],  stats_ref["cvar99"],  "%", True,  "red"),
        ("Maximum Drawdown",      f"{stats_cur['dd_max']:.2f}%", stats_cur["dd_max"],  stats_ref["dd_max"],  "%", True,  "navy"),
    ]

    cols = st.columns(len(kpis))
    for i, (label, val_str, val, ref, unit, inverse, card_class) in enumerate(kpis):
        dh = delta_html(val, ref, unit, inverse)
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card {card_class}">
              <p>{label}</p>
              <h3>{val_str}</h3>
              <div class="delta">{dh}</div>
            </div>""", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 1.9])

    with col_left:
        st.markdown('<div class="section-title">📊 Répartition des Poids</div>', unsafe_allow_html=True)
        poids_pct = w_norm * 100
        fig_pie = go.Figure(go.Pie(
            labels=[n.replace("FCP ", "").replace("SICAV ", "") for n in NOMS],
            values=poids_pct, textinfo="label+percent", textfont_size=10,
            marker=dict(colors=px.colors.qualitative.Set3, line=dict(color="white", width=1.5)),
            hole=0.35,
        ))
        fig_pie.update_layout(
            showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
            height=380, paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text=f"<b>Total: {total_poids:.0f}%</b>", x=0.5, y=0.5,
                              font_size=14, showarrow=False, font=dict(color=COLORS["dark_green"]))]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-title">📊 Comparaison des Poids</div>', unsafe_allow_html=True)
        noms_short  = [n[:15] for n in NOMS]
        delta_poids = poids_pct - np.array([POIDS_ACTUELS[n] for n in NOMS])
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Poids Actuel", x=noms_short,
            y=[POIDS_ACTUELS[n] for n in NOMS],
            marker_color=COLORS["gray"], opacity=0.7,
            text=[f"{POIDS_ACTUELS[n]:.1f}%" for n in NOMS],
            textposition="outside", textfont=dict(size=8)
        ))
        fig_bar.add_trace(go.Bar(
            name="Poids Courant", x=noms_short, y=poids_pct,
            marker_color=[COLORS["mid_green"] if d >= 0 else COLORS["red"] for d in delta_poids],
            opacity=0.85,
            text=[f"{p:.1f}%" for p in poids_pct],
            textposition="outside", textfont=dict(size=8)
        ))
        fig_bar.add_hline(y=1,  line_dash="dot", line_color=COLORS["light_green"], line_width=1)
        fig_bar.add_hline(y=25, line_dash="dot", line_color=COLORS["red"],         line_width=1)
        fig_bar.update_layout(
            barmode="group", height=380,
            margin=dict(t=10, b=80, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(244,247,251,0.5)",
            xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
            yaxis=dict(title="Poids (%)", gridcolor="#E2E8F0", range=[0, 30]),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-title">📈 Performance Cumulée Simulée (247 jours)</div>', unsafe_allow_html=True)
    r_cur   = R_GLOBAL @ w_norm
    r_ref   = R_GLOBAL @ w_ref
    cum_cur = (1 + r_cur).cumprod() - 1
    cum_ref = (1 + r_ref).cumprod() - 1
    t_idx   = np.arange(len(r_cur))
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=t_idx, y=cum_ref * 100, name="Portefeuille Actuel",
        line=dict(color=COLORS["gray"], dash="dash", width=2), opacity=0.8
    ))
    fig_cum.add_trace(go.Scatter(
        x=t_idx, y=cum_cur * 100, name="Portefeuille Courant",
        line=dict(color=COLORS["mid_green"], width=2.5),
        fill="tonexty", fillcolor="rgba(44,95,45,0.1)"
    ))
    fig_cum.add_hline(y=0, line_color=COLORS["gray"], line_width=0.8)
    fig_cum.update_layout(
        height=280, margin=dict(t=10, b=30, l=40, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(244,247,251,0.5)",
        legend=dict(orientation="h", x=0, y=1.12),
        yaxis=dict(title="Performance Cumulative (%)", gridcolor="#E2E8F0"),
        xaxis=dict(title="Jours", gridcolor="#E2E8F0"),
        hovermode="x unified"
    )
    st.plotly_chart(fig_cum, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — FRONTIÈRE EFFICIENTE
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">🎯 Frontière Efficiente de Markowitz — Vue enrichie</div>',
                unsafe_allow_html=True)

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        colorby = st.selectbox("Colorier la frontière par :",
                               ["Ratio de Sharpe", "VaR 99%", "CVaR 99%"], index=0)
    with ctrl2:
        show_indiv = st.checkbox("Afficher les OPCVM individuels", value=True)
    with ctrl3:
        show_cml = st.checkbox("Afficher la CML", value=True)

    with st.spinner("Calcul de la frontière efficiente (200 points)..."):
        fe_vols, fe_rets, fe_sharpes, fe_var99s, fe_cvar99s = calculer_frontiere()

    optim_points = {}
    with st.spinner("Calcul des 3 portefeuilles optimisés..."):
        for m in METHODES:
            w_opt = optimiser(m, w_ref)
            optim_points[m] = calcul_stats(w_opt)

    col_opt    = {"Min Variance": COLORS["mid_green"], "Max Sharpe": COLORS["navy"], "Min CVaR": COLORS["red"]}
    markers_m  = {"Min Variance": "square", "Max Sharpe": "triangle-up", "Min CVaR": "diamond"}

    color_map_choice = {
        "Ratio de Sharpe": (fe_sharpes, "Sharpe",             "RdYlGn",   False),
        "VaR 99%":         (fe_var99s,  "VaR 99%<br>(%/j)",  "RdYlGn_r", False),
        "CVaR 99%":        (fe_cvar99s, "CVaR 99%<br>(%/j)", "RdYlGn_r", False),
    }
    color_vals, color_title, colorscale, _ = color_map_choice[colorby]

    fig_fe = go.Figure()

    if len(fe_vols) > 0:
        hover_fe = [
            f"Vol: {v:.2f}%<br>Perf: {r:.2f}%<br>Sharpe: {s:.3f}<br>VaR: {va:.4f}%<br>CVaR: {cv:.4f}%"
            for v, r, s, va, cv in zip(fe_vols, fe_rets, fe_sharpes, fe_var99s, fe_cvar99s)
        ]
        fig_fe.add_trace(go.Scatter(
            x=fe_vols, y=fe_rets, mode="markers",
            marker=dict(color=color_vals, colorscale=colorscale, size=8, opacity=0.9,
                        colorbar=dict(title=color_title, thickness=14, len=0.55, y=0.75, yanchor="top"),
                        line=dict(width=0)),
            name="Frontière Efficiente",
            hovertemplate="%{customdata}<extra>Frontière Efficiente</extra>",
            customdata=hover_fe,
        ))
        sorted_idx = np.argsort(fe_vols)
        fig_fe.add_trace(go.Scatter(
            x=fe_vols[sorted_idx], y=fe_rets[sorted_idx], mode="lines",
            line=dict(color="rgba(150,180,150,0.35)", width=3),
            showlegend=False, hoverinfo="skip",
        ))

    tang_vol = tang_ret = tang_sh = None
    if len(fe_sharpes) > 0:
        tang_idx = np.argmax(fe_sharpes)
        tang_vol = fe_vols[tang_idx]
        tang_ret = fe_rets[tang_idx]
        tang_sh  = fe_sharpes[tang_idx]

        fig_fe.add_trace(go.Scatter(
            x=[tang_vol], y=[tang_ret], mode="markers+text",
            marker=dict(size=22, color=COLORS["gold"], symbol="star",
                        line=dict(color=COLORS["dark_green"], width=2)),
            text=["Portefeuille<br>Tangent"], textposition="top right",
            textfont=dict(size=10, color=COLORS["gold"], family="Arial Black"),
            name=f"Tangent (Sharpe={tang_sh:.3f})",
            hovertemplate=(f"<b>Portefeuille Tangent</b><br>Vol: {tang_vol:.2f}%<br>"
                           f"Perf: {tang_ret:.2f}%<br>Sharpe: {tang_sh:.3f}<extra></extra>"),
        ))

        if show_cml:
            slope_cml = (tang_ret / 100 - RF) / (tang_vol / 100 + 1e-8)
            vol_end   = tang_vol * 1.5
            vol_cml   = np.array([0, tang_vol, vol_end])
            ret_cml   = (RF + slope_cml * vol_cml / 100) * 100
            fig_fe.add_trace(go.Scatter(
                x=vol_cml[:2], y=ret_cml[:2], mode="lines",
                line=dict(color=COLORS["gold"], width=2.5),
                name=f"CML (Rf={RF*100:.2f}%)", hoverinfo="skip",
            ))
            fig_fe.add_trace(go.Scatter(
                x=vol_cml[1:], y=ret_cml[1:], mode="lines",
                line=dict(color=COLORS["gold"], width=1.5, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ))
            fig_fe.add_trace(go.Scatter(
                x=[0], y=[RF * 100], mode="markers+text",
                marker=dict(size=10, color=COLORS["gold"], symbol="circle",
                            line=dict(color="white", width=1.5)),
                text=[f"Rf={RF*100:.2f}%"], textposition="top right",
                textfont=dict(size=9, color=COLORS["gold"]),
                showlegend=False, hoverinfo="skip",
            ))

    if show_indiv:
        for nom in NOMS:
            m   = META[nom]
            aj  = m["alpha_j"]
            col = COLORS["mid_green"] if aj > 0 else COLORS["red"]
            fig_fe.add_trace(go.Scatter(
                x=[m["vol"]], y=[m["perf"]], mode="markers+text",
                marker=dict(size=9, color=col, symbol="circle", opacity=0.75,
                            line=dict(color="white", width=1)),
                text=[nom[:12]], textposition="top center",
                textfont=dict(size=7.5, color=COLORS["dark_green"]),
                name=nom[:20], showlegend=False,
                hovertemplate=(f"<b>{nom}</b><br>Vol: {m['vol']:.2f}%<br>Perf: {m['perf']:.2f}%<br>"
                               f"Sharpe: {m['sharpe']:.3f}<br>α: {aj:+.2f}%<extra>OPCVM individuel</extra>"),
            ))

    fig_fe.add_trace(go.Scatter(
        x=[stats_ref["vol"]], y=[stats_ref["perf"]], mode="markers+text",
        marker=dict(size=18, color=COLORS["gray"], symbol="square",
                    line=dict(color="white", width=2)),
        text=["Actuel"], textposition="top center",
        textfont=dict(size=10, color=COLORS["gray"], family="Arial Black"),
        name="Portefeuille Actuel",
        hovertemplate=(f"<b>Portefeuille Actuel</b><br>Vol: {stats_ref['vol']:.2f}%<br>"
                       f"Perf: {stats_ref['perf']:.2f}%<br>Sharpe: {stats_ref['sharpe']:.3f}<extra></extra>"),
    ))
    fig_fe.add_trace(go.Scatter(
        x=[stats_cur["vol"]], y=[stats_cur["perf"]], mode="markers+text",
        marker=dict(size=22, color=COLORS["light_green"], symbol="star",
                    line=dict(color=COLORS["dark_green"], width=2.5)),
        text=["Votre portefeuille"], textposition="top left",
        textfont=dict(size=10, color=COLORS["mid_green"], family="Arial Black"),
        name="Portefeuille Courant",
        hovertemplate=(f"<b>Portefeuille Courant</b><br>Vol: {stats_cur['vol']:.2f}%<br>"
                       f"Perf: {stats_cur['perf']:.2f}%<br>Sharpe: {stats_cur['sharpe']:.3f}<extra></extra>"),
    ))

    for m_name, s in optim_points.items():
        fig_fe.add_trace(go.Scatter(
            x=[s["vol"]], y=[s["perf"]], mode="markers+text",
            marker=dict(size=16, color=col_opt[m_name], symbol=markers_m[m_name],
                        line=dict(color="white", width=1.5)),
            text=[m_name[:12]], textposition="bottom center",
            textfont=dict(size=8.5, color=col_opt[m_name], family="Arial Black"),
            name=m_name,
            hovertemplate=(f"<b>{m_name}</b><br>Vol: {s['vol']:.2f}%<br>Perf: {s['perf']:.2f}%<br>"
                           f"Sharpe: {s['sharpe']:.3f}<br>VaR: {s['var99']:.4f}%<extra></extra>"),
        ))

    if len(fe_vols) > 0:
        dist = np.sqrt((fe_vols - stats_ref["vol"])**2 + (fe_rets - stats_ref["perf"])**2)
        nearest_idx = np.argmin(dist)
        fig_fe.add_trace(go.Scatter(
            x=[stats_ref["vol"], fe_vols[nearest_idx]],
            y=[stats_ref["perf"], fe_rets[nearest_idx]], mode="lines",
            line=dict(color=COLORS["gray"], dash="dot", width=1.2),
            showlegend=False, hoverinfo="skip",
        ))
        fig_fe.add_annotation(
            x=(stats_ref["vol"] + fe_vols[nearest_idx]) / 2,
            y=(stats_ref["perf"] + fe_rets[nearest_idx]) / 2,
            text="Δ vers frontière", showarrow=False,
            font=dict(size=8, color=COLORS["gray"]),
            bgcolor="rgba(255,255,255,0.7)",
        )

    fig_fe.update_layout(
        height=650, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(244,247,251,0.7)",
        xaxis=dict(title="Volatilité Annualisée (%)", gridcolor="#E2E8F0", zeroline=False,
                   showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="#CBD5E1"),
        yaxis=dict(title="Performance Annualisée (%)", gridcolor="#E2E8F0", zeroline=False,
                   showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="#CBD5E1"),
        legend=dict(orientation="v", x=1.12, y=1, bgcolor="rgba(255,255,255,0.92)",
                    bordercolor=COLORS["light_green"], borderwidth=1, font=dict(size=9), tracegroupgap=4),
        margin=dict(t=20, b=40, l=55, r=220), font=dict(family="Calibri"), hovermode="closest",
    )
    st.plotly_chart(fig_fe, use_container_width=True)

    leg1, leg2, leg3, leg4, leg5 = st.columns(5)
    for col_w, icon, label, sub, border in [
        (leg1, "⭐", "Tangent", "Max Sharpe sur la frontière", COLORS["gold"]),
        (leg2, "🟩", "Min Variance", "Risque minimal", COLORS["mid_green"]),
        (leg3, "🔺", "Max Sharpe", "Meilleur ratio", COLORS["navy"]),
        (leg4, "💎", "Min CVaR", "Risque extrême minimal", COLORS["red"]),
        (leg5, "⬛", "Actuel", "Portefeuille référence", COLORS["gray"]),
    ]:
        with col_w:
            st.markdown(f"""<div style="text-align:center;padding:8px;background:white;border-radius:8px;
            border:2px solid {border};font-size:0.8rem;">{icon} <b>{label}</b><br>{sub}</div>""",
            unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 Résumé des 3 Méthodes d\'Optimisation</div>', unsafe_allow_html=True)
    rows_m = []
    for m_name, s in optim_points.items():
        rows_m.append({"Méthode": m_name, "Performance (%)": f"{s['perf']:.2f}%",
                        "Volatilité (%)": f"{s['vol']:.2f}%", "Sharpe": f"{s['sharpe']:.3f}",
                        "Sortino": f"{s['sortino']:.3f}", "VaR 99% (j)": f"{s['var99']:.4f}%",
                        "CVaR 99% (j)": f"{s['cvar99']:.4f}%", "Drawdown Max (%)": f"{s['dd_max']:.2f}%"})
    if tang_vol is not None:
        tang_s = calcul_stats(optimiser("Max Sharpe", w_ref))
        rows_m.insert(0, {"Méthode": "⭐ Tangent (FE)", "Performance (%)": f"{tang_ret:.2f}%",
                          "Volatilité (%)": f"{tang_vol:.2f}%", "Sharpe": f"{tang_sh:.3f}",
                          "Sortino": f"{tang_s['sortino']:.3f}", "VaR 99% (j)": f"{fe_var99s[tang_idx]:.4f}%",
                          "CVaR 99% (j)": f"{fe_cvar99s[tang_idx]:.4f}%", "Drawdown Max (%)": f"{tang_s['dd_max']:.2f}%"})
    st.dataframe(pd.DataFrame(rows_m).set_index("Méthode"), use_container_width=True, height=230)


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — ANALYSE DÉTAILLÉE
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">📋 Indicateurs Individuels par OPCVM</div>', unsafe_allow_html=True)

    rows_d = []
    for i, nom in enumerate(NOMS):
        m  = META[nom]
        wp = w_norm[i] * 100
        wa = POIDS_ACTUELS[nom]
        rows_d.append({
            "OPCVM": nom[:28], "Poids Actuel (%)": f"{wa:.1f}%", "Poids Courant (%)": f"{wp:.1f}%",
            "Δ (pp)": f"{wp-wa:+.1f}", "Performance (%)": f"{m['perf']:.2f}%",
            "Volatilité (%)": f"{m['vol']:.2f}%", "Beta (β)": f"{m['beta']:.2f}",
            "Alpha Jensen (%)": f"{m['alpha_j']:+.2f}%", "Sharpe": f"{m['sharpe']:.3f}",
            "Sortino": f"{m['sortino']:.3f}", "VaR 99% (%)": f"{m['var99']:.3f}%",
            "CVaR 99% (%)": f"{m['cvar99']:.3f}%", "Tracking Error (%)": f"{m['te']:.3f}%",
            "Information Ratio": f"{m['ir']:.3f}", "Drawdown Max (%)": f"{m['dd']:.2f}%",
        })

    st.dataframe(pd.DataFrame(rows_d).set_index("OPCVM"), use_container_width=True, height=500)

    st.markdown('<div class="section-title">🔵 Alpha Jensen vs Sharpe Ratio (Taille = Poids Courant)</div>',
                unsafe_allow_html=True)
    fig_sc = go.Figure()
    for i, nom in enumerate(NOMS):
        m  = META[nom]
        wp = w_norm[i] * 100
        col = COLORS["mid_green"] if m["alpha_j"] > 0 else COLORS["red"]
        fig_sc.add_trace(go.Scatter(
            x=[m["alpha_j"]], y=[m["sharpe"]], mode="markers+text",
            marker=dict(size=max(wp / 2, 8), color=col, opacity=0.7,
                        line=dict(color="white", width=1)),
            text=[nom[:14]], textposition="top center",
            textfont=dict(size=9, color=COLORS["dark_green"]),
            name=nom[:20], showlegend=False,
            hovertemplate=(f"<b>{nom}</b><br>α: {m['alpha_j']:+.2f}%<br>"
                           f"Sharpe: {m['sharpe']:.3f}<br>Poids: {wp:.1f}%<extra></extra>"),
        ))
    fig_sc.add_vline(x=0, line_dash="dot", line_color=COLORS["red"],  line_width=1.5)
    fig_sc.add_hline(y=2, line_dash="dot", line_color=COLORS["navy"], line_width=1.5)
    fig_sc.update_layout(
        height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(244,247,251,0.7)",
        xaxis=dict(title="Alpha Jensen (%)", gridcolor="#E2E8F0", zeroline=True),
        yaxis=dict(title="Sharpe Ratio",     gridcolor="#E2E8F0", zeroline=True),
        margin=dict(t=30, b=40, l=50, r=80),
    )
    st.plotly_chart(fig_sc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — COMPARAISON MÉTHODES
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">⚖️ Comparaison Visuelle des 3 Méthodes d\'Optimisation</div>',
                unsafe_allow_html=True)

    with st.spinner("Calcul des poids optimisés..."):
        w_optimises = {}
        for m in METHODES:
            w_optimises[m] = optimiser(m, w_ref)
            if m not in optim_points:
                optim_points[m] = calcul_stats(w_optimises[m])

    col_h, col_pv = st.columns([1.3, 1.7])

    with col_h:
        st.markdown("#### 📊 Matrice des Poids (%)")
        noms_short  = [n[:14] for n in NOMS]
        methodes_h  = ["Actuel"] + list(w_optimises.keys())
        w_matrix    = np.vstack([w_ref * 100, *[w_optimises[m] * 100 for m in w_optimises]])
        fig_hm = go.Figure(go.Heatmap(
            z=w_matrix, x=noms_short, y=methodes_h, colorscale="Greens",
            text=np.round(w_matrix, 1), texttemplate="%{text}%", textfont=dict(size=8, color="black"),
            hovertemplate="<b>%{y}</b> — %{x}<br>Poids: %{z:.1f}%<extra></extra>",
            colorbar=dict(title="Poids (%)", thickness=15, len=0.8), zmin=0, zmax=25,
        ))
        fig_hm.update_layout(
            height=420, paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
            yaxis=dict(tickfont=dict(size=9)),
            margin=dict(t=30, b=80, l=100, r=20),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    with col_pv:
        st.markdown("#### 📈 Performance vs Volatilité")
        perf_m   = [stats_ref["perf"]] + [optim_points[m]["perf"] for m in optim_points]
        vol_m    = [stats_ref["vol"]]  + [optim_points[m]["vol"]  for m in optim_points]
        noms_ptf = ["Actuel"] + list(optim_points.keys())
        cols_bar2 = [COLORS["gray"], COLORS["mid_green"], COLORS["navy"], COLORS["red"]]
        fig_pv = go.Figure()
        fig_pv.add_trace(go.Scatter(
            x=vol_m, y=perf_m, mode="markers+text",
            marker=dict(size=28, color=cols_bar2, line=dict(color="white", width=2)),
            text=noms_ptf, textposition="middle center",
            textfont=dict(size=8, color="white", family="Arial Black"),
            hovertemplate="<b>%{text}</b><br>Vol: %{x:.2f}%<br>Perf: %{y:.2f}%<extra></extra>",
        ))
        if len(vol_m) > 1:
            z  = np.polyfit(vol_m, perf_m, 1)
            p  = np.poly1d(z)
            xt = np.linspace(min(vol_m) * 0.9, max(vol_m) * 1.1, 100)
            fig_pv.add_trace(go.Scatter(
                x=xt, y=p(xt), mode="lines", name="Tendance",
                line=dict(color=COLORS["gold"], dash="dash", width=1.5), opacity=0.5
            ))
        fig_pv.update_layout(
            height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(244,247,251,0.7)",
            xaxis=dict(title="Volatilité Annualisée (%)", gridcolor="#E2E8F0"),
            yaxis=dict(title="Performance Annualisée (%)", gridcolor="#E2E8F0"),
            margin=dict(t=20, b=30, l=50, r=30), hovermode="closest",
        )
        st.plotly_chart(fig_pv, use_container_width=True)

    st.markdown('<div class="section-title">📉 Comparaison des Mesures de Risque</div>', unsafe_allow_html=True)
    vars_ptf  = [stats_ref["var99"]]  + [optim_points[m]["var99"]  for m in optim_points]
    cvars_ptf = [stats_ref["cvar99"]] + [optim_points[m]["cvar99"] for m in optim_points]
    fig_var = go.Figure()
    fig_var.add_trace(go.Bar(
        name="VaR 99%", x=noms_ptf, y=vars_ptf, marker_color=cols_bar2, opacity=0.85,
        text=[f"{v:.4f}%" for v in vars_ptf], textposition="inside", textfont=dict(color="white", size=10),
    ))
    fig_var.add_trace(go.Bar(
        name="CVaR 99%", x=noms_ptf, y=cvars_ptf, marker_color=cols_bar2, opacity=0.45,
        text=[f"{v:.4f}%" for v in cvars_ptf], textposition="inside", textfont=dict(color="white", size=9),
    ))
    fig_var.update_layout(
        barmode="group", height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(244,247,251,0.7)",
        yaxis=dict(title="VaR / CVaR (%/jour)", gridcolor="#E2E8F0"),
        legend=dict(orientation="h", x=0, y=1.12), margin=dict(t=40, b=30, l=50, r=30),
    )
    st.plotly_chart(fig_var, use_container_width=True)

    st.markdown('<div class="section-title">⭐ Comparaison Sharpe & Sortino</div>', unsafe_allow_html=True)
    sharpes  = [stats_ref["sharpe"]]  + [optim_points[m]["sharpe"]  for m in optim_points]
    sortinos = [stats_ref["sortino"]] + [optim_points[m]["sortino"] for m in optim_points]
    fig_sh = go.Figure()
    fig_sh.add_trace(go.Bar(
        name="Sharpe", x=noms_ptf, y=sharpes, marker_color=cols_bar2, opacity=0.85,
        text=[f"{v:.3f}" for v in sharpes], textposition="outside"
    ))
    fig_sh.add_trace(go.Bar(
        name="Sortino", x=noms_ptf, y=sortinos, marker_color=cols_bar2, opacity=0.45,
        text=[f"{v:.3f}" for v in sortinos], textposition="outside"
    ))
    fig_sh.update_layout(
        barmode="group", height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(244,247,251,0.7)",
        yaxis=dict(title="Ratio", gridcolor="#E2E8F0",
                   range=[0, max(sharpes + sortinos) * 1.15]),
        legend=dict(orientation="h", x=0, y=1.12), margin=dict(t=40, b=30, l=50, r=30),
    )
    st.plotly_chart(fig_sh, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 5 — OPTIMISEUR IA  (NOUVEAU)
# ═══════════════════════════════════════════════════════════════════
with tab5:

    # ── En-tête ──────────────────────────────────────────────────
    st.markdown(f"""
    <div class="ai-tab-header">
      <div style="font-size:2.8rem">🤖</div>
      <div>
        <h2>Optimiseur IA — Markowitz Intelligent</h2>
        <p>Analyse conversationnelle de votre portefeuille · Powered by Claude claude-sonnet-4-20250514</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Note d'information ───────────────────────────────────────
    st.markdown(f"""
    <div class="ai-info-box">
      <b>🧠 Comment ça fonctionne ?</b><br>
      L'IA analyse en temps réel les <b>métriques actuelles de votre portefeuille</b> (performance, volatilité, Sharpe,
      VaR, CVaR, drawdown) ainsi que les <b>indicateurs individuels des 14 OPCVM</b> (Alpha Jensen, beta, Sortino…)
      pour vous fournir une analyse Markowitz personnalisée et des recommandations d'allocation.<br><br>
      <b>Exemples de questions :</b> « Quels OPCVM dois-je surpondérer pour maximiser mon Sharpe ? »,
      « Analyse le risque de mon portefeuille actuel », « Propose une allocation pour un profil prudent »,
      « Explique pourquoi CDG Rendement a le meilleur Sharpe ».
    </div>
    """, unsafe_allow_html=True)

    # ── Résumé contextuel du portefeuille courant ─────────────────
    with st.expander("📊 Contexte transmis à l'IA (portefeuille courant)", expanded=False):
        ctx_rows = []
        for i, nom in enumerate(NOMS):
            m  = META[nom]
            wp = w_norm[i] * 100
            ctx_rows.append({
                "OPCVM": nom, "Poids (%)": f"{wp:.1f}%",
                "Perf (%)": f"{m['perf']:.2f}%", "Vol (%)": f"{m['vol']:.2f}%",
                "Sharpe": f"{m['sharpe']:.3f}", "Alpha Jensen": f"{m['alpha_j']:+.2f}%",
                "Beta": f"{m['beta']:.2f}", "CVaR 99%": f"{m['cvar99']:.3f}%",
            })
        st.dataframe(pd.DataFrame(ctx_rows).set_index("OPCVM"), use_container_width=True)
        col_ctx1, col_ctx2, col_ctx3, col_ctx4 = st.columns(4)
        col_ctx1.metric("Performance portefeuille", f"{stats_cur['perf']:.2f}%")
        col_ctx2.metric("Volatilité",               f"{stats_cur['vol']:.2f}%")
        col_ctx3.metric("Sharpe",                   f"{stats_cur['sharpe']:.3f}")
        col_ctx4.metric("Max Drawdown",             f"{stats_cur['dd_max']:.2f}%")

    # ── Construction du contexte pour l'IA ───────────────────────
    def build_portfolio_context():
        lines = [
            "=== CONTEXTE PORTEFEUILLE OPCVM MAROCAIN ===",
            f"Taux sans risque (Rf) : {RF*100:.2f}%",
            f"Nombre d'OPCVM : {N}",
            "",
            "--- MÉTRIQUES DU PORTEFEUILLE COURANT ---",
            f"Performance annualisée : {stats_cur['perf']:.2f}%",
            f"Volatilité annualisée  : {stats_cur['vol']:.2f}%",
            f"Sharpe Ratio           : {stats_cur['sharpe']:.3f}",
            f"Sortino Ratio          : {stats_cur['sortino']:.3f}",
            f"VaR 99% (journalière)  : {stats_cur['var99']:.4f}%",
            f"CVaR 99% (journalière) : {stats_cur['cvar99']:.4f}%",
            f"Maximum Drawdown       : {stats_cur['dd_max']:.2f}%",
            "",
            "--- ALLOCATIONS ET MÉTRIQUES PAR OPCVM ---",
        ]
        for i, nom in enumerate(NOMS):
            m  = META[nom]
            wp = w_norm[i] * 100
            lines.append(
                f"{nom}: poids={wp:.1f}%, perf={m['perf']:.2f}%, vol={m['vol']:.2f}%, "
                f"sharpe={m['sharpe']:.3f}, sortino={m['sortino']:.3f}, "
                f"alpha_j={m['alpha_j']:+.2f}%, beta={m['beta']:.2f}, "
                f"VaR99={m['var99']:.3f}%, CVaR99={m['cvar99']:.3f}%, "
                f"te={m['te']:.3f}%, ir={m['ir']:.3f}, dd={m['dd']:.2f}%"
            )
        lines += [
            "",
            "--- CONTRAINTES D'OPTIMISATION ---",
            "Allocation min par fonds : 1%",
            "Allocation max par fonds : 25%",
            "Somme des poids = 100%",
            "Méthodes disponibles : Min Variance, Max Sharpe, Min CVaR",
        ]
        return "\n".join(lines)

    SYSTEM_PROMPT = f"""Tu es un expert en gestion de portefeuille et optimisation Markowitz, spécialisé dans les OPCVM marocains.
Tu analyses les portefeuilles selon la théorie moderne du portefeuille (MPT).

{build_portfolio_context()}

INSTRUCTIONS :
- Réponds en français, de façon précise et professionnelle.
- Appuie tes recommandations sur les métriques disponibles (Sharpe, alpha Jensen, CVaR, beta, etc.).
- Lorsque tu proposes des allocations, respecte : min 1%, max 25% par fonds, somme = 100%.
- Structure tes réponses avec des sections claires.
- Si tu proposes un tableau d'allocation, utilise le format Markdown.
- Sois concis mais complet — évite les généralités, reste ancré dans les données réelles."""

    # ── Interface de chat ─────────────────────────────────────────
    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = []

    st.markdown('<div class="section-title">💬 Discussion avec l\'IA</div>', unsafe_allow_html=True)

    # Boutons de questions prédéfinies
    st.markdown("**Questions rapides :**")
    q_cols = st.columns(4)
    quick_questions = [
        "Analyse le profil risque/rendement de mon portefeuille actuel",
        "Quels OPCVM surpondérer pour maximiser le Sharpe ?",
        "Propose une allocation optimale pour un profil prudent",
        "Pourquoi CDG Rendement domine en ratio de Sharpe ?",
    ]
    for idx, (col_q, q) in enumerate(zip(q_cols, quick_questions)):
        with col_q:
            if st.button(q[:40] + "…", key=f"quick_{idx}", use_container_width=True):
                st.session_state.ai_messages.append({"role": "user", "content": q})
                st.rerun()

    # Historique de conversation
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.ai_messages:
            with st.chat_message(msg["role"],
                                 avatar="🧑‍💼" if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])

    # Champ de saisie
    user_input = st.chat_input("Posez votre question sur le portefeuille… (ex: Analyse mon Sharpe)")

    # ── Appel API Anthropic ───────────────────────────────────────
    if user_input:
        st.session_state.ai_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("L'IA analyse votre portefeuille…"):
                try:
                    import anthropic
                    client = anthropic.Anthropic()

                    # Fenêtre glissante : 10 derniers messages max
                    history = st.session_state.ai_messages[-10:]
                    # S'assurer que le premier message est de l'utilisateur
                    while history and history[0]["role"] != "user":
                        history = history[1:]

                    response = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=1500,
                        system=SYSTEM_PROMPT,
                        messages=history,
                    )
                    answer = response.content[0].text
                    st.markdown(answer)
                    st.session_state.ai_messages.append({"role": "assistant", "content": answer})

                except ImportError:
                    err = ("⚠️ **Package `anthropic` manquant.** "
                           "Installez-le avec : `pip install anthropic`")
                    st.error(err)
                    st.session_state.ai_messages.append({"role": "assistant", "content": err})
                except Exception as e:
                    err = f"⚠️ **Erreur API :** {str(e)}"
                    st.error(err)
                    st.session_state.ai_messages.append({"role": "assistant", "content": err})

    # ── Actions de gestion du chat ────────────────────────────────
    if st.session_state.ai_messages:
        col_clear, col_export = st.columns([1, 3])
        with col_clear:
            if st.button("🗑️ Effacer la conversation", use_container_width=True):
                st.session_state.ai_messages = []
                st.rerun()
        with col_export:
            # Export de la conversation en texte
            conv_text = "\n\n".join(
                [f"{'Vous' if m['role']=='user' else 'IA'}: {m['content']}"
                 for m in st.session_state.ai_messages]
            )
            st.download_button(
                label="📥 Télécharger la conversation",
                data=conv_text,
                file_name="analyse_opcvm_ia.txt",
                mime="text/plain",
                use_container_width=True,
            )

    # ── Note technique ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style="padding:10px 16px;background:#F0F4F1;border-radius:8px;font-size:0.78rem;
                color:{COLORS['gray']};border-left:4px solid {COLORS['light_green']};">
      <b>Note technique :</b> L'IA reçoit l'intégralité des métriques de votre portefeuille courant
      (poids normalisés, performance, volatilité, Sharpe, alpha Jensen, beta, VaR, CVaR, tracking error,
      information ratio, drawdown). Le modèle utilisé est <b>claude-sonnet-4-20250514</b>.
      La clé API doit être définie via la variable d'environnement <code>ANTHROPIC_API_KEY</code>.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 6 — MARKOWITZ EXACT (allocations sans bornes arbitraires)
# ═══════════════════════════════════════════════════════════════════
with tab6:

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{COLORS['dark_green']},{COLORS['navy']});
                padding:18px 24px;border-radius:10px;margin-bottom:20px;
                border-left:6px solid {COLORS['gold']};">
      <h2 style="color:white;margin:0;font-size:1.3rem">📐 Optimisation Markowitz — Allocations Exactes</h2>
      <p style="color:{COLORS['mint']};margin:4px 0 0;font-size:0.85rem">
        Seule contrainte : somme des poids = 100% et poids ≥ 0 (long only) · Pas de borne maximale par fonds
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Fonctions d'optimisation sans bornes max ──────────────────
    def markowitz_exact(methode, R=R_GLOBAL):
        mu  = R.mean(axis=0) * 252
        cov = np.cov(R.T) * 252
        n   = R.shape[1]

        # Long-only uniquement, pas de borne max
        bounds      = [(0.0, 1.0)] * n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

        def neg_sharpe(w):
            ret = mu @ w
            vol = np.sqrt(w @ cov @ w + 1e-12)
            return -(ret - RF) / vol

        def portfolio_vol(w):
            return w @ cov @ w          # minimise variance

        def neg_cvar(w):
            r = R @ w
            q = np.percentile(r, 1)
            tail = r[r <= q]
            return -tail.mean() if len(tail) > 0 else -q

        obj_map = {
            "Max Sharpe":   neg_sharpe,
            "Min Variance": portfolio_vol,
            "Min CVaR":     neg_cvar,
        }

        rng  = np.random.default_rng(0)
        best = None
        # Plusieurs points de départ pour robustesse
        w0_list = [np.ones(n) / n]
        for _ in range(15):
            w0 = rng.dirichlet(np.ones(n))
            w0_list.append(w0)

        for w0 in w0_list:
            try:
                res = minimize(
                    obj_map[methode], w0, method="SLSQP",
                    bounds=bounds, constraints=constraints,
                    options={"maxiter": 2000, "ftol": 1e-12},
                )
                if res.success and (best is None or res.fun < best.fun):
                    best = res
            except Exception:
                pass

        return best.x if (best is not None and best.success) else np.ones(n) / n

    def stats_exact(w, R=R_GLOBAL):
        mu  = R.mean(axis=0) * 252
        cov = np.cov(R.T) * 252
        r_p = R @ w
        perf    = mu @ w * 100
        vol     = np.sqrt(w @ cov @ w) * 100
        sharpe  = (perf/100 - RF) / (vol/100) if vol > 1e-8 else 0
        r_neg   = r_p[r_p < RF_DAILY]
        dv      = r_neg.std() * np.sqrt(252) if len(r_neg) > 1 else vol / 100
        sortino = (perf/100 - RF) / dv if dv > 1e-8 else 0
        var99   = np.percentile(r_p, 1) * 100
        mask    = r_p <= np.percentile(r_p, 1)
        cvar99  = r_p[mask].mean() * 100 if mask.any() else var99
        return {"perf": perf, "vol": vol, "sharpe": sharpe,
                "sortino": sortino, "var99": var99, "cvar99": cvar99}

    # ── Contrôles ─────────────────────────────────────────────────
    st.markdown(f'<div class="section-title">⚙️ Paramètres d\'optimisation</div>', unsafe_allow_html=True)

    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
    with ctrl1:
        methode_mk = st.selectbox(
            "Méthode",
            ["Max Sharpe", "Min Variance", "Min CVaR"],
            key="mk_methode",
        )
    with ctrl2:
        seuil_affichage = st.slider(
            "Seuil d'affichage (min %)",
            min_value=0.0, max_value=5.0, value=0.5, step=0.1,
            key="mk_seuil",
            help="Masquer les fonds avec une allocation inférieure à ce seuil",
        )
    with ctrl3:
        st.markdown(f"""
        <div style="background:{COLORS['mint']};border-left:4px solid {COLORS['mid_green']};
                    padding:10px 14px;border-radius:6px;font-size:0.83rem;color:{COLORS['dark_green']};margin-top:4px;">
          <b>Long-only sans contrainte de borne max</b> — l'optimiseur peut concentrer 100% sur
          un seul fonds si c'est mathématiquement optimal. Utilisez le seuil d'affichage pour
          filtrer les allocations marginales.
        </div>
        """, unsafe_allow_html=True)

    # ── Calcul ────────────────────────────────────────────────────
    with st.spinner(f"Calcul Markowitz exact — {methode_mk}…"):
        w_mk   = markowitz_exact(methode_mk)
        s_mk   = stats_exact(w_mk)
        s_ref  = stats_exact(w_ref)

    # ── KPIs comparaison ──────────────────────────────────────────
    st.markdown(f'<div class="section-title">📌 Résultats vs Portefeuille Actuel</div>', unsafe_allow_html=True)

    def kpi_delta(label, val, ref, fmt=".2f", unit="%", inverse=False, card_cls=""):
        diff = val - ref
        good = (diff > 0) if not inverse else (diff < 0)
        signe = "+" if diff > 0 else ""
        cls   = "delta-pos" if good else "delta-neg"
        return f"""
        <div class="metric-card {card_cls}">
          <p>{label}</p>
          <h3>{val:{fmt}}{unit}</h3>
          <div class="delta">
            <span class="{cls}">{signe}{diff:{fmt}}{unit} vs actuel</span>
          </div>
        </div>"""

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.markdown(kpi_delta("Performance",  s_mk["perf"],    s_ref["perf"],    ".2f", "%", False, ""),      unsafe_allow_html=True)
    with k2: st.markdown(kpi_delta("Volatilité",   s_mk["vol"],     s_ref["vol"],     ".2f", "%", True,  "red"),   unsafe_allow_html=True)
    with k3: st.markdown(kpi_delta("Sharpe",       s_mk["sharpe"],  s_ref["sharpe"],  ".3f", "",  False, "gold"),  unsafe_allow_html=True)
    with k4: st.markdown(kpi_delta("Sortino",      s_mk["sortino"], s_ref["sortino"], ".3f", "",  False, "gold"),  unsafe_allow_html=True)
    with k5: st.markdown(kpi_delta("CVaR 99%",     s_mk["cvar99"],  s_ref["cvar99"],  ".4f", "%", True,  "navy"),  unsafe_allow_html=True)

    st.markdown("---")

    # ── Tableau interactif des allocations ────────────────────────
    st.markdown(f'<div class="section-title">📋 Allocations Optimales Exactes</div>', unsafe_allow_html=True)

    # Construire le dataframe complet
    rows_mk = []
    for i, nom in enumerate(NOMS):
        m       = META[nom]
        w_opt_i = w_mk[i] * 100
        w_act_i = POIDS_ACTUELS[nom]
        delta_i = w_opt_i - w_act_i
        rows_mk.append({
            "OPCVM":              nom,
            "Poids Actuel (%)":   round(w_act_i, 2),
            "Poids Optimal (%)":  round(w_opt_i, 4),
            "Δ (pp)":             round(delta_i, 4),
            "Performance (%)":    m["perf"],
            "Volatilité (%)":     m["vol"],
            "Sharpe":             m["sharpe"],
            "Alpha Jensen (%)":   m["alpha_j"],
            "Beta":               m["beta"],
            "CVaR 99% (%)":       m["cvar99"],
        })

    df_mk = pd.DataFrame(rows_mk)

    # Filtre seuil
    df_mk_display = df_mk[df_mk["Poids Optimal (%)"] >= seuil_affichage].copy()
    df_mk_display = df_mk_display.sort_values("Poids Optimal (%)", ascending=False)

    # Affichage du nombre de fonds retenus
    n_actifs = len(df_mk_display)
    poids_couverts = df_mk_display["Poids Optimal (%)"].sum()
    c_info1, c_info2, c_info3 = st.columns(3)
    with c_info1:
        st.markdown(f"""<div class="metric-card"><p>Fonds actifs (≥ {seuil_affichage:.1f}%)</p>
        <h3>{n_actifs} / {N}</h3></div>""", unsafe_allow_html=True)
    with c_info2:
        st.markdown(f"""<div class="metric-card gold"><p>Poids cumulé affiché</p>
        <h3>{poids_couverts:.2f}%</h3></div>""", unsafe_allow_html=True)
    with c_info3:
        st.markdown(f"""<div class="metric-card navy"><p>Méthode appliquée</p>
        <h3 style="font-size:1rem">{methode_mk}</h3></div>""", unsafe_allow_html=True)

    # Tableau stylisé avec st.dataframe + coloration
    df_show = df_mk_display.set_index("OPCVM")

    def color_delta(val):
        if val > 0:   return "color: #2C5F2D; font-weight: 600"
        elif val < 0: return "color: #C0392B; font-weight: 600"
        return ""

    def color_poids(val):
        if val >= 20:  return "background-color: #1A3C2E; color: white; font-weight:700"
        elif val >= 10: return "background-color: #2C5F2D; color: white; font-weight:700"
        elif val >= 5:  return "background-color: #97BC62; color: #1A3C2E; font-weight:600"
        elif val >= 1:  return "background-color: #D4EDDA; color: #1A3C2E"
        return "color: #64748B"

    styled = (
        df_show.style
        .applymap(color_poids, subset=["Poids Optimal (%)"])
        .applymap(color_delta, subset=["Δ (pp)"])
        .format({
            "Poids Actuel (%)":  "{:.2f}%",
            "Poids Optimal (%)": "{:.4f}%",
            "Δ (pp)":            "{:+.4f}",
            "Performance (%)":   "{:.2f}%",
            "Volatilité (%)":    "{:.2f}%",
            "Sharpe":            "{:.3f}",
            "Alpha Jensen (%)":  "{:+.2f}%",
            "Beta":              "{:.2f}",
            "CVaR 99% (%)":      "{:.3f}%",
        })
        .set_table_styles([{
            "selector": "th",
            "props": [("background-color", COLORS["dark_green"]),
                      ("color", "white"), ("font-weight", "600"),
                      ("font-size", "12px"), ("padding", "8px 10px")],
        }])
    )

    st.dataframe(styled, use_container_width=True, height=420)

    # Bouton export CSV
    csv_data = df_mk_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Exporter les allocations (CSV)",
        data=csv_data,
        file_name=f"markowitz_exact_{methode_mk.replace(' ','_')}.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # ── Graphiques ────────────────────────────────────────────────
    col_g1, col_g2 = st.columns([1.2, 1.8])

    with col_g1:
        st.markdown(f'<div class="section-title">🥧 Répartition Optimale</div>', unsafe_allow_html=True)
        df_pie = df_mk_display[df_mk_display["Poids Optimal (%)"] > 0]
        fig_mk_pie = go.Figure(go.Pie(
            labels=df_pie["OPCVM"].str[:20],
            values=df_pie["Poids Optimal (%)"].round(4),
            textinfo="label+percent",
            textfont_size=10,
            marker=dict(
                colors=px.colors.qualitative.Set2,
                line=dict(color="white", width=1.5),
            ),
            hole=0.38,
        ))
        fig_mk_pie.update_layout(
            showlegend=False, height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10),
            annotations=[dict(
                text=f"<b>{methode_mk}</b>",
                x=0.5, y=0.5, font_size=11, showarrow=False,
                font=dict(color=COLORS["dark_green"]),
            )],
        )
        st.plotly_chart(fig_mk_pie, use_container_width=True)

    with col_g2:
        st.markdown(f'<div class="section-title">📊 Optimal vs Actuel</div>', unsafe_allow_html=True)
        df_bar = df_mk.sort_values("Poids Optimal (%)", ascending=False)
        noms_b = df_bar["OPCVM"].str[:16].tolist()

        fig_mk_bar = go.Figure()
        fig_mk_bar.add_trace(go.Bar(
            name="Actuel",
            x=noms_b,
            y=df_bar["Poids Actuel (%)"].tolist(),
            marker_color=COLORS["gray"], opacity=0.65,
            text=[f"{v:.1f}%" for v in df_bar["Poids Actuel (%)"]],
            textposition="outside", textfont=dict(size=8),
        ))
        fig_mk_bar.add_trace(go.Bar(
            name="Optimal",
            x=noms_b,
            y=df_bar["Poids Optimal (%)"].round(2).tolist(),
            marker_color=[
                COLORS["mid_green"] if d >= 0 else COLORS["red"]
                for d in df_bar["Δ (pp)"]
            ],
            opacity=0.9,
            text=[f"{v:.2f}%" for v in df_bar["Poids Optimal (%)"]],
            textposition="outside", textfont=dict(size=8),
        ))
        fig_mk_bar.update_layout(
            barmode="group", height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(244,247,251,0.5)",
            xaxis=dict(tickangle=-40, tickfont=dict(size=8)),
            yaxis=dict(title="Poids (%)", gridcolor="#E2E8F0"),
            legend=dict(orientation="h", x=0, y=1.1),
            margin=dict(t=10, b=90, l=40, r=10),
        )
        st.plotly_chart(fig_mk_bar, use_container_width=True)

    # ── Tableau complet (tous fonds, même à 0%) ───────────────────
    with st.expander("🔍 Voir tous les fonds (y compris allocation nulle)", expanded=False):
        df_all = df_mk.sort_values("Poids Optimal (%)", ascending=False).set_index("OPCVM")
        styled_all = (
            df_all.style
            .applymap(color_poids, subset=["Poids Optimal (%)"])
            .applymap(color_delta, subset=["Δ (pp)"])
            .format({
                "Poids Actuel (%)":  "{:.2f}%",
                "Poids Optimal (%)": "{:.4f}%",
                "Δ (pp)":            "{:+.4f}",
                "Performance (%)":   "{:.2f}%",
                "Volatilité (%)":    "{:.2f}%",
                "Sharpe":            "{:.3f}",
                "Alpha Jensen (%)":  "{:+.2f}%",
                "Beta":              "{:.2f}",
                "CVaR 99% (%)":      "{:.3f}%",
            })
        )
        st.dataframe(styled_all, use_container_width=True, height=500)

    # ── Note méthodologique ───────────────────────────────────────
    st.markdown(f"""
    <div style="padding:12px 16px;background:#F0F4F1;border-radius:8px;font-size:0.8rem;
                color:{COLORS['gray']};border-left:4px solid {COLORS['light_green']};margin-top:16px;">
      <b>Méthodologie :</b> Optimisation SLSQP avec 15 points de départ aléatoires pour robustesse globale.
      Seules contraintes : <code>Σ wᵢ = 100%</code> et <code>wᵢ ≥ 0</code> (long-only).
      Les rendements synthétiques (247 obs) sont calibrés sur les métriques réelles de chaque OPCVM.
      <b>Max Sharpe</b> = maximise (μ − Rf) / σ.
      <b>Min Variance</b> = minimise wᵀΣw.
      <b>Min CVaR</b> = minimise la perte espérée au-delà du quantile 1%.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;padding:20px;">
  <p style="color:{COLORS['gray']};font-size:0.75rem;margin:0;">
    <b>OPCVM Portfolio Dashboard v3</b> — Rendements synthétiques cohérents avec les statistiques réelles.<br>
    Rf = 2.25% · 247 jours · 3 méthodes : Min Variance · Max Sharpe · Min CVaR · Optimiseur SLSQP<br>
    🤖 Onglet IA propulsé par Claude claude-sonnet-4-20250514 · Anthropic API
  </p>
  <p style="color:{COLORS['light_green']};font-size:0.8rem;margin:8px 0 0;font-weight:600;letter-spacing:0.05em;">
    © 2026 · Ben said Raydae
  </p>
</div>
""", unsafe_allow_html=True)
