"""
OPCVM Portfolio Dashboard — Streamlit (Version Corrigée)
=================================================
Interface interactive pour :
  · Ajuster les pondérations de chaque OPCVM en temps réel
  · Visualiser performance, risque et VaR instantanément
  · Tracer la frontière efficiente avec position du portefeuille courant
  · Comparer avec les 4 méthodes d'optimisation (Min Var, Max Sharpe, Min CVaR, Score)

Lancement :
    pip install streamlit pandas numpy scipy plotly openpyxl
    streamlit run opcvm_dashboard.py
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

# ── Palette verte (cohérente avec la présentation) ───────────────────────────
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

# CSS personnalisé thème vert
st.markdown(f"""
<style>
    /* Fond principal */
    .stApp {{ background-color: #F5F9F6; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['dark_green']};
    }}
    [data-testid="stSidebar"] * {{ color: white !important; }}
    [data-testid="stSidebar"] .stSlider label {{ color: {COLORS['light_green']} !important; }}
    [data-testid="stSidebar"] .stSelectbox label {{ color: {COLORS['light_green']} !important; }}
    [data-testid="stSidebar"] .stButton button {{
        background-color: {COLORS['mid_green']};
        color: white;
        border: none;
    }}
    [data-testid="stSidebar"] .stButton button:hover {{
        background-color: {COLORS['light_green']};
        color: {COLORS['dark_green']};
    }}

    /* Header */
    .main-header {{
        background: linear-gradient(135deg, {COLORS['dark_green']}, {COLORS['mid_green']});
        padding: 18px 24px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 6px solid {COLORS['light_green']};
    }}
    .main-header h1 {{ color: white; margin: 0; font-size: 1.6rem; }}
    .main-header p  {{ color: {COLORS['mint']}; margin: 4px 0 0; font-size: 0.9rem; }}

    /* Metric cards */
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

    /* Section titles */
    .section-title {{
        background: {COLORS['mid_green']};
        color: white;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
        margin: 16px 0 10px;
    }}

    /* Alerte somme */
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

    /* Tabs */
    .stTabs [data-baseweb="tab"] {{ background: white; border-radius: 6px 6px 0 0; }}
    .stTabs [aria-selected="true"] {{ background: {COLORS['mid_green']}; color: white !important; }}

    div[data-testid="stNumberInput"] label {{ font-size: 0.8rem; color: {COLORS['dark_green']}; font-weight: 600; }}
    
    /* Dataframe styling */
    .stDataFrame {{ background: white; border-radius: 8px; padding: 8px; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DONNÉES OPCVM  (rendements journaliers & métadonnées)
# ══════════════════════════════════════════════════════════════════════════════

# Poids actuels du portefeuille (en %)
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

# Indicateurs individuels calculés (Rf = 2.25%)
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


# ── Génération de rendements synthétiques cohérents avec les stats réelles ───
@st.cache_data
def generer_rendements_synthetiques():
    """
    Simule 247 rendements journaliers par OPCVM cohérents avec
    les statistiques réelles (perf, vol, corrélations estimées).
    """
    np.random.seed(42)
    T = 247

    # Rendements journaliers moyens et volatilités journalières
    mu_j  = np.array([META[n]["perf"] / 100 / 252     for n in NOMS])
    sig_j = np.array([META[n]["vol"]  / 100 / np.sqrt(252) for n in NOMS])

    # Matrice de corrélation approximée par les bêtas
    betas = np.array([META[n]["beta"] for n in NOMS])
    # r_i ≈ α_i + β_i * r_market + ε_i
    r_market = np.random.normal(0.0003, 0.008, T)

    R = np.zeros((T, N))
    for i in range(N):
        alpha_j_daily = mu_j[i] - betas[i] * 0.0003
        idio_vol = max(sig_j[i] * np.sqrt(1 - min(betas[i]**2 * 0.35, 0.9)), 1e-5)
        eps = np.random.normal(0, idio_vol, T)
        R[:, i] = alpha_j_daily + betas[i] * r_market + eps

    # Recaler exactement perf annuelle et vol
    for i in range(N):
        R[:, i] = R[:, i] - R[:, i].mean() + mu_j[i]
        scale    = sig_j[i] / (R[:, i].std() + 1e-12)
        R[:, i]  = (R[:, i] - mu_j[i]) * scale + mu_j[i]

    return R

R_GLOBAL = generer_rendements_synthetiques()


# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS CALCUL PORTEFEUILLE
# ══════════════════════════════════════════════════════════════════════════════

def calcul_stats(w_arr, R=R_GLOBAL):
    """Calcule toutes les statistiques d'un vecteur de poids."""
    r_ptf   = R @ w_arr
    perf    = r_ptf.mean() * 252 * 100
    vol     = r_ptf.std()  * np.sqrt(252) * 100
    sharpe  = (perf/100 - RF) / (vol/100) if vol > 1e-8 else 0
    r_neg   = r_ptf[r_ptf < RF_DAILY]
    dv      = r_neg.std() * np.sqrt(252) if len(r_neg) > 1 else vol/100
    sortino = (perf/100 - RF) / dv if dv > 1e-8 else 0
    var99   = np.percentile(r_ptf, 1, method="linear") * 100
    cvar99  = r_ptf[r_ptf <= np.percentile(r_ptf, 1)].mean() * 100 if len(r_ptf[r_ptf <= np.percentile(r_ptf, 1)]) > 0 else var99
    cum     = np.cumprod(1 + r_ptf)
    roll    = np.maximum.accumulate(cum)
    dd_max  = ((cum - roll) / roll).min() * 100
    return {
        "perf":    perf,
        "vol":     vol,
        "sharpe":  sharpe,
        "sortino": sortino,
        "var99":   var99,
        "cvar99":  cvar99,
        "dd_max":  dd_max,
    }


def optimiser(methode, w_actuel, R=R_GLOBAL):
    """Lance l'optimisation SLSQP selon la méthode choisie."""
    mu  = R.mean(axis=0) * 252
    cov = np.cov(R.T) * 252
    n   = len(w_actuel)
    bounds = [(0.01, 0.25)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    
    # Rendement minimum à 50% du rendement actuel
    ret_min = (mu @ w_actuel) * 0.5
    if methode != "Min Variance":
        constraints.append({"type": "ineq", "fun": lambda w: (mu @ w) - ret_min})

    def cvar_fn(w):
        r = R @ w
        var_thresh = np.percentile(r, 1)
        return -r[r <= var_thresh].mean() if len(r[r <= var_thresh]) > 0 else -np.percentile(r, 1)

    objectives = {
        "Min Variance": lambda w: w @ cov @ w,
        "Max Sharpe":   lambda w: -(mu @ w - RF) / (np.sqrt(w @ cov @ w + 1e-12)),
        "Min CVaR":     cvar_fn,
        "Score Composite": lambda w: -(
            np.array([META[n]["sharpe"]  for n in NOMS]) / 5.0 * 0.30 +
            np.array([max(META[n]["alpha_j"], -10) for n in NOMS]) / 15.0 * 0.30 +
            np.array([META[n]["perf"] / 100 for n in NOMS]) / 0.35 * 0.25 +
            np.array([max(-META[n]["dd"], 0) / 20.0 for n in NOMS]) * 0.15
        ) @ w,
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
    """Calcule la frontière efficiente (120 points)."""
    mu  = R.mean(axis=0) * 252
    cov = np.cov(R.T) * 252
    n   = R.shape[1]

    ret_min = mu.min()
    ret_max = mu.max() * 0.85
    targets = np.linspace(ret_min, ret_max, 100)

    vols, rets, var99s = [], [], []
    constraints_base = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.01, 0.25)] * n
    obj    = lambda w: w @ cov @ w

    for t in targets:
        cst = constraints_base + [{"type": "eq", "fun": lambda w, tt=t: mu @ w - tt}]
        rng = np.random.default_rng(0)
        best = None
        for _ in range(5):
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
            w = best.x
            r_ptf = R @ w
            vols.append(np.sqrt(w @ cov @ w) * 100)
            rets.append(mu @ w * 100)
            var99s.append(np.percentile(r_ptf, 1, method="linear") * 100)

    return np.array(vols), np.array(rets), np.array(var99s)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — SLIDERS DE PONDÉRATION
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown(f"""
<div style="background:{COLORS['mid_green']};padding:12px;border-radius:8px;margin-bottom:12px;
            border-left:4px solid {COLORS['light_green']}">
  <h2 style="color:white;margin:0;font-size:1.1rem">⚖️ Pondérations OPCVM</h2>
  <p style="color:{COLORS['mint']};font-size:0.8rem;margin:4px 0 0">Ajustez les poids (1% – 25%)</p>
</div>
""", unsafe_allow_html=True)

# Initialisation des valeurs par défaut dans session_state
for nom in NOMS:
    if f"slider_{nom}" not in st.session_state:
        st.session_state[f"slider_{nom}"] = float(POIDS_ACTUELS[nom])

# Catégories pour grouper les sliders
CATEGORIES = {
    "🏦 Obligations Long Terme": ["AFG GOV BOND FUND", "ALPHA SECURE FUND", "CAM OBLIBANQUES", "CDG RENDEMENT", "OBLIG CT", "CDG TAWFIR", "EMERGENCE SERENITE"],
    "📈 Fonds Diversifiés":  ["AD BALANCED FUND", "AFG OPTIMAL FUND", "CDG IZDIHAR", "CAPITAL TRUST EQUILIBRE"],
    "💰 Obligations Court Terme":   ["AD SELECT BANK", "ALPHA BANQUES FUND", "AD YIELD FUND"],
}

poids_user = {}
for cat, fonds in CATEGORIES.items():
    with st.sidebar.expander(cat, expanded=True):
        for nom in fonds:
            poids_user[nom] = st.slider(
                nom.replace("FCP ", "").replace("SICAV ", ""),
                min_value=0.0, max_value=25.0,
                value=st.session_state[f"slider_{nom}"],
                step=0.1, key=f"slider_{nom}",
                format="%.1f%%"
            )

# Somme des poids
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

# Boutons sidebar
st.sidebar.markdown("---")
col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    if st.button("🔄 Réinitialiser", use_container_width=True):
        for nom in NOMS:
            st.session_state[f"slider_{nom}"] = float(POIDS_ACTUELS[nom])
        st.rerun()

with col_sb2:
    methode_optim = st.selectbox("Méthode d'optimisation", 
                                  ["Min Variance", "Max Sharpe", "Min CVaR", "Score Composite"],
                                  label_visibility="collapsed")

if st.sidebar.button(f"⚡ Appliquer {methode_optim}", use_container_width=True, type="primary"):
    w_cur = np.array([poids_user[n] / 100 for n in NOMS])
    w_cur = np.clip(w_cur, 0.01, 0.25)
    w_cur /= w_cur.sum()
    with st.spinner(f"Optimisation {methode_optim} en cours..."):
        w_opt = optimiser(methode_optim, w_cur)
    for i, nom in enumerate(NOMS):
        st.session_state[f"slider_{nom}"] = round(w_opt[i] * 100, 1)
    st.rerun()

st.sidebar.markdown(f"""
<div style="margin-top:12px;padding:8px;background:{COLORS['dark_green']};border-radius:6px;font-size:0.75rem;color:{COLORS['gray']};text-align:center;">
  📊 Rf = 2.25% · 247 obs · 14 OPCVM
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CALCULS PRINCIPAUX
# ══════════════════════════════════════════════════════════════════════════════

# Normalisation des poids
w_raw  = np.array([poids_user[n] / 100 for n in NOMS])
w_norm = w_raw / w_raw.sum()

# Stats portefeuille courant
stats_cur = calcul_stats(w_norm)

# Stats portefeuille actuel (référence)
w_ref      = np.array([POIDS_ACTUELS[n] / 100 for n in NOMS])
stats_ref  = calcul_stats(w_ref)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="main-header">
  <h1>📊 OPCVM Portfolio Dashboard</h1>
  <p>Optimisation interactive des poids · Frontière Efficiente · Rf = 2.25% · 14 OPCVM · 247 observations</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ONGLETS PRINCIPAUX
# ══════════════════════════════════════════════════════════════════════════════

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

    # ── KPIs ligne 1 ─────────────────────────────────────────────
    st.markdown('<div class="section-title">📌 Indicateurs du Portefeuille Courant</div>', unsafe_allow_html=True)

    def delta_html(val, ref, unit="%", inverse=False):
        diff = val - ref
        good = diff > 0 if not inverse else diff < 0
        signe = "+" if diff > 0 else ""
        cls = "delta-pos" if good else "delta-neg"
        return f'<span class="{cls}">{signe}{diff:.2f}{unit} vs actuel</span>'

    kpis = [
        ("Performance Annuelle", f"{stats_cur['perf']:.2f}%",  stats_cur["perf"],  stats_ref["perf"],  "%",  False, ""),
        ("Volatilité Annualisée", f"{stats_cur['vol']:.2f}%",   stats_cur["vol"],   stats_ref["vol"],   "%",  True,  "red"),
        ("Sharpe Ratio",       f"{stats_cur['sharpe']:.3f}", stats_cur["sharpe"],stats_ref["sharpe"],"",   False, "gold"),
        ("Sortino Ratio",            f"{stats_cur['sortino']:.3f}",stats_cur["sortino"],stats_ref["sortino"],"", False, "gold"),
        ("VaR 99% (journalière)",       f"{stats_cur['var99']:.4f}%", stats_cur["var99"], stats_ref["var99"],  "%",  True,  "red"),
        ("CVaR 99% (journalière)",      f"{stats_cur['cvar99']:.4f}%",stats_cur["cvar99"],stats_ref["cvar99"], "%",  True,  "red"),
        ("Maximum Drawdown",       f"{stats_cur['dd_max']:.2f}%",stats_cur["dd_max"],stats_ref["dd_max"], "%",  True,  "navy"),
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
            </div>
            """, unsafe_allow_html=True)

    # ── Graphiques ligne 2 ─────────────────────────────────────────
    col_left, col_right = st.columns([1.1, 1.9])

    with col_left:
        st.markdown('<div class="section-title">🥧 Répartition des Poids</div>', unsafe_allow_html=True)

        # Camembert interactif
        poids_pct = w_norm * 100
        colors_pie = px.colors.qualitative.Set3
        
        fig_pie = go.Figure(go.Pie(
            labels=[n.replace("FCP ", "").replace("SICAV ", "") for n in NOMS],
            values=poids_pct,
            textinfo="label+percent",
            textfont_size=10,
            marker=dict(colors=colors_pie, line=dict(color="white", width=1.5)),
            hole=0.35,
        ))
        fig_pie.update_layout(
            showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
            height=380, paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text=f"<b>Total: {total_poids:.0f}%</b>", x=0.5, y=0.5,
                              font_size=14, showarrow=False,
                              font=dict(color=COLORS["dark_green"], weight="bold"))]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-title">📊 Comparaison des Poids</div>', unsafe_allow_html=True)

        noms_short = [n.replace("FCP ", "").replace("SICAV ", "")[:15] for n in NOMS]
        delta_poids = poids_pct - np.array([POIDS_ACTUELS[n] for n in NOMS])

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Poids Actuel", x=noms_short,
            y=[POIDS_ACTUELS[n] for n in NOMS],
            marker_color=COLORS["gray"], opacity=0.7,
            text=[f"{POIDS_ACTUELS[n]:.1f}%" for n in NOMS],
            textposition="outside",
            textfont=dict(size=8)
        ))
        fig_bar.add_trace(go.Bar(
            name="Poids Courant", x=noms_short, y=poids_pct,
            marker_color=[COLORS["mid_green"] if d >= 0 else COLORS["red"] for d in delta_poids],
            opacity=0.85,
            text=[f"{p:.1f}%" for p in poids_pct],
            textposition="outside",
            textfont=dict(size=8)
        ))
        fig_bar.add_hline(y=1, line_dash="dot", line_color=COLORS["light_green"], line_width=1, 
                         annotation_text="min 1%", annotation_position="bottom right")
        fig_bar.add_hline(y=25, line_dash="dot", line_color=COLORS["red"], line_width=1, 
                         annotation_text="max 25%", annotation_position="top right")
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

    # ── Évolution performance cumulée simulation ───────────────────
    st.markdown('<div class="section-title">📈 Performance Cumulée Simulée (247 jours)</div>', unsafe_allow_html=True)

    r_cur = R_GLOBAL @ w_norm
    r_ref = R_GLOBAL @ w_ref
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
        fill="tonexty",
        fillcolor="rgba(44,95,45,0.1)"
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
    st.markdown('<div class="section-title">🎯 Frontière Efficiente de Markowitz</div>', unsafe_allow_html=True)

    with st.spinner("Calcul de la frontière efficiente..."):
        fe_vols, fe_rets, fe_var99s = calculer_frontiere()

    # Points des 4 méthodes optimisées
    optim_points = {}
    with st.spinner("Calcul des 4 portefeuilles optimisés..."):
        for m in ["Min Variance", "Max Sharpe", "Min CVaR", "Score Composite"]:
            w_opt = optimiser(m, w_ref)
            optim_points[m] = calcul_stats(w_opt)

    col_opt = {
        "Min Variance":    COLORS["mid_green"],
        "Max Sharpe":      COLORS["navy"],
        "Min CVaR":        COLORS["red"],
        "Score Composite": COLORS["purple"],
    }

    fig_fe = go.Figure()

    # Frontière efficiente colorée par VaR
    if len(fe_vols) > 0:
        fig_fe.add_trace(go.Scatter(
            x=fe_vols, y=fe_rets,
            mode="markers",
            marker=dict(
                color=fe_var99s, colorscale="RdYlGn_r",
                size=6, opacity=0.8,
                colorbar=dict(title="VaR 99%<br>(%/jour)", thickness=12, len=0.6),
                line=dict(width=0)
            ),
            name="Frontière Efficiente",
            hovertemplate="Volatilité: %{x:.2f}%<br>Performance: %{y:.2f}%<br>VaR: %{marker.color:.4f}%<extra></extra>",
        ))

    # OPCVM individuels
    for nom in NOMS:
        m = META[nom]
        aj = m["alpha_j"]
        fig_fe.add_trace(go.Scatter(
            x=[m["vol"]], y=[m["perf"]],
            mode="markers+text",
            marker=dict(size=10, color=COLORS["light_green"] if aj > 0 else COLORS["red"],
                        symbol="circle", opacity=0.8,
                        line=dict(color="white", width=1)),
            text=[nom.replace("FCP ", "").replace("SICAV "", "")[:12]],
            textposition="top center", 
            textfont=dict(size=8, color=COLORS["dark_green"]),
            name=nom[:20], 
            showlegend=False,
            hovertemplate=f"<b>{nom}</b><br>Vol: {m['vol']:.2f}%<br>Perf: {m['perf']:.2f}%<br>Sharpe: {m['sharpe']:.3f}<br>α Jensen: {aj:+.2f}%<extra></extra>",
        ))

    # Portefeuille actuel (référence)
    fig_fe.add_trace(go.Scatter(
        x=[stats_ref["vol"]], y=[stats_ref["perf"]],
        mode="markers+text",
        marker=dict(size=18, color=COLORS["gray"], symbol="square",
                    line=dict(color="white", width=2)),
        text=["Actuel"], textposition="top center",
        textfont=dict(size=11, color=COLORS["gray"], family="Arial Black"),
        name="Portefeuille Actuel",
        hovertemplate=f"<b>Portefeuille Actuel</b><br>Vol: {stats_ref['vol']:.2f}%<br>Perf: {stats_ref['perf']:.2f}%<br>Sharpe: {stats_ref['sharpe']:.3f}<extra></extra>",
    ))

    # Portefeuille courant (user)
    fig_fe.add_trace(go.Scatter(
        x=[stats_cur["vol"]], y=[stats_cur["perf"]],
        mode="markers+text",
        marker=dict(size=22, color=COLORS["light_green"], symbol="star",
                    line=dict(color=COLORS["dark_green"], width=2)),
        text=["Votre Portefeuille"], textposition="top center",
        textfont=dict(size=11, color=COLORS["mid_green"], family="Arial Black"),
        name="Portefeuille Courant",
        hovertemplate=f"<b>Portefeuille Courant</b><br>Vol: {stats_cur['vol']:.2f}%<br>Perf: {stats_cur['perf']:.2f}%<br>Sharpe: {stats_cur['sharpe']:.3f}<extra></extra>",
    ))

    # 4 méthodes d'optimisation
    markers_m = {"Min Variance": "square", "Max Sharpe": "triangle-up", "Min CVaR": "diamond", "Score Composite": "pentagon"}
    for m_name, s in optim_points.items():
        fig_fe.add_trace(go.Scatter(
            x=[s["vol"]], y=[s["perf"]],
            mode="markers+text",
            marker=dict(size=16, color=col_opt[m_name], symbol=markers_m[m_name],
                        line=dict(color="white", width=1.5)),
            text=[m_name[:12]], textposition="bottom center",
            textfont=dict(size=9, color=col_opt[m_name], weight="bold"),
            name=m_name,
            hovertemplate=f"<b>{m_name}</b><br>Vol: {s['vol']:.2f}%<br>Perf: {s['perf']:.2f}%<br>Sharpe: {s['sharpe']:.3f}<br>VaR: {s['var99']:.4f}%<extra></extra>",
        ))

    # Ligne du marché (CML)
    if len(fe_vols) > 0:
        best_sh_idx = np.argmax([(fe_rets[i]/100 - RF) / (fe_vols[i]/100 + 1e-8) for i in range(len(fe_vols))])
        slope_cml   = (fe_rets[best_sh_idx]/100 - RF) / (fe_vols[best_sh_idx]/100 + 1e-8)
        vol_range   = np.array([0, fe_vols.max() * 1.1])
        cml_rets    = (RF + slope_cml * vol_range / 100) * 100
        fig_fe.add_trace(go.Scatter(
            x=vol_range, y=cml_rets, mode="lines", name=f"CML (Rf={RF*100:.2f}%)",
            line=dict(color=COLORS["gold"], dash="dot", width=2), opacity=0.8,
        ))
        fig_fe.add_annotation(
            x=0, y=RF * 100, text=f"Rf = {RF*100:.2f}%",
            showarrow=False, font=dict(size=10, color=COLORS["gold"]),
            xanchor="left", yanchor="bottom"
        )

    fig_fe.update_layout(
        height=600,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(244,247,251,0.7)",
        xaxis=dict(title="Volatilité Annualisée (%)", gridcolor="#E2E8F0", zeroline=False),
        yaxis=dict(title="Performance Annualisée (%)", gridcolor="#E2E8F0", zeroline=False),
        legend=dict(orientation="v", x=1.02, y=1, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor=COLORS["light_green"], borderwidth=1),
        margin=dict(t=20, b=40, l=50, r=180),
        font=dict(family="Calibri"),
        hovermode="closest"
    )
    st.plotly_chart(fig_fe, use_container_width=True)

    # Résumé tableau 4 méthodes
    st.markdown('<div class="section-title">📊 Résumé des 4 Méthodes d\'Optimisation</div>', unsafe_allow_html=True)
    rows_m = []
    for m_name, s in optim_points.items():
        rows_m.append({
            "Méthode": m_name,
            "VaR 99% (j)": f"{s['var99']:.4f}%",
            "CVaR 99% (j)": f"{s['cvar99']:.4f}%",
            "Performance (%)": f"{s['perf']:.2f}%",
            "Sharpe": f"{s['sharpe']:.3f}",
            "Sortino": f"{s['sortino']:.3f}",
            "Volatilité (%)": f"{s['vol']:.2f}%",
            "Drawdown Max (%)": f"{s['dd_max']:.2f}%",
        })
    df_m = pd.DataFrame(rows_m)
    st.dataframe(df_m.set_index("Méthode"), use_container_width=True, height=250)


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — ANALYSE DÉTAILLÉE PAR OPCVM
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">📋 Indicateurs Individuels par OPCVM</div>', unsafe_allow_html=True)

    rows_d = []
    for i, nom in enumerate(NOMS):
        m = META[nom]
        wp = w_norm[i] * 100
        wa = POIDS_ACTUELS[nom]
        rows_d.append({
            "OPCVM": nom.replace("FCP ", "").replace("SICAV ", ""),
            "Poids Actuel (%)": f"{wa:.1f}%",
            "Poids Courant (%)": f"{wp:.1f}%",
            "Δ (pp)": f"{wp-wa:+.1f}",
            "Performance (%)": f"{m['perf']:.2f}%",
            "Beta (β)": f"{m['beta']:.2f}",
            "Alpha Jensen (%)": f"{m['alpha_j']:+.2f}%",
            "Sharpe": f"{m['sharpe']:.3f}",
            "Sortino": f"{m['sortino']:.3f}",
            "VaR 99% (%)": f"{m['var99']:.3f}%",
            "CVaR 99% (%)": f"{m['cvar99']:.3f}%",
            "Tracking Error (%)": f"{m['te']:.3f}%",
            "Information Ratio": f"{m['ir']:.3f}",
            "Drawdown Max (%)": f"{m['dd']:.2f}%",
        })

    df_d = pd.DataFrame(rows_d).set_index("OPCVM")
    st.dataframe(df_d, use_container_width=True, height=500)

    # Scatter : α Jensen vs Sharpe (bubble = poids courant)
    st.markdown('<div class="section-title">🔵 Alpha Jensen vs Sharpe Ratio (Taille = Poids Courant)</div>', unsafe_allow_html=True)

    fig_sc = go.Figure()
    for i, nom in enumerate(NOMS):
        m   = META[nom]
        wp  = w_norm[i] * 100
        col = COLORS["mid_green"] if m["alpha_j"] > 0 else COLORS["red"]
        fig_sc.add_trace(go.Scatter(
            x=[m["alpha_j"]], y=[m["sharpe"]],
            mode="markers+text",
            marker=dict(size=max(wp / 2, 8), color=col, opacity=0.7,
                        line=dict(color="white", width=1)),
            text=[nom.replace("FCP ", "").replace("SICAV ", "")[:14]],
            textposition="top center", 
            textfont=dict(size=9, color=COLORS["dark_green"]),
            name=nom[:20], 
            showlegend=False,
            hovertemplate=f"<b>{nom}</b><br>α Jensen: {m['alpha_j']:+.2f}%<br>Sharpe: {m['sharpe']:.3f}<br>Poids: {wp:.1f}%<extra></extra>",
        ))
    fig_sc.add_vline(x=0, line_dash="dot", line_color=COLORS["red"], line_width=1.5, 
                     annotation_text="α = 0", annotation_position="top")
    fig_sc.add_hline(y=2, line_dash="dot", line_color=COLORS["navy"], line_width=1.5, 
                     annotation_text="Sharpe = 2", annotation_position="right")
    fig_sc.update_layout(
        height=450, 
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(244,247,251,0.7)",
        xaxis=dict(title="Alpha Jensen (%)", gridcolor="#E2E8F0", zeroline=True),
        yaxis=dict(title="Sharpe Ratio", gridcolor="#E2E8F0", zeroline=True),
        margin=dict(t=30, b=40, l=50, r=80),
    )
    st.plotly_chart(fig_sc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — COMPARAISON MÉTHODES (radar + heatmap poids)
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">⚖️ Comparaison Visuelle des 4 Méthodes d\'Optimisation</div>', unsafe_allow_html=True)

    with st.spinner("Calcul des poids optimisés..."):
        w_optimises = {}
        for m in ["Min Variance", "Max Sharpe", "Min CVaR", "Score Composite"]:
            w_optimises[m] = optimiser(m, w_ref)
            # Mettre à jour les stats pour ces poids
            if m not in optim_points:
                optim_points[m] = calcul_stats(w_optimises[m])

    col_r, col_h = st.columns([1, 1.6])

    with col_r:
        # Radar chart amélioré
        cats = ["VaR 99%", "CVaR 99%", "Performance", "Sharpe Ratio", "Sortino Ratio"]
        # Inverser VaR et CVaR pour meilleure lisibilité (plus bas = mieux)
        
        all_ptfs = {"Portefeuille Actuel": stats_ref, **optim_points}
        col_radar = {
            "Portefeuille Actuel": COLORS["gray"],
            "Min Variance": COLORS["mid_green"],
            "Max Sharpe": COLORS["navy"],
            "Min CVaR": COLORS["red"],
            "Score Composite": COLORS["purple"],
        }

        # Normalisation des valeurs
        radar_vals = {}
        for nm, s in all_ptfs.items():
            # Inverser VaR et CVaR car plus bas est mieux
            radar_vals[nm] = [
                -s["var99"],  # inversé
                -s["cvar99"], # inversé
                s["perf"],
                s["sharpe"],
                s["sortino"]
            ]
        
        # Normalisation entre 0 et 1
        radar_norm = {k: [] for k in all_ptfs}
        for i in range(len(cats)):
            vals = [radar_vals[k][i] for k in all_ptfs]
            mn, mx = min(vals), max(vals)
            if mx - mn < 1e-10:
                mx = mn + 1
            for k in all_ptfs:
                radar_norm[k].append((radar_vals[k][i] - mn) / (mx - mn))

        fig_radar = go.Figure()
        for nm in all_ptfs:
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_norm[nm],
                theta=cats,
                fill="toself",
                name=nm,
                line=dict(color=col_radar[nm], width=2),
                opacity=0.7
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickvals=[0, 0.5, 1]),
                angularaxis=dict(tickfont=dict(size=9))
            ),
            showlegend=True,
            height=400,
            legend=dict(font=dict(size=8), x=1.05, y=1),
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=40, l=40, r=80),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_h:
        # Heatmap des poids
        noms_short = [n.replace("FCP ", "").replace("SICAV ", "")[:14] for n in NOMS]
        methodes_h = ["Actuel"] + list(w_optimises.keys())
        w_matrix = np.vstack([
            w_ref * 100,
            *[w_optimises[m] * 100 for m in w_optimises]
        ])

        fig_hm = go.Figure(go.Heatmap(
            z=w_matrix,
            x=noms_short,
            y=methodes_h,
            colorscale="Greens",
            text=np.round(w_matrix, 1),
            texttemplate="%{text}%",
            textfont=dict(size=8, color="black"),
            hovertemplate="<b>%{y}</b> — %{x}<br>Poids: %{z:.1f}%<extra></extra>",
            colorbar=dict(title="Poids (%)", thickness=15, len=0.8),
            zmin=0,
            zmax=25,
        ))
        fig_hm.update_layout(
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
            yaxis=dict(tickfont=dict(size=9)),
            margin=dict(t=30, b=80, l=100, r=20),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    # VaR bar chart comparatif
    st.markdown('<div class="section-title">📉 Comparaison des Mesures de Risque</div>', unsafe_allow_html=True)

    fig_var = go.Figure()
    noms_ptf = ["Actuel"] + list(optim_points.keys())
    vars_ptf = [stats_ref["var99"]] + [optim_points[m]["var99"] for m in optim_points]
    cvars_ptf = [stats_ref["cvar99"]] + [optim_points[m]["cvar99"] for m in optim_points]
    cols_bar2 = [COLORS["gray"], COLORS["mid_green"], COLORS["navy"], COLORS["red"], COLORS["purple"]]

    fig_var.add_trace(go.Bar(
        name="VaR 99%", 
        x=noms_ptf, 
        y=vars_ptf,
        marker_color=cols_bar2, 
        opacity=0.85,
        text=[f"{v:.4f}%" for v in vars_ptf], 
        textposition="inside",
        textfont=dict(color="white", size=10),
    ))
    fig_var.add_trace(go.Bar(
        name="CVaR 99%", 
        x=noms_ptf, 
        y=cvars_ptf,
        marker_color=cols_bar2, 
        opacity=0.45,
        text=[f"{v:.4f}%" for v in cvars_ptf], 
        textposition="inside",
        textfont=dict(color="white", size=9),
    ))
    fig_var.update_layout(
        barmode="group", 
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(244,247,251,0.7)",
        yaxis=dict(title="VaR / CVaR (% / jour)", gridcolor="#E2E8F0"),
        legend=dict(orientation="h", x=0, y=1.12),
        margin=dict(t=40, b=30, l=50, r=30),
    )
    st.plotly_chart(fig_var, use_container_width=True)

    # Sharpe & Sortino comparison
    st.markdown('<div class="section-title">⭐ Comparaison des Ratios de Performance</div>', unsafe_allow_html=True)
    
    fig_sh = go.Figure()
    sharpes = [stats_ref["sharpe"]] + [optim_points[m]["sharpe"] for m in optim_points]
    sortinos = [stats_ref["sortino"]] + [optim_points[m]["sortino"] for m in optim_points]
    
    fig_sh.add_trace(go.Bar(
        name="Sharpe Ratio", 
        x=noms_ptf, 
        y=sharpes, 
        marker_color=cols_bar2, 
        opacity=0.85,
        text=[f"{v:.3f}" for v in sharpes], 
        textposition="outside"
    ))
    fig_sh.add_trace(go.Bar(
        name="Sortino Ratio", 
        x=noms_ptf, 
        y=sortinos, 
        marker_color=cols_bar2, 
        opacity=0.45,
        text=[f"{v:.3f}" for v in sortinos], 
        textposition="outside"
    ))
    fig_sh.update_layout(
        barmode="group", 
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(244,247,251,0.7)",
        yaxis=dict(title="Ratio", gridcolor="#E2E8F0", range=[0, max(sharpes + sortinos) * 1.1]),
        legend=dict(orientation="h", x=0, y=1.12),
        margin=dict(t=40, b=30, l=50, r=30),
    )
    st.plotly_chart(fig_sh, use_container_width=True)
    
    # Performance vs Volatilité des méthodes
    st.markdown('<div class="section-title">📊 Performance vs Volatilité par Méthode</div>', unsafe_allow_html=True)
    
    perf_methods = [stats_ref["perf"]] + [optim_points[m]["perf"] for m in optim_points]
    vol_methods = [stats_ref["vol"]] + [optim_points[m]["vol"] for m in optim_points]
    
    fig_pv = go.Figure()
    fig_pv.add_trace(go.Scatter(
        x=vol_methods,
        y=perf_methods,
        mode="markers+text",
        marker=dict(size=25, color=cols_bar2, line=dict(color="white", width=2)),
        text=noms_ptf,
        textposition="middle center",
        textfont=dict(size=9, color="white", weight="bold"),
        hovertemplate="<b>%{text}</b><br>Volatilité: %{x:.2f}%<br>Performance: %{y:.2f}%<extra></extra>"
    ))
    
    # Ajouter une ligne de tendance
    z = np.polyfit(vol_methods, perf_methods, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(vol_methods), max(vol_methods), 100)
    fig_pv.add_trace(go.Scatter(
        x=x_trend,
        y=p(x_trend),
        mode="lines",
        name="Tendance",
        line=dict(color=COLORS["gold"], dash="dash", width=1.5),
        opacity=0.6
    ))
    
    fig_pv.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(244,247,251,0.7)",
        xaxis=dict(title="Volatilité Annualisée (%)", gridcolor="#E2E8F0"),
        yaxis=dict(title="Performance Annualisée (%)", gridcolor="#E2E8F0"),
        margin=dict(t=20, b=30, l=50, r=30),
        hovermode="closest"
    )
    st.plotly_chart(fig_pv, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {COLORS['gray']}; font-size: 0.75rem; padding: 20px;">
  <b>OPCVM Portfolio Dashboard</b> — Données basées sur des rendements synthétiques cohérents avec les statistiques réelles.<br>
  Rf = 2.25% · Période: 247 jours · Méthodes d'optimisation: SLSQP
</div>
""", unsafe_allow_html=True)
