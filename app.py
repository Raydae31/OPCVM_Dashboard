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
    [data-testid="stSidebar"] {{
        background-color: {COLORS['dark_green']};
    }}
    /* Tous les textes sidebar en blanc */
    [data-testid="stSidebar"] * {{ color: white !important; }}
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {{ color: white !important; }}

    /* Sliders */
    [data-testid="stSidebar"] .stSlider label {{ color: {COLORS['light_green']} !important; }}
    [data-testid="stSidebar"] [data-testid="stTickBarMin"],
    [data-testid="stSidebar"] [data-testid="stTickBarMax"] {{ color: {COLORS['mint']} !important; }}

    /* Selectbox */
    [data-testid="stSidebar"] .stSelectbox label {{ color: {COLORS['light_green']} !important; }}
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] {{
        background-color: {COLORS['mid_green']} !important;
        border: 1px solid {COLORS['light_green']} !important;
    }}

    /* BOUTONS — règles très explicites pour forcer le rendu */
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
    /* Bouton primary (Appliquer) */
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

    /* Expander dans sidebar */
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
    # Valeurs réelles issues du classement multi-critères
    # Perf = performance annualisée réelle | Beta OLS | Alpha Jensen corrigé | Sharpe ajusté Rf=2.25%
    # VaR99 et CVaR99 = valeurs journalières historiques individuelles
    "AFG GOV BOND FUND":       {"perf":  6.11, "vol": 2.36, "sharpe": 1.63, "sortino": 2.31, "alpha_j":  0.69, "beta": 1.23, "te": 1.50, "ir":  0.82, "dd": -2.71, "var99": -0.384, "cvar99": -0.412},
    "AD BALANCED FUND":        {"perf":  9.58, "vol":32.49, "sharpe": 1.93, "sortino": 2.88, "alpha_j":-49.55, "beta": 1.35, "te":10.52, "ir": -0.89, "dd": -8.87, "var99": -1.906, "cvar99": -2.145},
    "AFG OPTIMAL FUND":        {"perf": 12.22, "vol": 8.79, "sharpe": 1.23, "sortino": 1.46, "alpha_j": -3.29, "beta": 1.54, "te": 3.52, "ir":  0.49, "dd": -7.75, "var99": -1.573, "cvar99": -1.831},
    "CDG IZDIHAR":             {"perf":  8.98, "vol":24.76, "sharpe": 2.34, "sortino": 3.97, "alpha_j":-15.01, "beta": 2.85, "te":22.95, "ir":  1.13, "dd": -6.71, "var99": -1.293, "cvar99": -1.524},
    "AD SELECT BANK":          {"perf":  3.61, "vol": 0.55, "sharpe": 2.44, "sortino": 3.23, "alpha_j":  0.90, "beta": 1.19, "te": 0.32, "ir":  2.99, "dd": -0.19, "var99": -0.081, "cvar99": -0.094},
    "ALPHA BANQUES FUND":      {"perf":  3.32, "vol": 0.50, "sharpe": 2.16, "sortino": 2.90, "alpha_j":  0.66, "beta": 1.06, "te": 0.28, "ir":  2.41, "dd": -0.18, "var99": -0.078, "cvar99": -0.091},
    "ALPHA SECURE FUND":       {"perf":  5.21, "vol": 1.71, "sharpe": 1.73, "sortino": 2.18, "alpha_j":  0.52, "beta": 0.94, "te": 0.95, "ir":  0.38, "dd": -1.59, "var99": -0.274, "cvar99": -0.315},
    "CAM OBLIBANQUES":         {"perf":  3.38, "vol": 0.55, "sharpe": 2.04, "sortino": 2.84, "alpha_j":  0.60, "beta": 1.15, "te": 0.25, "ir":  2.58, "dd": -0.17, "var99": -0.074, "cvar99": -0.088},
    "CDG RENDEMENT":           {"perf": 25.39, "vol": 5.05, "sharpe": 4.58, "sortino": 6.49, "alpha_j":  6.76, "beta": 0.66, "te": 3.48, "ir": -0.38, "dd": -1.69, "var99": -0.283, "cvar99": -0.321},
    "OBLIG CT":                {"perf":  3.44, "vol": 0.55, "sharpe": 2.18, "sortino": 2.69, "alpha_j":  0.66, "beta": 1.04, "te": 0.22, "ir":  3.02, "dd": -0.19, "var99": -0.096, "cvar99": -0.112},
    "CDG TAWFIR":              {"perf":  5.34, "vol": 1.91, "sharpe": 1.62, "sortino": 2.08, "alpha_j":  0.52, "beta": 0.99, "te": 1.16, "ir":  0.42, "dd": -1.69, "var99": -0.344, "cvar99": -0.388},
    "EMERGENCE SERENITE":      {"perf": 23.04, "vol": 4.83, "sharpe": 4.31, "sortino": 5.79, "alpha_j": -1.78, "beta": 1.10, "te": 2.34, "ir":  0.12, "dd": -1.66, "var99": -0.616, "cvar99": -0.684},
    "CAPITAL TRUST EQUILIBRE": {"perf": 12.26, "vol": 8.22, "sharpe": 1.22, "sortino": 1.45, "alpha_j": -1.79, "beta": 1.30, "te": 4.27, "ir":  0.23, "dd": -7.63, "var99": -1.710, "cvar99": -1.965},
    "AD YIELD FUND":           {"perf":  3.31, "vol": 0.59, "sharpe": 1.81, "sortino": 2.45, "alpha_j":  0.59, "beta": 1.21, "te": 0.36, "ir":  1.83, "dd": -0.21, "var99": -0.089, "cvar99": -0.103},
}

NOMS     = list(POIDS_ACTUELS.keys())
N        = len(NOMS)
RF       = 0.0225
RF_DAILY = RF / 252

# 3 méthodes seulement (Score Composite supprimé)
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
    """
    Calcule les statistiques du portefeuille.

    Performance, volatilité, Sharpe, Sortino, drawdown → via rendements synthétiques
    (trajectoire journalière cohérente avec les vrais paramètres).

    VaR 99% et CVaR 99% → agrégation pondérée des VaR/CVaR individuelles réelles
    avec facteur de diversification calibré (0.7465) pour que le portefeuille
    de référence donne exactement VaR = -0.375% (valeur historique réelle).
    """
    r_ptf   = R @ w_arr
    perf    = r_ptf.mean() * 252 * 100
    vol     = r_ptf.std()  * np.sqrt(252) * 100
    sharpe  = (perf/100 - RF) / (vol/100) if vol > 1e-8 else 0
    r_neg   = r_ptf[r_ptf < RF_DAILY]
    dv      = r_neg.std() * np.sqrt(252) if len(r_neg) > 1 else vol/100
    sortino = (perf/100 - RF) / dv if dv > 1e-8 else 0

    # ── VaR & CVaR : agrégation pondérée des valeurs individuelles réelles ──
    # Facteur de diversification calibré : VaR_ptf_réelle / VaR_pondérée_brute
    # = -0.3570% / -0.5023% = 0.7107 (portefeuille de référence historique)
    FACTEUR_DIV = 0.7107
    var_indiv  = np.array([META[nom]["var99"]  for nom in NOMS])
    cvar_indiv = np.array([META[nom]["cvar99"] for nom in NOMS])
    var99  = (w_arr @ var_indiv)  * FACTEUR_DIV
    cvar99 = (w_arr @ cvar_indiv) * FACTEUR_DIV

    cum     = np.cumprod(1 + r_ptf)
    roll    = np.maximum.accumulate(cum)
    dd_max  = ((cum - roll) / roll).min() * 100

    return {"perf": perf, "vol": vol, "sharpe": sharpe, "sortino": sortino,
            "var99": var99, "cvar99": cvar99, "dd_max": dd_max}


def calculer_bornes_dynamiques(mu_ann, vol_ann, w_actuel):
    """
    Calcule des bornes min/max réalistes par OPCVM selon leur profil.

    Logique :
      - Borne max = f(score risque/rendement) : les OPCVM à fort Sharpe et
        faible volatilité peuvent recevoir plus de poids.
      - Borne min = fraction du poids actuel : on garde une présence minimale
        sur chaque fonds déjà en portefeuille (diversification).
      - Plafond absolu : 35% (évite la concentration excessive).
      - Plancher absolu : 0.5% (présence symbolique minimale).

    Paramètres
    ----------
    mu_ann  : rendements annualisés (array n)
    vol_ann : volatilités annualisées (array n)
    w_actuel: poids actuels normalisés (array n)
    """
    n = len(mu_ann)

    # Score Sharpe simplifié par OPCVM (rendement / volatilité)
    sharpe_ind = np.array([META[nom]["sharpe"] for nom in NOMS])
    vol_ind    = np.array([META[nom]["vol"]    for nom in NOMS])

    # --- Borne MAXIMALE ---
    # Base : proportionnelle au score Sharpe de chaque OPCVM
    # Plus le Sharpe est élevé, plus on peut lui allouer
    sharpe_pos = np.clip(sharpe_ind, 0.1, None)
    poids_sharpe = sharpe_pos / sharpe_pos.sum()  # normalise entre 0 et 1

    # Score de volatilité inversée : OPCVM peu volatils peuvent peser plus
    vol_inv = 1.0 / (vol_ind + 0.1)
    poids_vol = vol_inv / vol_inv.sum()

    # Score composite : 60% Sharpe + 40% faible vol
    score = 0.60 * poids_sharpe + 0.40 * poids_vol

    # Max = score * facteur_amplification, plafonné à 35%
    facteur = 3.5  # score moyen ~1/n → max moyen ~3.5/n ≈ 25% pour 14 OPCVM
    max_bounds = np.clip(score * facteur, 0.05, 0.35)

    # --- Borne MINIMALE ---
    # Présence minimale = 30% du poids actuel, plancher à 0.5%
    min_bounds = np.clip(w_actuel * 0.30, 0.005, 0.10)

    # Cohérence : si min > max, on abaisse min
    for i in range(n):
        if min_bounds[i] >= max_bounds[i]:
            min_bounds[i] = max(0.005, max_bounds[i] * 0.5)

    return list(zip(min_bounds, max_bounds))


def optimiser(methode, w_actuel, R=R_GLOBAL):
    """
    Optimisation avec bornes dynamiques par OPCVM.

    Les bornes sont calculées selon le profil risque/rendement de chaque fonds :
      - OPCVM à fort Sharpe et faible volatilité → borne max plus élevée
      - OPCVM à faible Sharpe et forte volatilité → borne max plus basse
      - Présence minimale maintenue sur chaque fonds (diversification)
      - Somme = 100% obligatoire
    """
    mu  = R.mean(axis=0) * 252
    cov = np.cov(R.T) * 252
    n   = len(w_actuel)

    # Bornes dynamiques basées sur le profil de chaque OPCVM
    bounds = calculer_bornes_dynamiques(mu, np.sqrt(np.diag(cov)), w_actuel)

    # Contrainte : somme = 100%
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    # Rendement min = 80% du rendement actuel (plus conservateur qu'avant)
    ret_min = (mu @ w_actuel) * 0.80
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

    # Point de départ 1 : poids actuels
    starts = [w_actuel.copy()]

    # Points de départ 2-6 : poids proportionnels au score Sharpe + bruit
    sharpe_ind = np.array([META[nom]["sharpe"] for nom in NOMS])
    sharpe_pos = np.clip(sharpe_ind, 0.1, None)
    w_sharpe   = sharpe_pos / sharpe_pos.sum()

    for alpha in [1.0, 0.7, 0.5, 0.3]:
        w0 = alpha * w_sharpe + (1 - alpha) * rng.dirichlet(np.ones(n))
        # Respecter les bornes
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        w0 = np.clip(w0, lo, hi)
        if w0.sum() > 1e-8:
            w0 /= w0.sum()
        starts.append(w0)

    # Points de départ 7-13 : Dirichlet aléatoires clippés
    for _ in range(7):
        w0 = rng.dirichlet(np.ones(n))
        w0 = np.clip(w0, lo, hi)
        if w0.sum() > 1e-8:
            w0 /= w0.sum()
        starts.append(w0)

    for w0 in starts:
        try:
            res = minimize(objectives[methode], w0, method="SLSQP",
                           bounds=bounds, constraints=constraints,
                           options={"maxiter": 2000, "ftol": 1e-12})
            if res.success and (best is None or res.fun < best.fun):
                best = res
        except Exception:
            pass

    if best is not None and best.success:
        w_opt = best.x
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        w_opt = np.clip(w_opt, lo, hi)
        w_opt /= w_opt.sum()
        return w_opt
    return w_actuel


@st.cache_data
def calculer_frontiere(R=R_GLOBAL):
    """
    Frontière efficiente enrichie — 200 points.
    Retourne vol, ret, sharpe, var99, cvar99 pour chaque point.
    """
    mu  = R.mean(axis=0) * 252
    cov = np.cov(R.T) * 252
    n   = R.shape[1]

    ret_min = mu.min()
    ret_max = mu.max() * 0.85
    targets = np.linspace(ret_min, ret_max, 200)

    vols, rets, sharpes, var99s, cvar99s = [], [], [], [], []
    constraints_base = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    # Bornes cohérentes avec l'optimiseur : long only, max 35% par OPCVM
    bounds = [(0.0, 0.35)] * n
    obj    = lambda w: w @ cov @ w

    for t in targets:
        cst = constraints_base + [{"type": "eq", "fun": lambda w, tt=t: mu @ w - tt}]
        rng = np.random.default_rng(0)
        best = None
        for _ in range(8):
            w0 = rng.dirichlet(np.ones(n))   # point de depart libre sur le simplex
            try:
                res = minimize(obj, w0, method="SLSQP", bounds=bounds,
                               constraints=cst, options={"maxiter": 800, "ftol": 1e-10})
                if res.success and (best is None or res.fun < best.fun):
                    best = res
            except Exception:
                pass
        if best is not None and best.success:
            w    = best.x
            vol_p = np.sqrt(w @ cov @ w) * 100
            ret_p = mu @ w * 100
            sh_p  = (ret_p/100 - RF) / (vol_p/100) if vol_p > 1e-8 else 0
            # VaR/CVaR agrégées pondérées avec facteur de diversification
            FACTEUR_DIV = 0.7465
            var_indiv  = np.array([META[nom]["var99"]  for nom in NOMS])
            cvar_indiv = np.array([META[nom]["cvar99"] for nom in NOMS])
            v99  = (w @ var_indiv)  * FACTEUR_DIV
            cv99 = (w @ cvar_indiv) * FACTEUR_DIV
            vols.append(vol_p)
            rets.append(ret_p)
            sharpes.append(sh_p)
            var99s.append(v99)
            cvar99s.append(cv99)

    return (np.array(vols), np.array(rets),
            np.array(sharpes), np.array(var99s), np.array(cvar99s))


# ══════════════════════════════════════════════════════════════════════════════
# GESTION SESSION STATE — poids des sliders
# ══════════════════════════════════════════════════════════════════════════════
# Stratégie : on stocke les poids cibles dans "poids_cible_{nom}".
# Au début de chaque cycle, on écrit ces valeurs directement dans
# "slider_{nom}" (clé du widget) AVANT que les sliders soient créés.
# Streamlit accepte cela car les widgets n'existent pas encore dans ce cycle.

# Initialiser les poids cibles au premier lancement
for nom in NOMS:
    if f"poids_cible_{nom}" not in st.session_state:
        st.session_state[f"poids_cible_{nom}"] = float(POIDS_ACTUELS[nom])

# Injecter les valeurs cibles dans les clés slider AVANT création des widgets
for nom in NOMS:
    st.session_state[f"slider_{nom}"] = st.session_state[f"poids_cible_{nom}"]


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown(f"""
<div style="background:{COLORS['mid_green']};padding:12px;border-radius:8px;margin-bottom:12px;
            border-left:4px solid {COLORS['light_green']}">
  <h2 style="color:white;margin:0;font-size:1.1rem">⚖️ Pondérations OPCVM</h2>
  <p style="color:{COLORS['mint']};font-size:0.8rem;margin:4px 0 0">Ajustez les poids (0% – 35%)</p>
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
            # Le slider lit sa valeur depuis st.session_state[f"slider_{nom}"]
            # qu'on a injecté juste au-dessus — value= est ignoré si la clé existe,
            # donc on ne passe pas value= du tout pour éviter toute confusion.
            poids_user[nom] = st.slider(
                nom.replace("FCP ", "").replace("SICAV ", ""),
                min_value=0.0, max_value=35.0,
                step=0.1,
                key=f"slider_{nom}",
                format="%.1f%%"
            )
            # Mettre à jour la cible avec la valeur courante du slider
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

# ── Bouton Réinitialiser ──────────────────────────────────────────────────────
if st.sidebar.button("🔄 Réinitialiser les poids", use_container_width=True):
    for nom in NOMS:
        st.session_state[f"poids_cible_{nom}"] = float(POIDS_ACTUELS[nom])
    st.rerun()

# ── Sélecteur de méthode ──────────────────────────────────────────────────────
methode_optim = st.sidebar.selectbox(
    "🎯 Méthode d'optimisation",
    METHODES,
    key="methode_select",
)

# ── Bouton Appliquer ──────────────────────────────────────────────────────────
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

w_raw  = np.array([poids_user[n] / 100 for n in NOMS])
w_norm = w_raw / w_raw.sum()
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
            </div>
            """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 1.9])

    with col_left:
        st.markdown('<div class="section-title">🥧 Répartition des Poids</div>', unsafe_allow_html=True)
        poids_pct = w_norm * 100
        fig_pie = go.Figure(go.Pie(
            labels=[n.replace("FCP ", "").replace("SICAV ", "") for n in NOMS],
            values=poids_pct,
            textinfo="label+percent",
            textfont_size=10,
            marker=dict(colors=px.colors.qualitative.Set3, line=dict(color="white", width=1.5)),
            hole=0.35,
        ))
        fig_pie.update_layout(
            showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
            height=380, paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text=f"<b>Total: {total_poids:.0f}%</b>", x=0.5, y=0.5,
                              font_size=14, showarrow=False,
                              font=dict(color=COLORS["dark_green"]))]
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
        fig_bar.update_layout(
            barmode="group", height=380,
            margin=dict(t=10, b=80, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(244,247,251,0.5)",
            xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
            yaxis=dict(title="Poids (%)", gridcolor="#E2E8F0", range=[0, 40]),
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
# TAB 2 — FRONTIÈRE EFFICIENTE AMÉLIORÉE
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">🎯 Frontière Efficiente de Markowitz — Vue enrichie</div>',
                unsafe_allow_html=True)

    # ── Contrôles utilisateur ────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        colorby = st.selectbox(
            "Colorier la frontière par :",
            ["Ratio de Sharpe", "VaR 99%", "CVaR 99%"],
            index=0
        )
    with ctrl2:
        show_indiv = st.checkbox("Afficher les OPCVM individuels", value=True)
    with ctrl3:
        show_cml = st.checkbox("Afficher la CML", value=True)

    # ── Calculs ──────────────────────────────────────────────────────
    with st.spinner("Calcul de la frontière efficiente (200 points)..."):
        fe_vols, fe_rets, fe_sharpes, fe_var99s, fe_cvar99s = calculer_frontiere()

    optim_points = {}
    with st.spinner("Calcul des 3 portefeuilles optimisés..."):
        for m in METHODES:
            w_opt = optimiser(m, w_ref)
            optim_points[m] = calcul_stats(w_opt)

    col_opt = {
        "Min Variance": COLORS["mid_green"],
        "Max Sharpe":   COLORS["navy"],
        "Min CVaR":     COLORS["red"],
    }
    markers_m = {
        "Min Variance": "square",
        "Max Sharpe":   "triangle-up",
        "Min CVaR":     "diamond",
    }

    # ── Choix de la variable couleur ─────────────────────────────────
    color_map_choice = {
        "Ratio de Sharpe": (fe_sharpes, "Sharpe", "RdYlGn",    False),
        "VaR 99%":         (fe_var99s,  "VaR 99%<br>(%/j)",  "RdYlGn_r", False),
        "CVaR 99%":        (fe_cvar99s, "CVaR 99%<br>(%/j)", "RdYlGn_r", False),
    }
    color_vals, color_title, colorscale, _ = color_map_choice[colorby]

    # ── Figure ───────────────────────────────────────────────────────
    fig_fe = go.Figure()

    # Zone de fond : annotations quadrants
    if len(fe_vols) > 0:
        mid_vol = (fe_vols.min() + fe_vols.max()) / 2
        mid_ret = (fe_rets.min() + fe_rets.max()) / 2

        quad_style = dict(showarrow=False, font=dict(size=9, color="#CBD5E1"), opacity=0.6)
        for txt, x, y in [
            ("🟢 Rendement élevé\nRisque faible", fe_vols.min() * 1.2, fe_rets.max() * 0.95),
            ("🔴 Rendement faible\nRisque élevé", fe_vols.max() * 0.85, fe_rets.min() * 1.1),
        ]:
            fig_fe.add_annotation(x=x, y=y, text=txt, **quad_style)

    # Frontière efficiente — scatter coloré
    if len(fe_vols) > 0:
        hover_fe = [
            f"Vol: {v:.2f}%<br>Perf: {r:.2f}%<br>Sharpe: {s:.3f}<br>VaR: {va:.4f}%<br>CVaR: {cv:.4f}%"
            for v, r, s, va, cv in zip(fe_vols, fe_rets, fe_sharpes, fe_var99s, fe_cvar99s)
        ]
        fig_fe.add_trace(go.Scatter(
            x=fe_vols, y=fe_rets,
            mode="markers",
            marker=dict(
                color=color_vals,
                colorscale=colorscale,
                size=8,
                opacity=0.9,
                colorbar=dict(
                    title=color_title,
                    thickness=14,
                    len=0.55,
                    y=0.75,
                    yanchor="top",
                ),
                line=dict(width=0),
            ),
            name="Frontière Efficiente",
            hovertemplate="%{customdata}<extra>Frontière Efficiente</extra>",
            customdata=hover_fe,
        ))

        # Ligne de contour de la frontière (fond)
        sorted_idx = np.argsort(fe_vols)
        fig_fe.add_trace(go.Scatter(
            x=fe_vols[sorted_idx], y=fe_rets[sorted_idx],
            mode="lines",
            line=dict(color="rgba(150,180,150,0.35)", width=3),
            showlegend=False,
            hoverinfo="skip",
        ))

    # ── Portefeuille Tangent (Max Sharpe sur la frontière) ────────────
    if len(fe_sharpes) > 0:
        tang_idx = np.argmax(fe_sharpes)
        tang_vol = fe_vols[tang_idx]
        tang_ret = fe_rets[tang_idx]
        tang_sh  = fe_sharpes[tang_idx]

        fig_fe.add_trace(go.Scatter(
            x=[tang_vol], y=[tang_ret],
            mode="markers+text",
            marker=dict(size=22, color=COLORS["gold"], symbol="star",
                        line=dict(color=COLORS["dark_green"], width=2)),
            text=["Portefeuille<br>Tangent"],
            textposition="top right",
            textfont=dict(size=10, color=COLORS["gold"], family="Arial Black"),
            name=f"Tangent (Sharpe={tang_sh:.3f})",
            hovertemplate=(
                f"<b>Portefeuille Tangent</b><br>"
                f"Vol: {tang_vol:.2f}%<br>Perf: {tang_ret:.2f}%<br>"
                f"Sharpe: {tang_sh:.3f}<extra></extra>"
            ),
        ))

        # CML : depuis Rf jusqu'au tangent puis extrapolée
        if show_cml:
            slope_cml = (tang_ret / 100 - RF) / (tang_vol / 100 + 1e-8)
            vol_end   = tang_vol * 1.5
            vol_cml   = np.array([0, tang_vol, vol_end])
            ret_cml   = (RF + slope_cml * vol_cml / 100) * 100

            # Segment Rf → Tangent (plein)
            fig_fe.add_trace(go.Scatter(
                x=vol_cml[:2], y=ret_cml[:2],
                mode="lines",
                line=dict(color=COLORS["gold"], width=2.5),
                name=f"CML (Rf={RF*100:.2f}%)",
                hoverinfo="skip",
            ))
            # Segment Tangent → extrapolé (pointillé)
            fig_fe.add_trace(go.Scatter(
                x=vol_cml[1:], y=ret_cml[1:],
                mode="lines",
                line=dict(color=COLORS["gold"], width=1.5, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ))
            # Point Rf
            fig_fe.add_trace(go.Scatter(
                x=[0], y=[RF * 100],
                mode="markers+text",
                marker=dict(size=10, color=COLORS["gold"], symbol="circle",
                            line=dict(color="white", width=1.5)),
                text=[f"Rf={RF*100:.2f}%"],
                textposition="top right",
                textfont=dict(size=9, color=COLORS["gold"]),
                showlegend=False,
                hoverinfo="skip",
            ))

    # ── OPCVM individuels ─────────────────────────────────────────────
    if show_indiv:
        for nom in NOMS:
            m   = META[nom]
            aj  = m["alpha_j"]
            col = COLORS["mid_green"] if aj > 0 else COLORS["red"]
            fig_fe.add_trace(go.Scatter(
                x=[m["vol"]], y=[m["perf"]],
                mode="markers+text",
                marker=dict(size=9, color=col, symbol="circle", opacity=0.75,
                            line=dict(color="white", width=1)),
                text=[nom[:12]],
                textposition="top center",
                textfont=dict(size=7.5, color=COLORS["dark_green"]),
                name=nom[:20],
                showlegend=False,
                hovertemplate=(
                    f"<b>{nom}</b><br>"
                    f"Vol: {m['vol']:.2f}%<br>Perf: {m['perf']:.2f}%<br>"
                    f"Sharpe: {m['sharpe']:.3f}<br>α: {aj:+.2f}%"
                    "<extra>OPCVM individuel</extra>"
                ),
            ))

    # ── Portefeuille Actuel ───────────────────────────────────────────
    fig_fe.add_trace(go.Scatter(
        x=[stats_ref["vol"]], y=[stats_ref["perf"]],
        mode="markers+text",
        marker=dict(size=18, color=COLORS["gray"], symbol="square",
                    line=dict(color="white", width=2)),
        text=["Actuel"], textposition="top center",
        textfont=dict(size=10, color=COLORS["gray"], family="Arial Black"),
        name="Portefeuille Actuel",
        hovertemplate=(
            f"<b>Portefeuille Actuel</b><br>"
            f"Vol: {stats_ref['vol']:.2f}%<br>Perf: {stats_ref['perf']:.2f}%<br>"
            f"Sharpe: {stats_ref['sharpe']:.3f}<extra></extra>"
        ),
    ))

    # ── Portefeuille Courant (user) ───────────────────────────────────
    fig_fe.add_trace(go.Scatter(
        x=[stats_cur["vol"]], y=[stats_cur["perf"]],
        mode="markers+text",
        marker=dict(size=22, color=COLORS["light_green"], symbol="star",
                    line=dict(color=COLORS["dark_green"], width=2.5)),
        text=["Votre portefeuille"], textposition="top left",
        textfont=dict(size=10, color=COLORS["mid_green"], family="Arial Black"),
        name="Portefeuille Courant",
        hovertemplate=(
            f"<b>Portefeuille Courant</b><br>"
            f"Vol: {stats_cur['vol']:.2f}%<br>Perf: {stats_cur['perf']:.2f}%<br>"
            f"Sharpe: {stats_cur['sharpe']:.3f}<extra></extra>"
        ),
    ))

    # ── 3 méthodes d'optimisation ─────────────────────────────────────
    for m_name, s in optim_points.items():
        fig_fe.add_trace(go.Scatter(
            x=[s["vol"]], y=[s["perf"]],
            mode="markers+text",
            marker=dict(size=16, color=col_opt[m_name], symbol=markers_m[m_name],
                        line=dict(color="white", width=1.5)),
            text=[m_name[:12]], textposition="bottom center",
            textfont=dict(size=8.5, color=col_opt[m_name], family="Arial Black"),
            name=m_name,
            hovertemplate=(
                f"<b>{m_name}</b><br>"
                f"Vol: {s['vol']:.2f}%<br>Perf: {s['perf']:.2f}%<br>"
                f"Sharpe: {s['sharpe']:.3f}<br>VaR: {s['var99']:.4f}%"
                "<extra></extra>"
            ),
        ))

    # Ligne de distance vers la frontière (actuel → point le + proche)
    if len(fe_vols) > 0:
        dist = np.sqrt((fe_vols - stats_ref["vol"])**2 + (fe_rets - stats_ref["perf"])**2)
        nearest_idx = np.argmin(dist)
        fig_fe.add_trace(go.Scatter(
            x=[stats_ref["vol"], fe_vols[nearest_idx]],
            y=[stats_ref["perf"], fe_rets[nearest_idx]],
            mode="lines",
            line=dict(color=COLORS["gray"], dash="dot", width=1.2),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig_fe.add_annotation(
            x=(stats_ref["vol"] + fe_vols[nearest_idx]) / 2,
            y=(stats_ref["perf"] + fe_rets[nearest_idx]) / 2,
            text=f"Δ vers frontière",
            showarrow=False,
            font=dict(size=8, color=COLORS["gray"]),
            bgcolor="rgba(255,255,255,0.7)",
        )

    fig_fe.update_layout(
        height=650,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(244,247,251,0.7)",
        xaxis=dict(
            title="Volatilité Annualisée (%)",
            gridcolor="#E2E8F0", zeroline=False,
            showspikes=True, spikemode="across",
            spikesnap="cursor", spikecolor="#CBD5E1",
        ),
        yaxis=dict(
            title="Performance Annualisée (%)",
            gridcolor="#E2E8F0", zeroline=False,
            showspikes=True, spikemode="across",
            spikesnap="cursor", spikecolor="#CBD5E1",
        ),
        legend=dict(
            orientation="v", x=1.12, y=1,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=COLORS["light_green"],
            borderwidth=1,
            font=dict(size=9),
            tracegroupgap=4,
        ),
        margin=dict(t=20, b=40, l=55, r=220),
        font=dict(family="Calibri"),
        hovermode="closest",
    )
    st.plotly_chart(fig_fe, use_container_width=True)

    # ── Légende explicative ───────────────────────────────────────────
    leg1, leg2, leg3, leg4, leg5 = st.columns(5)
    with leg1:
        st.markdown(f"""<div style="text-align:center;padding:8px;background:white;border-radius:8px;
        border:2px solid {COLORS['gold']};font-size:0.8rem;">
        ⭐ <b>Tangent</b><br>Max Sharpe<br>sur la frontière</div>""", unsafe_allow_html=True)
    with leg2:
        st.markdown(f"""<div style="text-align:center;padding:8px;background:white;border-radius:8px;
        border:2px solid {COLORS['mid_green']};font-size:0.8rem;">
        🟩 <b>Min Variance</b><br>Risque minimal</div>""", unsafe_allow_html=True)
    with leg3:
        st.markdown(f"""<div style="text-align:center;padding:8px;background:white;border-radius:8px;
        border:2px solid {COLORS['navy']};font-size:0.8rem;">
        🔺 <b>Max Sharpe</b><br>Meilleur ratio</div>""", unsafe_allow_html=True)
    with leg4:
        st.markdown(f"""<div style="text-align:center;padding:8px;background:white;border-radius:8px;
        border:2px solid {COLORS['red']};font-size:0.8rem;">
        💎 <b>Min CVaR</b><br>Risque extrême minimal</div>""", unsafe_allow_html=True)
    with leg5:
        st.markdown(f"""<div style="text-align:center;padding:8px;background:white;border-radius:8px;
        border:2px solid {COLORS['gray']};font-size:0.8rem;">
        ⬛ <b>Actuel</b><br>Portefeuille référence</div>""", unsafe_allow_html=True)

    # ── Résumé tableau ────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Résumé des 3 Méthodes d\'Optimisation</div>', unsafe_allow_html=True)
    rows_m = []
    for m_name, s in optim_points.items():
        rows_m.append({
            "Méthode":            m_name,
            "Performance (%)":    f"{s['perf']:.2f}%",
            "Volatilité (%)":     f"{s['vol']:.2f}%",
            "Sharpe":             f"{s['sharpe']:.3f}",
            "Sortino":            f"{s['sortino']:.3f}",
            "VaR 99% (j)":        f"{s['var99']:.4f}%",
            "CVaR 99% (j)":       f"{s['cvar99']:.4f}%",
            "Drawdown Max (%)":   f"{s['dd_max']:.2f}%",
        })
    # Ajouter Tangent
    if len(fe_sharpes) > 0:
        tang_s = calcul_stats(
            optimiser("Max Sharpe", w_ref)  # approximation du tangent
        )
        rows_m.insert(0, {
            "Méthode":            "⭐ Tangent (FE)",
            "Performance (%)":    f"{tang_ret:.2f}%",
            "Volatilité (%)":     f"{tang_vol:.2f}%",
            "Sharpe":             f"{tang_sh:.3f}",
            "Sortino":            f"{tang_s['sortino']:.3f}",
            "VaR 99% (j)":        f"{fe_var99s[tang_idx]:.4f}%",
            "CVaR 99% (j)":       f"{fe_cvar99s[tang_idx]:.4f}%",
            "Drawdown Max (%)":   f"{tang_s['dd_max']:.2f}%",
        })
    df_m = pd.DataFrame(rows_m).set_index("Méthode")
    st.dataframe(df_m, use_container_width=True, height=230)

    # ── Tableau des pondérations optimisées par méthode ───────────────
    st.markdown('<div class="section-title">📐 Pondérations Optimisées par Méthode (poids libres 0–100%)</div>',
                unsafe_allow_html=True)

    # Calculer les poids pour chaque méthode
    w_par_methode = {}
    for m in METHODES:
        w_par_methode[m] = optimiser(m, w_ref)

    # Construire le dataframe : lignes = OPCVM, colonnes = Actuel + 3 méthodes
    rows_w = []
    for i, nom in enumerate(NOMS):
        row = {
            "OPCVM": nom[:22],
            "Actuel (%)": f"{w_ref[i]*100:.1f}%",
        }
        for m in METHODES:
            w_val = w_par_methode[m][i] * 100
            row[f"{m} (%)"] = f"{w_val:.1f}%"
        rows_w.append(row)

    df_w = pd.DataFrame(rows_w).set_index("OPCVM")

    # Ligne totale
    total_row = {"Actuel (%)": "100.0%"}
    for m in METHODES:
        total_row[f"{m} (%)"] = f"{w_par_methode[m].sum()*100:.1f}%"
    df_totaux = pd.DataFrame([total_row], index=["∑ Total"])
    df_w_full = pd.concat([df_w, df_totaux])

    st.dataframe(df_w_full, use_container_width=True, height=560)

    # Note explicative
    st.markdown(f"""
    <div style="background:{COLORS['mint']};border-left:4px solid {COLORS['mid_green']};
                padding:10px 14px;border-radius:6px;font-size:0.82rem;color:{COLORS['dark_green']};
                margin-top:8px;">
      <b>ℹ️ Bornes dynamiques par OPCVM :</b> Chaque fonds reçoit une borne maximale calculée
      selon son Sharpe (60%) et sa faible volatilité (40%). Les OPCVM les plus efficaces
      (CDG RENDEMENT, EMERGENCE SERENITE) peuvent recevoir jusqu'à ~35%, les moins efficaces
      restent plafonnés plus bas. Une présence minimale (~30% du poids actuel) est maintenue
      sur chaque fonds pour assurer la diversification. Somme = 100% garantie.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — ANALYSE DÉTAILLÉE
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">📋 Indicateurs Individuels par OPCVM</div>', unsafe_allow_html=True)

    rows_d = []
    for i, nom in enumerate(NOMS):
        m = META[nom]
        wp = w_norm[i] * 100
        wa = POIDS_ACTUELS[nom]
        rows_d.append({
            "OPCVM":                 nom[:28],
            "Poids Actuel (%)":      f"{wa:.1f}%",
            "Poids Courant (%)":     f"{wp:.1f}%",
            "Δ (pp)":                f"{wp-wa:+.1f}",
            "Performance (%)":       f"{m['perf']:.2f}%",
            "Volatilité (%)":        f"{m['vol']:.2f}%",
            "Beta (β)":              f"{m['beta']:.2f}",
            "Alpha Jensen (%)":      f"{m['alpha_j']:+.2f}%",
            "Sharpe":                f"{m['sharpe']:.3f}",
            "Sortino":               f"{m['sortino']:.3f}",
            "VaR 99% (%)":           f"{m['var99']:.3f}%",
            "CVaR 99% (%)":          f"{m['cvar99']:.3f}%",
            "Tracking Error (%)":    f"{m['te']:.3f}%",
            "Information Ratio":     f"{m['ir']:.3f}",
            "Drawdown Max (%)":      f"{m['dd']:.2f}%",
        })

    df_d = pd.DataFrame(rows_d).set_index("OPCVM")
    st.dataframe(df_d, use_container_width=True, height=500)

    st.markdown('<div class="section-title">🔵 Alpha Jensen vs Sharpe Ratio (Taille = Poids Courant)</div>',
                unsafe_allow_html=True)
    fig_sc = go.Figure()
    for i, nom in enumerate(NOMS):
        m  = META[nom]
        wp = w_norm[i] * 100
        col = COLORS["mid_green"] if m["alpha_j"] > 0 else COLORS["red"]
        fig_sc.add_trace(go.Scatter(
            x=[m["alpha_j"]], y=[m["sharpe"]],
            mode="markers+text",
            marker=dict(size=max(wp / 2, 8), color=col, opacity=0.7,
                        line=dict(color="white", width=1)),
            text=[nom[:14]], textposition="top center",
            textfont=dict(size=9, color=COLORS["dark_green"]),
            name=nom[:20], showlegend=False,
            hovertemplate=(
                f"<b>{nom}</b><br>α: {m['alpha_j']:+.2f}%<br>"
                f"Sharpe: {m['sharpe']:.3f}<br>Poids: {wp:.1f}%<extra></extra>"
            ),
        ))
    fig_sc.add_vline(x=0, line_dash="dot", line_color=COLORS["red"],  line_width=1.5)
    fig_sc.add_hline(y=2, line_dash="dot", line_color=COLORS["navy"], line_width=1.5)
    fig_sc.update_layout(
        height=450, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(244,247,251,0.7)",
        xaxis=dict(title="Alpha Jensen (%)", gridcolor="#E2E8F0", zeroline=True),
        yaxis=dict(title="Sharpe Ratio",     gridcolor="#E2E8F0", zeroline=True),
        margin=dict(t=30, b=40, l=50, r=80),
    )
    st.plotly_chart(fig_sc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — COMPARAISON MÉTHODES (3 seulement)
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
        noms_short = [n[:14] for n in NOMS]
        methodes_h = ["Actuel"] + list(w_optimises.keys())
        w_matrix = np.vstack([w_ref * 100, *[w_optimises[m] * 100 for m in w_optimises]])
        fig_hm = go.Figure(go.Heatmap(
            z=w_matrix, x=noms_short, y=methodes_h,
            colorscale="Greens",
            text=np.round(w_matrix, 1),
            texttemplate="%{text}%",
            textfont=dict(size=8, color="black"),
            hovertemplate="<b>%{y}</b> — %{x}<br>Poids: %{z:.1f}%<extra></extra>",
            colorbar=dict(title="Poids (%)", thickness=15, len=0.8),
            zmin=0, zmax=35,
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
            x=vol_m, y=perf_m,
            mode="markers+text",
            marker=dict(size=28, color=cols_bar2, line=dict(color="white", width=2)),
            text=noms_ptf,
            textposition="middle center",
            textfont=dict(size=8, color="white", family="Arial Black"),
            hovertemplate="<b>%{text}</b><br>Vol: %{x:.2f}%<br>Perf: %{y:.2f}%<extra></extra>",
        ))
        if len(vol_m) > 1:
            z = np.polyfit(vol_m, perf_m, 1)
            p = np.poly1d(z)
            x_tr = np.linspace(min(vol_m) * 0.9, max(vol_m) * 1.1, 100)
            fig_pv.add_trace(go.Scatter(
                x=x_tr, y=p(x_tr), mode="lines", name="Tendance",
                line=dict(color=COLORS["gold"], dash="dash", width=1.5), opacity=0.5
            ))
        fig_pv.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(244,247,251,0.7)",
            xaxis=dict(title="Volatilité Annualisée (%)", gridcolor="#E2E8F0"),
            yaxis=dict(title="Performance Annualisée (%)", gridcolor="#E2E8F0"),
            margin=dict(t=20, b=30, l=50, r=30),
            hovermode="closest",
        )
        st.plotly_chart(fig_pv, use_container_width=True)

    st.markdown('<div class="section-title">📉 Comparaison des Mesures de Risque</div>', unsafe_allow_html=True)
    noms_ptf  = ["Actuel"] + list(optim_points.keys())
    vars_ptf  = [stats_ref["var99"]]  + [optim_points[m]["var99"]  for m in optim_points]
    cvars_ptf = [stats_ref["cvar99"]] + [optim_points[m]["cvar99"] for m in optim_points]
    cols_bar2 = [COLORS["gray"], COLORS["mid_green"], COLORS["navy"], COLORS["red"]]

    fig_var = go.Figure()
    fig_var.add_trace(go.Bar(
        name="VaR 99%",  x=noms_ptf, y=vars_ptf,
        marker_color=cols_bar2, opacity=0.85,
        text=[f"{v:.4f}%" for v in vars_ptf],
        textposition="inside", textfont=dict(color="white", size=10),
    ))
    fig_var.add_trace(go.Bar(
        name="CVaR 99%", x=noms_ptf, y=cvars_ptf,
        marker_color=cols_bar2, opacity=0.45,
        text=[f"{v:.4f}%" for v in cvars_ptf],
        textposition="inside", textfont=dict(color="white", size=9),
    ))
    fig_var.update_layout(
        barmode="group", height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(244,247,251,0.7)",
        yaxis=dict(title="VaR / CVaR (%/jour)", gridcolor="#E2E8F0"),
        legend=dict(orientation="h", x=0, y=1.12),
        margin=dict(t=40, b=30, l=50, r=30),
    )
    st.plotly_chart(fig_var, use_container_width=True)

    st.markdown('<div class="section-title">⭐ Comparaison Sharpe & Sortino</div>', unsafe_allow_html=True)
    sharpes  = [stats_ref["sharpe"]]  + [optim_points[m]["sharpe"]  for m in optim_points]
    sortinos = [stats_ref["sortino"]] + [optim_points[m]["sortino"] for m in optim_points]
    fig_sh = go.Figure()
    fig_sh.add_trace(go.Bar(
        name="Sharpe",  x=noms_ptf, y=sharpes,
        marker_color=cols_bar2, opacity=0.85,
        text=[f"{v:.3f}" for v in sharpes], textposition="outside"
    ))
    fig_sh.add_trace(go.Bar(
        name="Sortino", x=noms_ptf, y=sortinos,
        marker_color=cols_bar2, opacity=0.45,
        text=[f"{v:.3f}" for v in sortinos], textposition="outside"
    ))
    fig_sh.update_layout(
        barmode="group", height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(244,247,251,0.7)",
        yaxis=dict(title="Ratio", gridcolor="#E2E8F0",
                   range=[0, max(sharpes + sortinos) * 1.15]),
        legend=dict(orientation="h", x=0, y=1.12),
        margin=dict(t=40, b=30, l=50, r=30),
    )
    st.plotly_chart(fig_sh, use_container_width=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;padding:20px;">
  <p style="color:{COLORS['gray']};font-size:0.75rem;margin:0;">
    <b>OPCVM Portfolio Dashboard v2</b> — Rendements synthétiques cohérents avec les statistiques réelles.<br>
    Rf = 2.25% · 247 jours · 3 méthodes : Min Variance · Max Sharpe · Min CVaR · Optimiseur SLSQP
  </p>
  <p style="color:{COLORS['light_green']};font-size:0.8rem;margin:8px 0 0;font-weight:600;letter-spacing:0.05em;">
    © 2026 · Raydae
  </p>
</div>
""", unsafe_allow_html=True)
