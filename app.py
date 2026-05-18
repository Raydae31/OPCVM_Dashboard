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
        padding: 8px 12px; border-radius: 6px; font-size: 0.85rem; color: {COLORS['dark_green']};
    }}
    .upload-box {{
        background: {COLORS['mid_green']}; border-radius: 8px; padding: 10px 12px;
        margin-bottom: 12px; border: 2px dashed {COLORS['light_green']};
    }}
    .data-badge-real {{
        background: {COLORS['light_green']}; color: {COLORS['dark_green']};
        padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700;
        display: inline-block; margin-left: 8px;
    }}
    .data-badge-sim {{
        background: {COLORS['gold']}; color: {COLORS['dark_green']};
        padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700;
        display: inline-block; margin-left: 8px;
    }}
    .stTabs [aria-selected="true"] {{ background: {COLORS['mid_green']}; color: white !important; }}
    .stFileUploader {{ background: transparent !important; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DONNÉES META — VALEURS RÉELLES CORRIGÉES
# VaR CDG RENDEMENT = -0.297% (hebdo/√5) corrigée
# VaR EMERGENCE SERENITE = -0.287% (hebdo/√5) corrigée
# ══════════════════════════════════════════════════════════════════════════════

POIDS_ACTUELS_BASE = {
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
    "AD YIELD FUND":           4.04,   # sera mis à 0 si absent du fichier réel
}

META_BASE = {
    "AFG GOV BOND FUND":       {"perf":  5.74, "vol":  2.36, "sharpe": 1.63, "sortino": 2.31, "alpha_j":  1.20, "beta": 1.23, "te":  1.50, "ir":  0.82, "dd": -2.71, "var99": -0.384,  "cvar99": -0.434},
    "AD BALANCED FUND":        {"perf":  9.58, "vol": 32.49, "sharpe": 1.93, "sortino": 2.88, "alpha_j": -2.33, "beta": 1.35, "te": 10.52, "ir": -0.89, "dd": -8.87, "var99": -1.906,  "cvar99": -2.154},
    "AFG OPTIMAL FUND":        {"perf": 12.22, "vol":  8.79, "sharpe": 1.23, "sortino": 1.46, "alpha_j":  1.57, "beta": 1.54, "te":  3.52, "ir":  0.49, "dd": -7.75, "var99": -1.573,  "cvar99": -1.777},
    "CDG IZDIHAR":             {"perf":  8.98, "vol": 24.76, "sharpe": 2.34, "sortino": 3.97, "alpha_j":  4.39, "beta": 2.85, "te": 22.95, "ir":  1.13, "dd": -6.71, "var99": -1.2394, "cvar99": -1.401},
    "AD SELECT BANK":          {"perf":  3.39, "vol":  0.55, "sharpe": 2.44, "sortino": 3.23, "alpha_j":  0.90, "beta": 1.19, "te":  0.32, "ir":  2.99, "dd": -0.19, "var99": -0.081,  "cvar99": -0.092},
    "ALPHA BANQUES FUND":      {"perf":  3.12, "vol":  0.50, "sharpe": 2.16, "sortino": 2.90, "alpha_j":  0.66, "beta": 1.06, "te":  0.28, "ir":  2.41, "dd": -0.18, "var99": -0.078,  "cvar99": -0.088},
    "ALPHA SECURE FUND":       {"perf":  4.89, "vol":  1.71, "sharpe": 1.73, "sortino": 2.18, "alpha_j":  0.35, "beta": 0.94, "te":  0.95, "ir":  0.38, "dd": -1.59, "var99": -0.274,  "cvar99": -0.310},
    "CAM OBLIBANQUES":         {"perf":  3.17, "vol":  0.55, "sharpe": 2.04, "sortino": 2.84, "alpha_j":  0.60, "beta": 1.15, "te":  0.25, "ir":  2.58, "dd": -0.17, "var99": -0.074,  "cvar99": -0.084},
    # ✅ VaR corrigée : -0.633% → -0.297% (hebdo/√5)
    "CDG RENDEMENT":           {"perf":  5.06, "vol":  5.05, "sharpe": 4.58, "sortino": 6.49, "alpha_j": -0.28, "beta": 0.66, "te":  3.48, "ir": -0.38, "dd": -1.69, "var99": -0.297,  "cvar99": -0.336},
    "OBLIG CT":                {"perf":  3.23, "vol":  0.55, "sharpe": 2.18, "sortino": 2.69, "alpha_j":  0.66, "beta": 1.04, "te":  0.22, "ir":  3.02, "dd": -0.19, "var99": -0.096,  "cvar99": -0.109},
    "CDG TAWFIR":              {"perf":  5.01, "vol":  1.91, "sharpe": 1.62, "sortino": 2.08, "alpha_j":  0.47, "beta": 0.99, "te":  1.16, "ir":  0.42, "dd": -1.69, "var99": -0.344,  "cvar99": -0.389},
    # ✅ VaR corrigée : -0.616% → -0.287% (hebdo/√5)
    "EMERGENCE SERENITE":      {"perf":  4.63, "vol":  4.83, "sharpe": 4.31, "sortino": 5.79, "alpha_j":  0.06, "beta": 1.10, "te":  2.34, "ir":  0.12, "dd": -1.66, "var99": -0.287,  "cvar99": -0.325},
    "CAPITAL TRUST EQUILIBRE": {"perf": 11.49, "vol":  8.22, "sharpe": 1.22, "sortino": 1.45, "alpha_j":  0.84, "beta": 1.30, "te":  4.27, "ir":  0.23, "dd": -7.63, "var99": -1.725,  "cvar99": -1.949},
    "AD YIELD FUND":           {"perf":  3.11, "vol":  0.59, "sharpe": 1.81, "sortino": 2.45, "alpha_j":  0.59, "beta": 1.21, "te":  0.36, "ir":  1.83, "dd": -0.21, "var99": -0.089,  "cvar99": -0.101},
}

RF       = 0.0225
RF_DAILY = RF / 252
METHODES = ["Min Variance", "Max Sharpe", "Min CVaR"]


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DU FICHIER EXCEL RÉEL
# ══════════════════════════════════════════════════════════════════════════════

def charger_rendements_excel(uploaded_file):
    """
    Lit le fichier Rendements_OPCVM_final.xlsx.
    Détecte automatiquement la structure :
      - Première colonne = dates (ou index)
      - Colonnes suivantes = noms des OPCVM
    Retourne (R_matrix, noms_actifs, freq_info)
    """
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names

        # Chercher la feuille la plus probable
        target_sheet = None
        for sh in sheet_names:
            sh_lower = sh.lower()
            if any(k in sh_lower for k in ["rendement", "return", "perf", "nav", "vl", "data"]):
                target_sheet = sh
                break
        if target_sheet is None:
            target_sheet = sheet_names[0]

        df_raw = pd.read_excel(uploaded_file, sheet_name=target_sheet, index_col=0)

        # Nettoyer : supprimer lignes/colonnes entièrement vides
        df_raw = df_raw.dropna(how="all").dropna(axis=1, how="all")

        # Convertir en float
        df_num = df_raw.apply(pd.to_numeric, errors="coerce")
        df_num = df_num.dropna(how="all")

        # Si les valeurs ressemblent à des VL (>> 1), calculer les rendements
        col_means = df_num.mean()
        if col_means.median() > 10:
            # VL → rendements
            df_ret = df_num.pct_change().dropna()
        else:
            # Déjà des rendements
            df_ret = df_num.dropna()

        # Normaliser les noms de colonnes (uppercase strip)
        df_ret.columns = [str(c).strip().upper() for c in df_ret.columns]

        # Détecter la fréquence
        freq = "journalière"
        try:
            idx = pd.to_datetime(df_raw.index, errors="coerce")
            if idx.notna().sum() > 5:
                diffs = idx.dropna().diff().dropna().dt.days.median()
                if diffs >= 5:
                    freq = "hebdomadaire"
                elif diffs >= 25:
                    freq = "mensuelle"
        except Exception:
            pass

        return df_ret, list(df_ret.columns), freq, target_sheet, None

    except Exception as e:
        return None, None, None, None, str(e)


def aligner_opcvm(df_ret_raw, noms_meta):
    """
    Fait correspondre les colonnes du fichier Excel aux noms META.
    Retourne un DataFrame aligné sur les noms META disponibles.
    """
    mapping = {}
    cols_excel = list(df_ret_raw.columns)
    for nom_meta in noms_meta:
        nom_up = nom_meta.upper()
        # Correspondance exacte
        if nom_up in cols_excel:
            mapping[nom_meta] = nom_up
            continue
        # Correspondance partielle (mots-clés)
        mots = [m for m in nom_up.split() if len(m) > 3]
        best_score = 0
        best_col   = None
        for col in cols_excel:
            score = sum(1 for m in mots if m in col)
            if score > best_score:
                best_score = score
                best_col   = col
        if best_score >= 1:
            mapping[nom_meta] = best_col

    return mapping


def recalculer_meta_depuis_rendements(df_ret, freq="journalière"):
    """
    Recalcule perf, vol, VaR, CVaR, drawdown, sharpe, sortino
    directement depuis les rendements réels.
    """
    if freq == "hebdomadaire":
        ann = 52
        sqrt_ann = np.sqrt(52)
    elif freq == "mensuelle":
        ann = 12
        sqrt_ann = np.sqrt(12)
    else:
        ann = 252
        sqrt_ann = np.sqrt(252)

    meta_recalc = {}
    for col in df_ret.columns:
        r = df_ret[col].dropna().values
        if len(r) < 10:
            continue

        perf_ann = ((1 + r).prod() ** (ann / len(r)) - 1) * 100
        vol_ann  = r.std() * sqrt_ann * 100

        rf_per   = RF / ann
        sharpe   = (perf_ann/100 - RF) / (vol_ann/100) if vol_ann > 1e-8 else 0

        r_neg    = r[r < rf_per]
        dv       = r_neg.std() * sqrt_ann if len(r_neg) > 1 else vol_ann/100
        sortino  = (perf_ann/100 - RF) / dv if dv > 1e-8 else 0

        var99    = np.percentile(r, 1, method="lower") * 100
        mask     = r < np.percentile(r, 1, method="lower")
        cvar99   = r[mask].mean() * 100 if mask.any() else var99

        cum  = np.cumprod(1 + r)
        roll = np.maximum.accumulate(cum)
        dd   = ((cum - roll) / roll).min() * 100

        meta_recalc[col] = {
            "perf": perf_ann, "vol": vol_ann, "sharpe": sharpe,
            "sortino": sortino, "var99": var99, "cvar99": cvar99, "dd": dd,
        }
    return meta_recalc


# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS D'OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════

def calcul_stats(w_arr, R, rf_daily=RF_DAILY):
    r_ptf  = R @ w_arr
    perf   = r_ptf.mean() * 252 * 100
    vol    = r_ptf.std()  * np.sqrt(252) * 100
    sharpe = (perf/100 - RF) / (vol/100) if vol > 1e-8 else 0
    r_neg  = r_ptf[r_ptf < rf_daily]
    dv     = r_neg.std() * np.sqrt(252) if len(r_neg) > 1 else vol/100
    sortino = (perf/100 - RF) / dv if dv > 1e-8 else 0
    var99   = np.percentile(r_ptf, 1, method="lower") * 100
    mask    = r_ptf < np.percentile(r_ptf, 1, method="lower")
    cvar99  = r_ptf[mask].mean() * 100 if mask.any() else var99
    cum     = np.cumprod(1 + r_ptf)
    roll    = np.maximum.accumulate(cum)
    dd_max  = ((cum - roll) / roll).min() * 100
    return {"perf": perf, "vol": vol, "sharpe": sharpe, "sortino": sortino,
            "var99": var99, "cvar99": cvar99, "dd_max": dd_max}


def calculer_bornes_dynamiques(w_actuel, meta, noms):
    sharpe_ind = np.array([meta[n]["sharpe"] for n in noms])
    vol_ind    = np.array([meta[n]["vol"]    for n in noms])
    sharpe_pos = np.clip(sharpe_ind, 0.1, None)
    poids_sharpe = sharpe_pos / sharpe_pos.sum()
    vol_inv  = 1.0 / (vol_ind + 0.1)
    poids_vol = vol_inv / vol_inv.sum()
    score = 0.60 * poids_sharpe + 0.40 * poids_vol
    max_bounds = np.clip(score * 3.5, 0.05, 0.35)
    min_bounds = np.clip(w_actuel * 0.30, 0.005, 0.10)
    for i in range(len(noms)):
        if min_bounds[i] >= max_bounds[i]:
            min_bounds[i] = max(0.005, max_bounds[i] * 0.5)
    return list(zip(min_bounds, max_bounds))


def optimiser(methode, w_actuel, R, meta, noms):
    mu  = R.mean(axis=0) * 252
    cov = np.cov(R.T) * 252
    n   = len(w_actuel)
    bounds = calculer_bornes_dynamiques(w_actuel, meta, noms)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    ret_min = (mu @ w_actuel) * 0.80
    if methode != "Min Variance":
        constraints.append({"type": "ineq", "fun": lambda w: (mu @ w) - ret_min})

    cvar99_ind = np.array([meta[n]["cvar99"] for n in noms])

    objectives = {
        "Min Variance": lambda w: w @ cov @ w,
        "Max Sharpe":   lambda w: -(mu @ w - RF) / (np.sqrt(w @ cov @ w + 1e-12)),
        "Min CVaR":     lambda w: float(w @ cvar99_ind),
    }

    rng  = np.random.default_rng(42)
    best = None
    lo   = np.array([b[0] for b in bounds])
    hi   = np.array([b[1] for b in bounds])

    sharpe_ind = np.array([meta[nm]["sharpe"] for nm in noms])
    w_sharpe   = np.clip(sharpe_ind, 0.1, None)
    w_sharpe  /= w_sharpe.sum()

    starts = [w_actuel.copy()]
    for alpha in [1.0, 0.7, 0.5, 0.3]:
        w0 = alpha * w_sharpe + (1 - alpha) * rng.dirichlet(np.ones(n))
        w0 = np.clip(w0, lo, hi); w0 /= w0.sum()
        starts.append(w0)
    for _ in range(7):
        w0 = np.clip(rng.dirichlet(np.ones(n)), lo, hi); w0 /= w0.sum()
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
        w_opt = np.clip(best.x, lo, hi); w_opt /= w_opt.sum()
        return w_opt
    return w_actuel


def calculer_frontiere(R, meta, noms):
    mu  = R.mean(axis=0) * 252
    cov = np.cov(R.T) * 252
    n   = R.shape[1]
    targets  = np.linspace(mu.min(), mu.max() * 0.85, 150)
    vols, rets, sharpes, var99s, cvar99s = [], [], [], [], []
    constraints_base = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 0.35)] * n
    obj    = lambda w: w @ cov @ w
    var99_ind  = np.array([meta[nm]["var99"]  for nm in noms])
    cvar99_ind = np.array([meta[nm]["cvar99"] for nm in noms])
    rng = np.random.default_rng(0)
    for t in targets:
        cst  = constraints_base + [{"type": "eq", "fun": lambda w, tt=t: mu @ w - tt}]
        best = None
        for _ in range(6):
            w0 = np.clip(rng.dirichlet(np.ones(n)), 0, 0.35); w0 /= w0.sum()
            try:
                res = minimize(obj, w0, method="SLSQP", bounds=bounds,
                               constraints=cst, options={"maxiter": 800, "ftol": 1e-10})
                if res.success and (best is None or res.fun < best.fun):
                    best = res
            except Exception:
                pass
        if best is not None and best.success:
            w     = best.x
            vol_p = np.sqrt(w @ cov @ w) * 100
            ret_p = mu @ w * 100
            sh_p  = (ret_p/100 - RF) / (vol_p/100) if vol_p > 1e-8 else 0
            v99   = float(w @ var99_ind)
            cv99  = float(w @ cvar99_ind)
            vols.append(vol_p); rets.append(ret_p); sharpes.append(sh_p)
            var99s.append(v99); cvar99s.append(cv99)
    return (np.array(vols), np.array(rets),
            np.array(sharpes), np.array(var99s), np.array(cvar99s))


def generer_rendements_synthetiques_fallback(meta, noms):
    """Rendements synthétiques (fallback si pas de fichier Excel)"""
    np.random.seed(42)
    T     = 247
    N     = len(noms)
    mu_j  = np.array([meta[n]["perf"] / 100 / 252         for n in noms])
    sig_j = np.array([meta[n]["vol"]  / 100 / np.sqrt(252) for n in noms])
    betas = np.array([meta[n]["beta"] for n in noms])
    r_market = np.random.normal(0.0003, 0.001558, T)
    R = np.zeros((T, N))
    for i in range(N):
        alpha_j_daily = mu_j[i] - betas[i] * 0.0003
        idio_vol = max(sig_j[i] * np.sqrt(1 - min(betas[i]**2 * 0.35, 0.9)), 1e-5)
        eps = np.random.normal(0, idio_vol, T)
        R[:, i] = alpha_j_daily + betas[i] * r_market + eps
    for i in range(N):
        R[:, i] = R[:, i] - R[:, i].mean() + mu_j[i]
        scale   = sig_j[i] / (R[:, i].std() + 1e-12)
        R[:, i] = (R[:, i] - mu_j[i]) * scale + mu_j[i]
    return R


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — UPLOAD EXCEL
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown(f"""
<div style="background:{COLORS['mid_green']};padding:12px;border-radius:8px;margin-bottom:12px;
            border-left:4px solid {COLORS['light_green']}">
  <h2 style="color:white;margin:0;font-size:1.1rem">📂 Données OPCVM</h2>
  <p style="color:{COLORS['mint']};font-size:0.78rem;margin:4px 0 0">
    Importez votre fichier Excel de rendements réels
  </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "📤 Charger Rendements_OPCVM_final.xlsx",
    type=["xlsx", "xls"],
    help="Le fichier doit contenir une feuille avec les rendements (ou VL) des OPCVM en colonnes.",
    key="excel_uploader"
)

# ── État des données ─────────────────────────────────────────────────────────
DATA_SOURCE = "synthétique"
df_ret_excel = None
mapping_excel = {}
freq_excel = "journalière"
sheet_used = ""
error_msg  = None

if uploaded_file is not None:
    df_ret_excel, cols_excel, freq_excel, sheet_used, error_msg = charger_rendements_excel(uploaded_file)

    if df_ret_excel is not None and len(df_ret_excel) > 0:
        # Aligner les colonnes du fichier avec les noms META
        mapping_excel = aligner_opcvm(df_ret_excel, list(META_BASE.keys()))
        DATA_SOURCE   = "réel"
        st.sidebar.markdown(f"""
        <div style="background:#1B4332;border-radius:6px;padding:8px 10px;margin-bottom:8px;
                    border-left:3px solid {COLORS['light_green']}">
          <p style="color:{COLORS['light_green']};margin:0;font-size:0.78rem;font-weight:700">
            ✅ Fichier chargé — données réelles
          </p>
          <p style="color:{COLORS['mint']};margin:2px 0 0;font-size:0.72rem">
            📄 Feuille : <b>{sheet_used}</b><br>
            📅 Fréquence : <b>{freq_excel}</b><br>
            📊 {len(df_ret_excel)} observations · {len(cols_excel)} colonnes<br>
            🔗 {len(mapping_excel)} OPCVM appariés
          </p>
        </div>
        """, unsafe_allow_html=True)

        # Afficher le mapping
        if mapping_excel:
            with st.sidebar.expander("🔗 Correspondances détectées", expanded=False):
                for meta_nom, excel_col in mapping_excel.items():
                    st.markdown(f"<small style='color:white'><b>{meta_nom[:18]}</b> → {excel_col[:18]}</small>", unsafe_allow_html=True)
                missing = [n for n in META_BASE if n not in mapping_excel]
                if missing:
                    st.markdown(f"<small style='color:{COLORS['gold']}'>⚠️ Non trouvés : {', '.join([m[:15] for m in missing])}</small>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"""
        <div style="background:#5B1C1C;border-radius:6px;padding:8px 10px;margin-bottom:8px;">
          <p style="color:#FFAAAA;margin:0;font-size:0.78rem">⚠️ Erreur de lecture</p>
          <p style="color:#FFD0D0;margin:2px 0 0;font-size:0.70rem">{error_msg or 'Format non reconnu'}</p>
        </div>
        """, unsafe_allow_html=True)
        DATA_SOURCE = "synthétique"
else:
    st.sidebar.markdown(f"""
    <div style="background:{COLORS['dark_green']};border-radius:6px;padding:8px 10px;margin-bottom:8px;
                border:1px dashed {COLORS['light_green']}">
      <p style="color:{COLORS['gold']};margin:0;font-size:0.78rem;font-weight:700">
        ⚠️ Mode synthétique actif
      </p>
      <p style="color:{COLORS['mint']};margin:2px 0 0;font-size:0.70rem">
        Chargez votre fichier Excel pour utiliser les vraies données de rendements.
      </p>
    </div>
    """, unsafe_allow_html=True)


# ── Construire la matrice R et META effectifs ─────────────────────────────────
if DATA_SOURCE == "réel" and df_ret_excel is not None and mapping_excel:
    # Garder uniquement les OPCVM appariés
    noms_actifs = [n for n in META_BASE if n in mapping_excel]

    # Filtrer AD YIELD FUND si absent du fichier réel
    if "AD YIELD FUND" not in mapping_excel:
        if "AD YIELD FUND" in noms_actifs:
            noms_actifs.remove("AD YIELD FUND")
        st.sidebar.markdown(f"""
        <div style="background:#3B2A00;border-radius:5px;padding:6px 10px;margin-bottom:6px;">
          <p style="color:{COLORS['gold']};margin:0;font-size:0.72rem">
            ℹ️ AD YIELD FUND non trouvé dans le fichier → exclu du portefeuille
          </p>
        </div>
        """, unsafe_allow_html=True)

    # Construire le DataFrame aligné
    cols_aligned = [mapping_excel[n] for n in noms_actifs]
    df_aligned   = df_ret_excel[cols_aligned].dropna()
    R_GLOBAL     = df_aligned.values

    # Recalculer META depuis les rendements réels
    meta_recalc_raw = recalculer_meta_depuis_rendements(df_aligned, freq_excel)
    META = {}
    for i, nom in enumerate(noms_actifs):
        col = mapping_excel[nom]
        base = dict(META_BASE[nom])  # alpha_j, beta, te, ir de la META_BASE
        if col in meta_recalc_raw:
            r = meta_recalc_raw[col]
            base.update({
                "perf":    r["perf"],
                "vol":     r["vol"],
                "sharpe":  r["sharpe"],
                "sortino": r["sortino"],
                "var99":   r["var99"],
                "cvar99":  r["cvar99"],
                "dd":      r["dd"],
            })
        META[nom] = base

    # Poids actuels — exclure AD YIELD FUND si absent, renormaliser
    POIDS_ACTUELS = {n: POIDS_ACTUELS_BASE[n] for n in noms_actifs}
    total_pa = sum(POIDS_ACTUELS.values())
    POIDS_ACTUELS = {n: v / total_pa * 100 for n, v in POIDS_ACTUELS.items()}

    obs_label  = f"{len(df_aligned)} obs ({freq_excel})"
    data_badge = "DONNÉES RÉELLES"

else:
    # Fallback synthétique avec META_BASE corrigé
    noms_actifs    = list(META_BASE.keys())
    META           = META_BASE
    POIDS_ACTUELS  = POIDS_ACTUELS_BASE
    R_GLOBAL       = generer_rendements_synthetiques_fallback(META, noms_actifs)
    obs_label      = "247 obs (synthétique)"
    data_badge     = "DONNÉES SYNTHÉTIQUES"

NOMS = noms_actifs
N    = len(NOMS)

# ── Session state ─────────────────────────────────────────────────────────────
for nom in NOMS:
    key = f"poids_cible_{nom}"
    if key not in st.session_state:
        st.session_state[key] = float(POIDS_ACTUELS[nom])
for nom in NOMS:
    st.session_state[f"slider_{nom}"] = st.session_state[f"poids_cible_{nom}"]


# ── Sliders pondérations ──────────────────────────────────────────────────────
st.sidebar.markdown("---")
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
for cat, fonds_cat in CATEGORIES.items():
    fonds_presents = [f for f in fonds_cat if f in NOMS]
    if not fonds_presents:
        continue
    with st.sidebar.expander(cat, expanded=True):
        for nom in fonds_presents:
            poids_user[nom] = st.slider(
                nom.replace("FCP ", "").replace("SICAV ", ""),
                min_value=0.0, max_value=35.0, step=0.1,
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
        f'Les résultats sont normalisés automatiquement.</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Réinitialiser les poids", use_container_width=True):
    for nom in NOMS:
        st.session_state[f"poids_cible_{nom}"] = float(POIDS_ACTUELS[nom])
    st.rerun()

methode_optim = st.sidebar.selectbox("🎯 Méthode d'optimisation", METHODES, key="methode_select")

if st.sidebar.button(f"⚡ Appliquer : {methode_optim}", use_container_width=True, type="primary"):
    w_cur = np.array([poids_user[n] / 100 for n in NOMS])
    w_cur = np.clip(w_cur, 0.01, 0.25); w_cur /= w_cur.sum()
    w_opt = optimiser(methode_optim, w_cur, R_GLOBAL, META, NOMS)
    for i, nom in enumerate(NOMS):
        st.session_state[f"poids_cible_{nom}"] = round(float(w_opt[i]) * 100, 1)
    st.rerun()

st.sidebar.markdown(f"""
<div style="margin-top:12px;padding:8px;background:{COLORS['dark_green']};border-radius:6px;
            font-size:0.75rem;color:{COLORS['gray']};text-align:center;">
  📊 Rf = 2.25% · {obs_label} · {N} OPCVM<br>
  <span style="color:{COLORS['light_green']};font-weight:700">{data_badge}</span>
</div>
""", unsafe_allow_html=True)


# ── Calculs principaux ────────────────────────────────────────────────────────
w_raw     = np.array([poids_user[n] / 100 for n in NOMS])
w_norm    = w_raw / w_raw.sum()
stats_cur = calcul_stats(w_norm, R_GLOBAL)
w_ref     = np.array([POIDS_ACTUELS[n] / 100 for n in NOMS])
stats_ref = calcul_stats(w_ref, R_GLOBAL)


# ── Header ────────────────────────────────────────────────────────────────────
badge_color = COLORS["light_green"] if DATA_SOURCE == "réel" else COLORS["gold"]
badge_label = "✅ Données réelles" if DATA_SOURCE == "réel" else "⚠️ Données synthétiques"

st.markdown(f"""
<div class="main-header">
  <h1>📊 OPCVM Portfolio Dashboard
    <span style="background:{badge_color};color:{COLORS['dark_green']};padding:3px 12px;
                 border-radius:12px;font-size:0.75rem;font-weight:700;margin-left:10px;
                 vertical-align:middle;">{badge_label}</span>
  </h1>
  <p>Optimisation interactive · Frontière Efficiente · Rf = 2.25% · {N} OPCVM · {obs_label}</p>
</div>
""", unsafe_allow_html=True)

# Bannière d'information sur les corrections appliquées
if DATA_SOURCE == "synthétique":
    st.info(
        "**Mode synthétique** — Les VaR de CDG RENDEMENT (-0.297%) et EMERGENCE SERENITE (-0.287%) "
        "ont été corrigées (méthode hebdo/√5). "
        "Chargez votre fichier `Rendements_OPCVM_final.xlsx` via la sidebar pour utiliser les vraies données.",
        icon="ℹ️"
    )
elif DATA_SOURCE == "réel":
    st.success(
        f"**Données réelles chargées** depuis `{sheet_used}` — "
        f"Tous les indicateurs (VaR, Sharpe, Sortino, drawdown) sont calculés sur vos {len(df_ret_excel)} observations réelles.",
        icon="✅"
    )


tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Tableau de Bord", "🎯 Frontière Efficiente",
    "📋 Analyse Détaillée", "⚖️ Comparaison Méthodes",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">📌 Indicateurs du Portefeuille Courant</div>', unsafe_allow_html=True)

    def delta_html(val, ref, unit="%", inverse=False):
        diff = val - ref
        good = diff > 0 if not inverse else diff < 0
        signe = "+" if diff > 0 else ""
        cls = "delta-pos" if good else "delta-neg"
        return f'<span class="{cls}">{signe}{diff:.2f}{unit} vs actuel</span>'

    kpis = [
        ("Performance Annuelle",   f"{stats_cur['perf']:.2f}%",    stats_cur["perf"],    stats_ref["perf"],    "%",  False, ""),
        ("Volatilité Annualisée",  f"{stats_cur['vol']:.2f}%",     stats_cur["vol"],     stats_ref["vol"],     "%",  True,  "red"),
        ("Sharpe Ratio",           f"{stats_cur['sharpe']:.3f}",   stats_cur["sharpe"],  stats_ref["sharpe"],  "",   False, "gold"),
        ("Sortino Ratio",          f"{stats_cur['sortino']:.3f}",  stats_cur["sortino"], stats_ref["sortino"], "",   False, "gold"),
        ("VaR 99% (journalière)",  f"{stats_cur['var99']:.4f}%",   stats_cur["var99"],   stats_ref["var99"],   "%",  True,  "red"),
        ("CVaR 99% (journalière)", f"{stats_cur['cvar99']:.4f}%",  stats_cur["cvar99"],  stats_ref["cvar99"],  "%",  True,  "red"),
        ("Maximum Drawdown",       f"{stats_cur['dd_max']:.2f}%",  stats_cur["dd_max"],  stats_ref["dd_max"],  "%",  True,  "navy"),
    ]
    cols = st.columns(len(kpis))
    for i, (label, val_str, val, ref, unit, inverse, card_class) in enumerate(kpis):
        dh = delta_html(val, ref, unit, inverse)
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card {card_class}">
              <p>{label}</p><h3>{val_str}</h3><div class="delta">{dh}</div>
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
            showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=380,
            paper_bgcolor="rgba(0,0,0,0)",
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
            y=[POIDS_ACTUELS[n] for n in NOMS], marker_color=COLORS["gray"], opacity=0.7,
            text=[f"{POIDS_ACTUELS[n]:.1f}%" for n in NOMS], textposition="outside", textfont=dict(size=8)
        ))
        fig_bar.add_trace(go.Bar(
            name="Poids Courant", x=noms_short, y=poids_pct,
            marker_color=[COLORS["mid_green"] if d >= 0 else COLORS["red"] for d in delta_poids],
            opacity=0.85,
            text=[f"{p:.1f}%" for p in poids_pct], textposition="outside", textfont=dict(size=8)
        ))
        fig_bar.update_layout(
            barmode="group", height=380, margin=dict(t=10, b=80, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(244,247,251,0.5)",
            xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
            yaxis=dict(title="Poids (%)", gridcolor="#E2E8F0", range=[0, 40]),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(f'<div class="section-title">📈 Performance Cumulée ({obs_label})</div>', unsafe_allow_html=True)
    r_cur = R_GLOBAL @ w_norm; r_ref = R_GLOBAL @ w_ref
    cum_cur = (1 + r_cur).cumprod() - 1; cum_ref = (1 + r_ref).cumprod() - 1
    t_idx = np.arange(len(r_cur))
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=t_idx, y=cum_ref*100, name="Portefeuille Actuel",
        line=dict(color=COLORS["gray"], dash="dash", width=2), opacity=0.8))
    fig_cum.add_trace(go.Scatter(x=t_idx, y=cum_cur*100, name="Portefeuille Courant",
        line=dict(color=COLORS["mid_green"], width=2.5),
        fill="tonexty", fillcolor="rgba(44,95,45,0.1)"))
    fig_cum.add_hline(y=0, line_color=COLORS["gray"], line_width=0.8)
    fig_cum.update_layout(
        height=280, margin=dict(t=10, b=30, l=40, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(244,247,251,0.5)",
        legend=dict(orientation="h", x=0, y=1.12),
        yaxis=dict(title="Performance Cumulative (%)", gridcolor="#E2E8F0"),
        xaxis=dict(title="Observations", gridcolor="#E2E8F0"), hovermode="x unified"
    )
    st.plotly_chart(fig_cum, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — FRONTIÈRE EFFICIENTE
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">🎯 Frontière Efficiente de Markowitz</div>', unsafe_allow_html=True)

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        colorby = st.selectbox("Colorier par :", ["Ratio de Sharpe", "VaR 99%", "CVaR 99%"], index=0)
    with ctrl2:
        show_indiv = st.checkbox("Afficher les OPCVM individuels", value=True)
    with ctrl3:
        show_cml = st.checkbox("Afficher la CML", value=True)

    with st.spinner("Calcul de la frontière (150 points)..."):
        fe_vols, fe_rets, fe_sharpes, fe_var99s, fe_cvar99s = calculer_frontiere(R_GLOBAL, META, NOMS)

    optim_points = {}
    with st.spinner("Calcul des portefeuilles optimisés..."):
        for m in METHODES:
            w_opt = optimiser(m, w_ref, R_GLOBAL, META, NOMS)
            optim_points[m] = calcul_stats(w_opt, R_GLOBAL)

    col_opt = {"Min Variance": COLORS["mid_green"], "Max Sharpe": COLORS["navy"], "Min CVaR": COLORS["red"]}
    markers_m = {"Min Variance": "square", "Max Sharpe": "triangle-up", "Min CVaR": "diamond"}

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

    if len(fe_sharpes) > 0:
        tang_idx = np.argmax(fe_sharpes)
        tang_vol, tang_ret, tang_sh = fe_vols[tang_idx], fe_rets[tang_idx], fe_sharpes[tang_idx]
        fig_fe.add_trace(go.Scatter(
            x=[tang_vol], y=[tang_ret], mode="markers+text",
            marker=dict(size=22, color=COLORS["gold"], symbol="star",
                        line=dict(color=COLORS["dark_green"], width=2)),
            text=["Portefeuille<br>Tangent"], textposition="top right",
            textfont=dict(size=10, color=COLORS["gold"], family="Arial Black"),
            name=f"Tangent (Sharpe={tang_sh:.3f})",
        ))
        if show_cml:
            slope_cml = (tang_ret/100 - RF) / (tang_vol/100 + 1e-8)
            vol_end   = tang_vol * 1.5
            vol_cml   = np.array([0, tang_vol, vol_end])
            ret_cml   = (RF + slope_cml * vol_cml/100) * 100
            fig_fe.add_trace(go.Scatter(x=vol_cml[:2], y=ret_cml[:2], mode="lines",
                line=dict(color=COLORS["gold"], width=2.5), name=f"CML (Rf={RF*100:.2f}%)", hoverinfo="skip"))
            fig_fe.add_trace(go.Scatter(x=vol_cml[1:], y=ret_cml[1:], mode="lines",
                line=dict(color=COLORS["gold"], width=1.5, dash="dot"), showlegend=False, hoverinfo="skip"))
            fig_fe.add_trace(go.Scatter(x=[0], y=[RF*100], mode="markers+text",
                marker=dict(size=10, color=COLORS["gold"], symbol="circle", line=dict(color="white", width=1.5)),
                text=[f"Rf={RF*100:.2f}%"], textposition="top right", textfont=dict(size=9, color=COLORS["gold"]),
                showlegend=False, hoverinfo="skip"))

    if show_indiv:
        for nom in NOMS:
            m   = META[nom]
            col = COLORS["mid_green"] if m.get("alpha_j", 0) > 0 else COLORS["red"]
            fig_fe.add_trace(go.Scatter(
                x=[m["vol"]], y=[m["perf"]], mode="markers+text",
                marker=dict(size=9, color=col, symbol="circle", opacity=0.75, line=dict(color="white", width=1)),
                text=[nom[:12]], textposition="top center", textfont=dict(size=7.5, color=COLORS["dark_green"]),
                name=nom[:20], showlegend=False,
                hovertemplate=(f"<b>{nom}</b><br>Vol: {m['vol']:.2f}%<br>Perf: {m['perf']:.2f}%<br>"
                               f"Sharpe: {m['sharpe']:.3f}<br>VaR: {m['var99']:.3f}%<extra>OPCVM</extra>"),
            ))

    fig_fe.add_trace(go.Scatter(
        x=[stats_ref["vol"]], y=[stats_ref["perf"]], mode="markers+text",
        marker=dict(size=18, color=COLORS["gray"], symbol="square", line=dict(color="white", width=2)),
        text=["Actuel"], textposition="top center",
        textfont=dict(size=10, color=COLORS["gray"], family="Arial Black"), name="Portefeuille Actuel",
    ))
    fig_fe.add_trace(go.Scatter(
        x=[stats_cur["vol"]], y=[stats_cur["perf"]], mode="markers+text",
        marker=dict(size=22, color=COLORS["light_green"], symbol="star", line=dict(color=COLORS["dark_green"], width=2.5)),
        text=["Votre portefeuille"], textposition="top left",
        textfont=dict(size=10, color=COLORS["mid_green"], family="Arial Black"), name="Portefeuille Courant",
    ))
    for m_name, s in optim_points.items():
        fig_fe.add_trace(go.Scatter(
            x=[s["vol"]], y=[s["perf"]], mode="markers+text",
            marker=dict(size=16, color=col_opt[m_name], symbol=markers_m[m_name], line=dict(color="white", width=1.5)),
            text=[m_name[:12]], textposition="bottom center",
            textfont=dict(size=8.5, color=col_opt[m_name], family="Arial Black"), name=m_name,
        ))

    fig_fe.update_layout(
        height=650, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(244,247,251,0.7)",
        xaxis=dict(title="Volatilité Annualisée (%)", gridcolor="#E2E8F0", zeroline=False),
        yaxis=dict(title="Performance Annualisée (%)", gridcolor="#E2E8F0", zeroline=False),
        legend=dict(orientation="v", x=1.12, y=1, bgcolor="rgba(255,255,255,0.92)",
                    bordercolor=COLORS["light_green"], borderwidth=1, font=dict(size=9)),
        margin=dict(t=20, b=40, l=55, r=220), font=dict(family="Calibri"), hovermode="closest",
    )
    st.plotly_chart(fig_fe, use_container_width=True)

    st.markdown('<div class="section-title">📊 Résumé des 3 Méthodes</div>', unsafe_allow_html=True)
    rows_m = []
    for m_name, s in optim_points.items():
        rows_m.append({
            "Méthode": m_name, "Performance (%)": f"{s['perf']:.2f}%",
            "Volatilité (%)": f"{s['vol']:.2f}%", "Sharpe": f"{s['sharpe']:.3f}",
            "Sortino": f"{s['sortino']:.3f}", "VaR 99% (j)": f"{s['var99']:.4f}%",
            "CVaR 99% (j)": f"{s['cvar99']:.4f}%", "Drawdown Max (%)": f"{s['dd_max']:.2f}%",
        })
    st.dataframe(pd.DataFrame(rows_m).set_index("Méthode"), use_container_width=True, height=160)

    st.markdown('<div class="section-title">📐 Pondérations Optimisées par Méthode</div>', unsafe_allow_html=True)
    w_par_methode = {}
    for m in METHODES:
        w_par_methode[m] = optimiser(m, w_ref, R_GLOBAL, META, NOMS)
    rows_w = []
    for i, nom in enumerate(NOMS):
        row = {"OPCVM": nom[:22], "Actuel (%)": f"{w_ref[i]*100:.1f}%"}
        for m in METHODES:
            row[f"{m} (%)"] = f"{w_par_methode[m][i]*100:.1f}%"
        rows_w.append(row)
    df_w = pd.DataFrame(rows_w).set_index("OPCVM")
    total_row = {"Actuel (%)": "100.0%"}
    for m in METHODES:
        total_row[f"{m} (%)"] = f"{w_par_methode[m].sum()*100:.1f}%"
    df_w_full = pd.concat([df_w, pd.DataFrame([total_row], index=["∑ Total"])])
    st.dataframe(df_w_full, use_container_width=True, height=560)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — ANALYSE DÉTAILLÉE
# ══════════════════════════════════════════════════════════════════
with tab3:
    source_label = "réelles" if DATA_SOURCE == "réel" else "corrigées (synthétique)"
    st.markdown(f'<div class="section-title">📋 Indicateurs Individuels — données {source_label}</div>', unsafe_allow_html=True)
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
            "Beta (β)":              f"{m.get('beta', 0):.2f}",
            "Alpha Jensen (%)":      f"{m.get('alpha_j', 0):+.2f}%",
            "Sharpe":                f"{m['sharpe']:.3f}",
            "Sortino":               f"{m['sortino']:.3f}",
            "VaR 99% (%)":           f"{m['var99']:.3f}%",
            "CVaR 99% (%)":          f"{m['cvar99']:.3f}%",
            "Tracking Error (%)":    f"{m.get('te', 0):.3f}%",
            "Information Ratio":     f"{m.get('ir', 0):.3f}",
            "Drawdown Max (%)":      f"{m['dd']:.2f}%",
        })
    st.dataframe(pd.DataFrame(rows_d).set_index("OPCVM"), use_container_width=True, height=500)

    st.markdown('<div class="section-title">🔵 Alpha Jensen vs Sharpe (Taille = Poids Courant)</div>', unsafe_allow_html=True)
    fig_sc = go.Figure()
    for i, nom in enumerate(NOMS):
        m  = META[nom]
        wp = w_norm[i] * 100
        col = COLORS["mid_green"] if m.get("alpha_j", 0) > 0 else COLORS["red"]
        fig_sc.add_trace(go.Scatter(
            x=[m.get("alpha_j", 0)], y=[m["sharpe"]], mode="markers+text",
            marker=dict(size=max(wp/2, 8), color=col, opacity=0.7, line=dict(color="white", width=1)),
            text=[nom[:14]], textposition="top center", textfont=dict(size=9, color=COLORS["dark_green"]),
            name=nom[:20], showlegend=False,
            hovertemplate=f"<b>{nom}</b><br>α: {m.get('alpha_j',0):+.2f}%<br>Sharpe: {m['sharpe']:.3f}<br>Poids: {wp:.1f}%<extra></extra>",
        ))
    fig_sc.add_vline(x=0, line_dash="dot", line_color=COLORS["red"],  line_width=1.5)
    fig_sc.add_hline(y=2, line_dash="dot", line_color=COLORS["navy"], line_width=1.5)
    fig_sc.update_layout(
        height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(244,247,251,0.7)",
        xaxis=dict(title="Alpha Jensen (%)", gridcolor="#E2E8F0"),
        yaxis=dict(title="Sharpe Ratio",     gridcolor="#E2E8F0"),
        margin=dict(t=30, b=40, l=50, r=80),
    )
    st.plotly_chart(fig_sc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 4 — COMPARAISON MÉTHODES
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">⚖️ Comparaison des 3 Méthodes</div>', unsafe_allow_html=True)

    with st.spinner("Calcul..."):
        w_optimises = {}
        for m in METHODES:
            w_optimises[m] = optimiser(m, w_ref, R_GLOBAL, META, NOMS)
            if m not in optim_points:
                optim_points[m] = calcul_stats(w_optimises[m], R_GLOBAL)

    col_h, col_pv = st.columns([1.3, 1.7])

    with col_h:
        st.markdown("#### 📊 Matrice des Poids (%)")
        noms_short  = [n[:14] for n in NOMS]
        methodes_h  = ["Actuel"] + list(w_optimises.keys())
        w_matrix    = np.vstack([w_ref*100, *[w_optimises[m]*100 for m in w_optimises]])
        fig_hm = go.Figure(go.Heatmap(
            z=w_matrix, x=noms_short, y=methodes_h, colorscale="Greens",
            text=np.round(w_matrix, 1), texttemplate="%{text}%",
            textfont=dict(size=8, color="black"),
            hovertemplate="<b>%{y}</b> — %{x}<br>Poids: %{z:.1f}%<extra></extra>",
            colorbar=dict(title="Poids (%)", thickness=15, len=0.8), zmin=0, zmax=35,
        ))
        fig_hm.update_layout(
            height=420, paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
            yaxis=dict(tickfont=dict(size=9)), margin=dict(t=30, b=80, l=100, r=20),
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
        ))
        fig_pv.update_layout(
            height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(244,247,251,0.7)",
            xaxis=dict(title="Volatilité Annualisée (%)", gridcolor="#E2E8F0"),
            yaxis=dict(title="Performance Annualisée (%)", gridcolor="#E2E8F0"),
            margin=dict(t=20, b=30, l=50, r=30), hovermode="closest",
        )
        st.plotly_chart(fig_pv, use_container_width=True)

    st.markdown('<div class="section-title">📉 Comparaison des Mesures de Risque</div>', unsafe_allow_html=True)
    noms_ptf  = ["Actuel"] + list(optim_points.keys())
    vars_ptf  = [stats_ref["var99"]]  + [optim_points[m]["var99"]  for m in optim_points]
    cvars_ptf = [stats_ref["cvar99"]] + [optim_points[m]["cvar99"] for m in optim_points]
    cols_bar2 = [COLORS["gray"], COLORS["mid_green"], COLORS["navy"], COLORS["red"]]
    fig_var = go.Figure()
    fig_var.add_trace(go.Bar(name="VaR 99%", x=noms_ptf, y=vars_ptf,
        marker_color=cols_bar2, opacity=0.85,
        text=[f"{v:.4f}%" for v in vars_ptf], textposition="inside", textfont=dict(color="white", size=10)))
    fig_var.add_trace(go.Bar(name="CVaR 99%", x=noms_ptf, y=cvars_ptf,
        marker_color=cols_bar2, opacity=0.45,
        text=[f"{v:.4f}%" for v in cvars_ptf], textposition="inside", textfont=dict(color="white", size=9)))
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
    fig_sh.add_trace(go.Bar(name="Sharpe", x=noms_ptf, y=sharpes, marker_color=cols_bar2, opacity=0.85,
        text=[f"{v:.3f}" for v in sharpes], textposition="outside"))
    fig_sh.add_trace(go.Bar(name="Sortino", x=noms_ptf, y=sortinos, marker_color=cols_bar2, opacity=0.45,
        text=[f"{v:.3f}" for v in sortinos], textposition="outside"))
    fig_sh.update_layout(
        barmode="group", height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(244,247,251,0.7)",
        yaxis=dict(title="Ratio", gridcolor="#E2E8F0", range=[0, max(sharpes+sortinos)*1.15]),
        legend=dict(orientation="h", x=0, y=1.12), margin=dict(t=40, b=30, l=50, r=30),
    )
    st.plotly_chart(fig_sh, use_container_width=True)


st.markdown("---")
st.markdown(f"""
<div style="text-align:center;padding:20px;">
  <p style="color:{COLORS['gray']};font-size:0.75rem;margin:0;">
    <b>OPCVM Portfolio Dashboard v4</b> — {obs_label} · 3 méthodes : Min Variance · Max Sharpe · Min CVaR
    <br><span style="color:{COLORS['gold']}">VaR CDG RENDEMENT et EMERGENCE SERENITE corrigées (hebdo/√5)</span>
  </p>
  <p style="color:{COLORS['light_green']};font-size:0.8rem;margin:8px 0 0;font-weight:600;">
    © 2026 · Ben said Raydae
  </p>
</div>
""", unsafe_allow_html=True)
