"""
OPCVM Portfolio Dashboard — Streamlit
Données réelles extraites des images fournies
Optimisation Markowitz avec vraies valeurs marché et benchmarks
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG PAGE
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="OPCVM Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette ──────────────────────────────────────────────────────────────────
C = {
    "dark":   "#1A3C2E",
    "mid":    "#2C5F2D",
    "light":  "#97BC62",
    "mint":   "#D4EDDA",
    "navy":   "#1E3A5F",
    "red":    "#C0392B",
    "gold":   "#F0C040",
    "gray":   "#64748B",
    "lgray":  "#E2E8F0",
    "bg":     "#F5F9F6",
    "white":  "#FFFFFF",
}

st.markdown(f"""
<style>
  .stApp {{ background-color:{C['bg']}; }}

  [data-testid="stSidebar"] {{
      background-color:{C['dark']};
  }}
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] div {{
      color:white !important;
  }}

  .hdr {{
      background:linear-gradient(135deg,{C['dark']},{C['mid']});
      padding:16px 22px; border-radius:10px; margin-bottom:18px;
      border-left:6px solid {C['light']};
  }}
  .hdr h1 {{ color:white; margin:0; font-size:1.5rem; }}
  .hdr p  {{ color:{C['mint']}; margin:3px 0 0; font-size:0.85rem; }}

  .kcard {{
      background:white; border-radius:9px; padding:12px 14px;
      border-left:5px solid {C['mid']};
      box-shadow:0 2px 8px rgba(0,0,0,0.06); margin-bottom:8px;
  }}
  .kcard.r {{ border-left-color:{C['red']}; }}
  .kcard.g {{ border-left-color:{C['gold']}; }}
  .kcard.n {{ border-left-color:{C['navy']}; }}
  .kcard h3 {{ margin:0; font-size:1.45rem; color:{C['dark']}; font-weight:700; }}
  .kcard p  {{ margin:2px 0 0; font-size:0.76rem; color:{C['gray']}; }}
  .kcard .dl {{ font-size:0.72rem; font-weight:600; margin-top:3px; }}
  .dpos {{ color:{C['mid']}; }} .dneg {{ color:{C['red']}; }}

  .stitle {{
      background:{C['mid']}; color:white; padding:6px 14px;
      border-radius:5px; font-weight:600; font-size:0.9rem;
      margin:12px 0 8px;
  }}
  .sumok  {{ background:{C['mint']}; border-left:4px solid {C['mid']};
             padding:7px 11px; border-radius:5px; font-size:0.82rem;
             color:{C['dark']}; }}
  .sumwrn {{ background:#FFF3CD; border-left:4px solid {C['gold']};
             padding:7px 11px; border-radius:5px; font-size:0.82rem;
             color:#856404; }}

  .opt-box {{
      background:white; border-radius:10px; padding:16px;
      border:2px solid {C['light']}; margin-bottom:12px;
      box-shadow:0 3px 10px rgba(0,0,0,0.07);
  }}
  .opt-box h4 {{ color:{C['dark']}; margin:0 0 8px; font-size:1rem; }}
  .opt-box .badge {{
      display:inline-block; background:{C['light']};
      color:{C['dark']}; padding:2px 8px; border-radius:12px;
      font-size:0.75rem; font-weight:700; margin-bottom:6px;
  }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DONNÉES RÉELLES (extraites des images)
# ══════════════════════════════════════════════════════════════════════════════

OPCVM_DATA = {
    "FCP AFG Gov Bond Fund": {
        "cat": "OMLT", "bench": "MBI GLOBAL",
        "val_mad": 106_870_060.2, "poids": 8.44,
        "perf": 5.74,  "alpha": 1.20,  "sharpe": 1.63,
        "vol":  2.36,  "beta":  0.79,  "alpha_j": -0.82,
        "sortino": 2.31, "te": 1.50,   "ir": -1.29,
        "dd": -2.71,   "var99": -0.384,
    },
    "AD Balanced Fund": {
        "cat": "Diversifié", "bench": "MASI 50% + MBI MLT 50%",
        "val_mad": 37_334_178, "poids": 2.95,
        "perf": 9.58,  "alpha": -2.33, "sharpe": 1.93,
        "vol":  32.49, "beta":  1.35,  "alpha_j": -49.55,
        "sortino": 2.88, "te": 10.52,  "ir": -0.89,
        "dd": -8.87,   "var99": -1.906,
    },
    "AFG Optimal Fund": {
        "cat": "Diversifié", "bench": "MBI GLOBAL 70% + MASI 30%",
        "val_mad": 57_098_944, "poids": 4.51,
        "perf": 12.22, "alpha": 1.57,  "sharpe": 1.23,
        "vol":  8.79,  "beta":  1.54,  "alpha_j": -3.29,
        "sortino": 1.46, "te": 3.52,   "ir": 0.49,
        "dd": -7.75,   "var99": -1.573,
    },
    "CDG Izdihar": {
        "cat": "Diversifié", "bench": "MBI GLOBAL 77%",
        "val_mad": 57_119_568.66, "poids": 4.51,
        "perf": 8.98,  "alpha": 4.39,  "sharpe": 2.34,
        "vol":  24.76, "beta":  2.85,  "alpha_j": -15.01,
        "sortino": 3.97, "te": 22.95,  "ir": 1.13,
        "dd": -6.71,   "var99": -1.293,
    },
    "AD Select Bank": {
        "cat": "OCT", "bench": "MBI CT 100%",
        "val_mad": 41_172_607.95, "poids": 3.25,
        "perf": 3.39,  "alpha": 0.90,  "sharpe": 2.44,
        "vol":  0.55,  "beta":  0.09,  "alpha_j": 0.69,
        "sortino": 3.23, "te": 2.06,   "ir": -2.44,
        "dd": -0.19,   "var99": -0.081,
    },
    "Alpha Banques Fund": {
        "cat": "OCT", "bench": "MBI CT 100%",
        "val_mad": 90_107_416.41, "poids": 7.11,
        "perf": 3.12,  "alpha": 0.66,  "sharpe": 2.16,
        "vol":  0.50,  "beta":  0.08,  "alpha_j": 0.46,
        "sortino": 2.90, "te": 2.06,   "ir": -2.57,
        "dd": -0.18,   "var99": -0.078,
    },
    "Alpha Secure Fund": {
        "cat": "OMLT", "bench": "MBI GLOBAL 100%",
        "val_mad": 105_197_655.44, "poids": 8.30,
        "perf": 4.89,  "alpha": 0.35,  "sharpe": 1.73,
        "vol":  1.71,  "beta":  0.47,  "alpha_j": 0.24,
        "sortino": 2.18, "te": 0.95,   "ir": -1.73,
        "dd": -1.59,   "var99": -0.274,
    },
    "CAM Oblibanques": {
        "cat": "OMLT", "bench": "85% MBI CT + 15% MBI MT",
        "val_mad": 106_365_936.8, "poids": 8.40,
        "perf": 3.17,  "alpha": 0.60,  "sharpe": 2.04,
        "vol":  0.55,  "beta":  0.09,  "alpha_j": 0.53,
        "sortino": 2.84, "te": 0.25,   "ir": -2.52,
        "dd": -0.17,   "var99": -0.074,
    },
    "FCP CDG Rendement": {
        "cat": "OMLT", "bench": "95% MBI MLT + 5% MASI",
        "val_mad": 106_492_417.2, "poids": 8.41,
        "perf": 5.06,  "alpha": -0.28, "sharpe": 4.58,
        "vol":  5.05,  "beta":  0.66,  "alpha_j": 6.76,
        "sortino": 6.49, "te": 3.48,   "ir": -0.38,
        "dd": -1.69,   "var99": -0.283,
    },
    "FCP Oblig CT": {
        "cat": "OMLT", "bench": "75% MBI CT + 25% MBI MT",
        "val_mad": 124_132_590.35, "poids": 9.80,
        "perf": 3.23,  "alpha": 0.66,  "sharpe": 2.18,
        "vol":  0.55,  "beta":  0.09,  "alpha_j": 0.61,
        "sortino": 2.69, "te": 0.22,   "ir": -2.48,
        "dd": -0.19,   "var99": -0.096,
    },
    "CDG Tawfir": {
        "cat": "OMLT", "bench": "MBI GLOBAL 100%",
        "val_mad": 109_954_679.94, "poids": 8.68,
        "perf": 5.01,  "alpha": 0.47,  "sharpe": 1.62,
        "vol":  1.91,  "beta":  0.61,  "alpha_j": -0.52,
        "sortino": 2.08, "te": 1.16,   "ir": 0.42,
        "dd": -1.69,   "var99": -0.344,
    },
    "FCP Emergence Serenite": {
        "cat": "OMLT", "bench": "MBI GLOBAL 100%",
        "val_mad": 218_687_186.75, "poids": 17.26,
        "perf": 4.63,  "alpha": 0.06,  "sharpe": 4.31,
        "vol":  4.83,  "beta":  0.85,  "alpha_j": 3.17,
        "sortino": 5.79, "te": 2.34,   "ir": 0.12,
        "dd": -1.66,   "var99": -0.616,
    },
    "FCP Capital Trust Equilibre": {
        "cat": "Diversifié", "bench": "70% MBI GLOBAL + 30% MASI",
        "val_mad": 55_113_276, "poids": 4.35,
        "perf": 11.49, "alpha": 0.84,  "sharpe": 1.22,
        "vol":  8.22,  "beta":  1.30,  "alpha_j": -1.79,
        "sortino": 1.45, "te": 4.27,   "ir": 0.23,
        "dd": -7.63,   "var99": -1.710,
    },
    "FCP AD Yield Fund": {
        "cat": "OCT", "bench": "MBI CT 100%",
        "val_mad": 51_223_513.15, "poids": 4.04,
        "perf": 3.11,  "alpha": 0.59,  "sharpe": 1.81,
        "vol":  0.59,  "beta":  0.09,  "alpha_j": 0.43,
        "sortino": 2.45, "te": 0.36,   "ir": 1.83,
        "dd": -0.21,   "var99": -0.089,
    },
}

NOMS        = list(OPCVM_DATA.keys())
N           = len(NOMS)
RF          = 0.0225
RF_D        = RF / 252
TOTAL_MAD   = sum(d["val_mad"] for d in OPCVM_DATA.values())  # ~1 266 870 031 MAD

CATEGORIES = {
    "🏦 Oblig. MLT": [n for n in NOMS if OPCVM_DATA[n]["cat"] == "OMLT"],
    "📈 Diversifié":  [n for n in NOMS if OPCVM_DATA[n]["cat"] == "Diversifié"],
    "💰 Oblig. CT":   [n for n in NOMS if OPCVM_DATA[n]["cat"] == "OCT"],
}


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION RENDEMENTS  (basée sur les vraies stats)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def build_returns():
    """
    Génère 247 rendements journaliers cohérents avec :
      - perf annuelle réelle
      - volatilité réelle
      - bêta réel (corrélation au marché)
    """
    np.random.seed(2025)
    T   = 247
    mu  = np.array([OPCVM_DATA[n]["perf"] / 100 / 252     for n in NOMS])
    sig = np.array([OPCVM_DATA[n]["vol"]  / 100 / np.sqrt(252) for n in NOMS])
    bet = np.array([OPCVM_DATA[n]["beta"] for n in NOMS])

    r_mkt = np.random.normal(0.0003, 0.009, T)
    R = np.zeros((T, N))
    for i in range(N):
        idio_var = max(sig[i]**2 - bet[i]**2 * np.var(r_mkt), 1e-10)
        eps = np.random.normal(0, np.sqrt(idio_var), T)
        R[:, i] = mu[i] + bet[i] * (r_mkt - r_mkt.mean()) + eps
        # Recalage exact mu et sigma
        R[:, i] = (R[:, i] - R[:, i].mean() + mu[i])
        cur_sig  = R[:, i].std()
        if cur_sig > 1e-12:
            R[:, i] = (R[:, i] - mu[i]) / cur_sig * sig[i] + mu[i]
    return R

R_G = build_returns()


# ══════════════════════════════════════════════════════════════════════════════
# CALCULS PORTEFEUILLE
# ══════════════════════════════════════════════════════════════════════════════
def ptf_stats(w):
    r      = R_G @ w
    perf   = float(r.mean()  * 252 * 100)
    vol    = float(r.std()   * np.sqrt(252) * 100)
    sharpe = float((perf/100 - RF) / (vol/100)) if vol > 0 else 0.0
    r_neg  = r[r < RF_D]
    dv     = float(r_neg.std() * np.sqrt(252)) if len(r_neg) > 1 else vol/100
    sortino= float((perf/100 - RF) / dv) if dv > 0 else 0.0
    var99  = float(np.percentile(r, 1, method="lower") * 100)
    cvar99 = float(r[r < var99/100].mean() * 100) if (r < var99/100).any() else var99
    cum    = np.cumprod(1 + r)
    roll   = np.maximum.accumulate(cum)
    dd_max = float(((cum - roll) / roll).min() * 100)
    val_mad= TOTAL_MAD * float(np.sum(w))  # reste constant (poids normalisés)
    return dict(perf=perf, vol=vol, sharpe=sharpe, sortino=sortino,
                var99=var99, cvar99=cvar99, dd_max=dd_max)


def run_optim(methode: str, w0: np.ndarray,
              w_min=0.01, w_max=0.25) -> np.ndarray:
    """
    Markowitz SLSQP — 3 objectifs :
      1. Min Variance  → minimise wᵀΣw
      2. Max Sharpe    → maximise (μ−Rf)/σ  (portefeuille tangent)
      3. Min CVaR      → minimise CVaR 99%  + contrainte perf ≥ 95% actuel
    """
    mu  = R_G.mean(axis=0) * 252
    cov = np.cov(R_G.T)    * 252
    bounds = [(w_min, w_max)] * N
    eq_sum = {"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}

    def cvar_fn(w):
        r = R_G @ w
        v = np.percentile(r, 1, method="lower")
        tail = r[r <= v]
        return float(-tail.mean()) if len(tail) > 0 else float(-v)

    if methode == "Min Variance":
        obj  = lambda w: float(w @ cov @ w)
        csts = [eq_sum]

    elif methode == "Max Sharpe":
        obj  = lambda w: float(-(mu @ w - RF) / max(np.sqrt(w @ cov @ w), 1e-9))
        csts = [eq_sum]

    elif methode == "Min CVaR":
        ret_floor = float(mu @ w0) * 0.95
        obj  = cvar_fn
        csts = [eq_sum,
                {"type": "ineq", "fun": lambda w: float(mu @ w) - ret_floor}]

    elif methode == "Max Perf / Min Risque":
        # Maximise μ − λ·σ²  avec λ calibré sur Sharpe
        lam  = 1.5
        obj  = lambda w: float(-(mu @ w) + lam * (w @ cov @ w))
        csts = [eq_sum]

    else:
        return w0

    rng  = np.random.default_rng(42)
    best = None
    starts = [np.clip(w0.copy(), w_min, w_max)]
    for _ in range(8):
        w_r = np.clip(rng.dirichlet(np.ones(N)), w_min, w_max)
        starts.append(w_r / w_r.sum())

    for s in starts:
        s = np.clip(s, w_min, w_max); s /= s.sum()
        try:
            res = minimize(obj, s, method="SLSQP", bounds=bounds,
                           constraints=csts,
                           options={"maxiter": 2000, "ftol": 1e-12})
            if best is None or (res.fun < best.fun):
                best = res
        except Exception:
            pass

    if best is not None:
        w = np.clip(best.x, w_min, w_max)
        return w / w.sum()
    return w0


@st.cache_data(show_spinner=False)
def compute_frontier(n_pts=200):
    """
    Frontière efficiente Markowitz :
    200 portefeuilles min-variance sur une grille de rendements cibles.
    Retourne (vols%, rets%, var99s%, sharpes, w_matrix)
    """
    mu    = R_G.mean(axis=0) * 252
    cov   = np.cov(R_G.T)    * 252
    bounds= [(0.01, 0.25)] * N
    obj   = lambda w: float(w @ cov @ w)

    r_lo = float(mu.min()) * 1.02
    r_hi = float(mu.max()) * 0.78
    tgts = np.linspace(r_lo, r_hi, n_pts)

    vols, rets, var99s, sharpes, ws = [], [], [], [], []
    rng = np.random.default_rng(0)

    for t in tgts:
        cst = [
            {"type": "eq", "fun": lambda w:      float(w.sum() - 1.0)},
            {"type": "eq", "fun": lambda w, tt=t: float(mu @ w - tt)},
        ]
        best = None
        for _ in range(5):
            w0 = np.clip(rng.dirichlet(np.ones(N)), 0.01, 0.25)
            w0 /= w0.sum()
            try:
                res = minimize(obj, w0, method="SLSQP", bounds=bounds,
                               constraints=cst,
                               options={"maxiter": 800, "ftol": 1e-10})
                if res.success and (best is None or res.fun < best.fun):
                    best = res
            except Exception:
                pass
        if best and best.success:
            w   = np.clip(best.x, 0, None); w /= w.sum()
            r_p = R_G @ w
            vol = float(np.sqrt(w @ cov @ w) * 100)
            ret = float(mu @ w * 100)
            v99 = float(np.percentile(r_p, 1, method="lower") * 100)
            sh  = float((ret/100 - RF) / (vol/100)) if vol > 0 else 0.0
            vols.append(vol); rets.append(ret)
            var99s.append(v99); sharpes.append(sh); ws.append(w)

    return (np.array(vols), np.array(rets),
            np.array(var99s), np.array(sharpes), ws)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE  — Fix bug Streamlit widget/session_state
# ══════════════════════════════════════════════════════════════════════════════
if "poids_init" not in st.session_state:
    st.session_state["poids_init"] = {n: float(OPCVM_DATA[n]["poids"]) for n in NOMS}

if "optim_done" not in st.session_state:
    st.session_state["optim_done"] = False

if "optim_result" not in st.session_state:
    st.session_state["optim_result"] = None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown(f"""
<div style="background:{C['mid']};padding:11px 14px;border-radius:8px;
     margin-bottom:10px;border-left:4px solid {C['light']}">
  <h2 style="color:white;margin:0;font-size:1.05rem">⚖️ Pondérations OPCVM</h2>
  <p style="color:{C['mint']};font-size:0.78rem;margin:3px 0 0">
     Valeur totale : {TOTAL_MAD/1e9:.3f} Md MAD · 14 fonds</p>
</div>
""", unsafe_allow_html=True)

# ── Paramètres Markowitz ─────────────────────────────────────────────────────
with st.sidebar.expander("⚙️ Paramètres Markowitz", expanded=True):
    methode_sel = st.selectbox(
        "Méthode d'optimisation",
        ["Min Variance", "Max Sharpe", "Min CVaR", "Max Perf / Min Risque"],
        index=1,
        help=(
            "• Min Variance : risque minimal absolu\n"
            "• Max Sharpe : meilleur rendement ajusté au risque\n"
            "• Min CVaR : minimise les pertes extrêmes (1% pires jours)\n"
            "• Max Perf / Min Risque : compromis performance/variance"
        ),
    )
    w_min_pct = st.slider("Poids minimum par OPCVM (%)", 0.5, 5.0, 1.0, 0.5)
    w_max_pct = st.slider("Poids maximum par OPCVM (%)", 10.0, 40.0, 25.0, 1.0)

# ── Boutons ──────────────────────────────────────────────────────────────────
col_sb1, col_sb2 = st.sidebar.columns(2)
do_reset = col_sb1.button("🔄 Réinit.", use_container_width=True)
do_optim = col_sb2.button("⚡ Optimiser", use_container_width=True, type="primary")

# Traitement AVANT les sliders
if do_reset:
    st.session_state["poids_init"] = {n: float(OPCVM_DATA[n]["poids"]) for n in NOMS}
    st.session_state["optim_result"] = None
    st.session_state["optim_done"]   = False
    st.rerun()

if do_optim:
    # Lire poids courants depuis session_state des sliders
    w_cur = np.array([
        st.session_state.get(f"w_{n}", float(OPCVM_DATA[n]["poids"])) / 100
        for n in NOMS
    ])
    w_cur = np.clip(w_cur, w_min_pct/100, w_max_pct/100)
    w_cur /= w_cur.sum()

    with st.spinner(f"⚡ Optimisation Markowitz — {methode_sel}…"):
        w_opt = run_optim(methode_sel, w_cur, w_min_pct/100, w_max_pct/100)

    new_p = {n: round(float(w_opt[i]) * 100, 2) for i, n in enumerate(NOMS)}
    st.session_state["poids_init"]   = new_p
    st.session_state["optim_result"] = {"methode": methode_sel, "poids": new_p}
    st.session_state["optim_done"]   = True
    st.rerun()

# ── Sliders ──────────────────────────────────────────────────────────────────
poids_user = {}
for cat, fonds_list in CATEGORIES.items():
    with st.sidebar.expander(cat, expanded=True):
        for nom in fonds_list:
            init = st.session_state["poids_init"].get(nom, float(OPCVM_DATA[nom]["poids"]))
            val  = st.slider(
                nom[:24],
                min_value=0.0, max_value=w_max_pct,
                value=float(np.clip(init, 0.0, w_max_pct)),
                step=0.1, key=f"w_{nom}", format="%.1f%%",
            )
            poids_user[nom] = val

# Somme
total_w = sum(poids_user.values())
delta_w = total_w - 100.0
if abs(delta_w) < 0.15:
    st.sidebar.markdown(f'<div class="sumok">✅ Somme = {total_w:.1f}%</div>',
                        unsafe_allow_html=True)
else:
    sg = "+" if delta_w > 0 else ""
    st.sidebar.markdown(
        f'<div class="sumwrn">⚠️ Somme = {total_w:.1f}% '
        f'({sg}{delta_w:.1f}pp)<br><small>→ normalisé auto.</small></div>',
        unsafe_allow_html=True,
    )

st.sidebar.markdown(f"""
<div style="margin-top:10px;padding:6px;background:{C['dark']};
     border-radius:6px;font-size:0.72rem;color:{C['gray']}">
  Rf = 2.25% · 247 obs. · 14 OPCVM · 1.267 Md MAD
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# POIDS NORMALISÉS
# ══════════════════════════════════════════════════════════════════════════════
w_raw  = np.array([poids_user[n] / 100 for n in NOMS])
w_norm = w_raw / w_raw.sum()
w_ref  = np.array([OPCVM_DATA[n]["poids"] / 100 for n in NOMS])

s_cur = ptf_stats(w_norm)
s_ref = ptf_stats(w_ref)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hdr">
  <h1>📊 OPCVM Portfolio Optimizer — Markowitz</h1>
  <p>Optimisation quantitative · Frontière Efficiente · Rf = 2.25% ·
     14 OPCVM · Valeur totale {TOTAL_MAD/1e9:.3f} Md MAD</p>
</div>
""", unsafe_allow_html=True)

# Bandeau résultat optimisation
if st.session_state["optim_done"] and st.session_state["optim_result"]:
    res_m = st.session_state["optim_result"]["methode"]
    st.success(
        f"✅ **Optimisation {res_m} appliquée** — "
        f"Sharpe : {s_cur['sharpe']:.3f} | "
        f"Perf : {s_cur['perf']:.2f}%/an | "
        f"VaR : {s_cur['var99']:.4f}%/j | "
        f"Vol : {s_cur['vol']:.2f}%/an"
    )


# ══════════════════════════════════════════════════════════════════════════════
# ONGLETS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Tableau de Bord",
    "🎯 Frontière Efficiente",
    "🔬 Optimisation Markowitz",
    "📋 OPCVM Détail",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — TABLEAU DE BORD
# ═══════════════════════════════════════════════════════════════
with tab1:

    def dh(val, ref, unit="%", inv=False):
        d   = val - ref
        good= (d < 0) if inv else (d > 0)
        sg  = "+" if d > 0 else ""
        cls = "dpos" if good else "dneg"
        return f'<span class="{cls}">{sg}{d:.2f}{unit} vs actuel</span>'

    kpis = [
        ("Perf. annuelle",  f"{s_cur['perf']:.2f}%",   s_cur["perf"],   s_ref["perf"],   "%", False, ""),
        ("Volatilité",      f"{s_cur['vol']:.2f}%",    s_cur["vol"],    s_ref["vol"],    "%", True,  "r"),
        ("Sharpe Ratio",    f"{s_cur['sharpe']:.3f}",  s_cur["sharpe"], s_ref["sharpe"], "",  False, "g"),
        ("Sortino",         f"{s_cur['sortino']:.3f}", s_cur["sortino"],s_ref["sortino"],"",  False, "g"),
        ("VaR 99%/jour",    f"{s_cur['var99']:.4f}%",  s_cur["var99"],  s_ref["var99"],  "%", True,  "r"),
        ("CVaR 99%/jour",   f"{s_cur['cvar99']:.4f}%", s_cur["cvar99"], s_ref["cvar99"], "%", True,  "r"),
        ("Drawdown Max",    f"{s_cur['dd_max']:.2f}%", s_cur["dd_max"], s_ref["dd_max"], "%", True,  "n"),
    ]
    cols_k = st.columns(len(kpis))
    for col, (lbl, vs, v, r_, u, inv, cls) in zip(cols_k, kpis):
        with col:
            st.markdown(f"""<div class="kcard {cls}">
              <p>{lbl}</p><h3>{vs}</h3>
              <div class="dl">{dh(v,r_,u,inv)}</div>
            </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown('<div class="stitle">🥧 Répartition courante</div>',
                    unsafe_allow_html=True)
        pie_col = [
            "#2C5F2D","#97BC62","#1A3C2E","#D4EDDA","#1E3A5F","#2980B9",
            "#F0C040","#E67E22","#C0392B","#8E44AD","#16A085","#F39C12",
            "#2ECC71","#3498DB",
        ]
        fig_p = go.Figure(go.Pie(
            labels=[n[:20] for n in NOMS], values=w_norm*100,
            textinfo="label+percent", textfont_size=8.5,
            marker=dict(colors=pie_col, line=dict(color="white",width=1.5)),
            hole=0.38,
        ))
        fig_p.update_layout(
            showlegend=False, margin=dict(t=5,b=5,l=5,r=5),
            height=320, paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text=f"<b>{total_w:.0f}%</b>",
                              x=0.5,y=0.5,font=dict(size=14,color=C["dark"]),
                              showarrow=False)],
        )
        st.plotly_chart(fig_p, use_container_width=True)

        # Valeur en MAD par OPCVM
        st.markdown('<div class="stitle">💰 Valeurs Marché (MAD)</div>',
                    unsafe_allow_html=True)
        val_df = pd.DataFrame({
            "OPCVM": [n[:22] for n in NOMS],
            "Poids %": [f"{w_norm[i]*100:.2f}%" for i in range(N)],
            "Valeur MAD": [f"{w_norm[i]*TOTAL_MAD:,.0f}" for i in range(N)],
        }).set_index("OPCVM")
        st.dataframe(val_df, use_container_width=True, height=280)

    with c2:
        st.markdown('<div class="stitle">📊 Poids courant vs actuel</div>',
                    unsafe_allow_html=True)
        short = [n[:18] for n in NOMS]
        delta_p = w_norm*100 - w_ref*100
        fig_b = go.Figure()
        fig_b.add_trace(go.Bar(
            name="Actuel", x=short, y=w_ref*100,
            marker_color=C["gray"], opacity=0.6,
        ))
        fig_b.add_trace(go.Bar(
            name="Courant", x=short, y=w_norm*100,
            marker_color=[C["mid"] if d >= 0 else C["red"] for d in delta_p],
            opacity=0.85,
        ))
        fig_b.add_hline(y=w_min_pct, line_dash="dot",
                        line_color=C["light"], line_width=1,
                        annotation_text=f"min {w_min_pct}%")
        fig_b.add_hline(y=w_max_pct, line_dash="dot",
                        line_color=C["red"],   line_width=1,
                        annotation_text=f"max {w_max_pct}%")
        fig_b.update_layout(
            barmode="group", height=290,
            margin=dict(t=5,b=75,l=30,r=5),
            legend=dict(orientation="h",y=1.04,x=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=f"rgba(244,247,251,0.5)",
            xaxis=dict(tickangle=-40, tickfont=dict(size=8)),
            yaxis=dict(title="Poids (%)", gridcolor=C["lgray"]),
        )
        st.plotly_chart(fig_b, use_container_width=True)

        # Perf cumulée
        st.markdown('<div class="stitle">📉 Performance cumulée simulée</div>',
                    unsafe_allow_html=True)
        rc = R_G @ w_norm; rr = R_G @ w_ref
        cc = (1+rc).cumprod()-1; cr = (1+rr).cumprod()-1
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(
            x=list(range(len(cr))), y=cr*100,
            name=f"Actuel ({cr[-1]*100:.2f}%)",
            line=dict(color=C["gray"],dash="dash",width=1.8), opacity=0.7,
        ))
        fig_c.add_trace(go.Scatter(
            x=list(range(len(cc))), y=cc*100,
            name=f"Courant ({cc[-1]*100:.2f}%)",
            line=dict(color=C["mid"],width=2.5),
        ))
        fig_c.add_hline(y=0, line_color=C["gray"], line_width=0.7)
        fig_c.update_layout(
            height=200, margin=dict(t=5,b=25,l=40,r=5),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=f"rgba(244,247,251,0.5)",
            legend=dict(orientation="h",y=1.08,x=0),
            yaxis=dict(title="Perf. cumulée (%)", gridcolor=C["lgray"]),
            xaxis=dict(title="Jours", gridcolor=C["lgray"]),
        )
        st.plotly_chart(fig_c, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — FRONTIÈRE EFFICIENTE
# ═══════════════════════════════════════════════════════════════
with tab2:

    st.markdown('<div class="stitle">🎯 Frontière Efficiente de Markowitz</div>',
                unsafe_allow_html=True)

    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        color_by = st.selectbox("Colorer par",
                                ["Sharpe Ratio","VaR 99%","Rendement"], index=0)
    with oc2:
        show_ind = st.checkbox("OPCVM individuels", True)
    with oc3:
        show_cml = st.checkbox("Capital Market Line", True)

    with st.spinner("Calcul de la frontière efficiente Markowitz (200 pts)…"):
        fe_v, fe_r, fe_va, fe_sh, fe_ws = compute_frontier(200)

    # 3 portefeuilles optimisés
    optim_pts = {}
    meth_col = {
        "Min Variance": C["mid"],
        "Max Sharpe":   C["navy"],
        "Min CVaR":     C["red"],
    }
    meth_sym = {
        "Min Variance": "square",
        "Max Sharpe":   "triangle-up",
        "Min CVaR":     "diamond",
    }
    with st.spinner("Calcul des portefeuilles optimaux…"):
        for m in meth_col:
            wo = run_optim(m, w_ref, w_min_pct/100, w_max_pct/100)
            optim_pts[m] = {"w": wo, "s": ptf_stats(wo)}

    # Couleur frontière
    cmap_cfg = {
        "Sharpe Ratio": (fe_sh, "Sharpe",     "RdYlGn",   False),
        "VaR 99%":      (fe_va, "VaR %/j",    "RdYlGn",   True),
        "Rendement":    (fe_r,  "Rend. %/an",  "Viridis",  False),
    }
    cvals, cbar_t, cscale, crev = cmap_cfg[color_by]

    fig_fe = go.Figure()

    # Zone atteignable
    if len(fe_v) > 1:
        fig_fe.add_trace(go.Scatter(
            x=np.concatenate([fe_v, fe_v[::-1]]),
            y=np.concatenate([fe_r,
                              np.full(len(fe_r), fe_r.min()-0.5)]),
            fill="toself",
            fillcolor="rgba(151,188,98,0.06)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))

    # Frontière colorée
    hover_fe = [f"Vol:{v:.2f}%  Perf:{r:.2f}%  VaR:{va:.4f}%  Sharpe:{sh:.3f}"
                for v,r,va,sh in zip(fe_v,fe_r,fe_va,fe_sh)]
    fig_fe.add_trace(go.Scatter(
        x=fe_v, y=fe_r, mode="markers",
        marker=dict(
            color=cvals, colorscale=cscale, reversescale=crev,
            size=6, opacity=0.80,
            colorbar=dict(title=dict(text=cbar_t,side="right"),
                          thickness=13, len=0.6, x=1.01),
            line=dict(width=0),
        ),
        name="Frontière efficiente",
        customdata=hover_fe,
        hovertemplate="%{customdata}<extra>Frontière efficiente</extra>",
    ))
    # Ligne de contour
    fig_fe.add_trace(go.Scatter(
        x=fe_v, y=fe_r, mode="lines",
        line=dict(color="rgba(44,95,45,0.20)", width=1.4),
        showlegend=False, hoverinfo="skip",
    ))

    # CML
    if show_cml and len(fe_sh) > 0:
        idx_tang = int(np.argmax(fe_sh))
        slope    = float(fe_sh[idx_tang])
        vr       = np.array([0.0, float(fe_v.max())*1.15])
        yr       = (RF + slope * vr/100) * 100
        fig_fe.add_trace(go.Scatter(
            x=vr, y=yr, mode="lines",
            name=f"CML  Sharpe tangent={slope:.2f}",
            line=dict(color=C["gold"], dash="dot", width=1.8), opacity=0.9,
        ))
        fig_fe.add_trace(go.Scatter(
            x=[0], y=[RF*100], mode="markers+text",
            marker=dict(size=9, color=C["gold"], symbol="star"),
            text=[f"Rf={RF*100:.2f}%"], textposition="middle right",
            textfont=dict(size=9, color=C["gold"]),
            showlegend=False,
        ))

    # OPCVM individuels
    if show_ind:
        for nom in NOMS:
            d   = OPCVM_DATA[nom]
            col = C["light"] if d["alpha_j"] > 0 else C["red"]
            fig_fe.add_trace(go.Scatter(
                x=[d["vol"]], y=[d["perf"]],
                mode="markers+text",
                marker=dict(size=7, color=col, opacity=0.65,
                            line=dict(color="white",width=0.6)),
                text=[nom[:13]], textposition="top right",
                textfont=dict(size=7, color=C["gray"]),
                showlegend=False,
                hovertemplate=(
                    f"<b>{nom}</b><br>"
                    f"Vol: {d['vol']:.2f}% | Perf: {d['perf']:.2f}%<br>"
                    f"Sharpe: {d['sharpe']:.2f} | α Jensen: {d['alpha_j']:+.2f}%"
                    "<extra></extra>"
                ),
            ))

    # Portefeuille actuel
    fig_fe.add_trace(go.Scatter(
        x=[s_ref["vol"]], y=[s_ref["perf"]],
        mode="markers+text",
        marker=dict(size=15, color=C["gray"], symbol="square",
                    line=dict(color="white",width=1.5)),
        text=["Actuel"], textposition="top center",
        textfont=dict(size=9,color=C["gray"],family="Arial Black"),
        name="Portefeuille Actuel",
        hovertemplate=(
            f"<b>Portefeuille Actuel</b><br>"
            f"Vol:{s_ref['vol']:.2f}% | Perf:{s_ref['perf']:.2f}%<br>"
            f"Sharpe:{s_ref['sharpe']:.3f} | VaR:{s_ref['var99']:.4f}%"
            "<extra></extra>"
        ),
    ))

    # Portefeuille courant (étoile dynamique)
    fig_fe.add_trace(go.Scatter(
        x=[s_cur["vol"]], y=[s_cur["perf"]],
        mode="markers+text",
        marker=dict(size=22, color=C["light"], symbol="star",
                    line=dict(color=C["dark"],width=2)),
        text=["Votre portefeuille"], textposition="top center",
        textfont=dict(size=10,color=C["mid"],family="Arial Black"),
        name="Portefeuille Courant",
        hovertemplate=(
            f"<b>Votre Portefeuille</b><br>"
            f"Vol:{s_cur['vol']:.2f}% | Perf:{s_cur['perf']:.2f}%<br>"
            f"Sharpe:{s_cur['sharpe']:.3f} | VaR:{s_cur['var99']:.4f}%<br>"
            f"Sortino:{s_cur['sortino']:.3f}"
            "<extra></extra>"
        ),
    ))

    # 3 méthodes
    for m, data in optim_pts.items():
        s = data["s"]
        fig_fe.add_trace(go.Scatter(
            x=[s["vol"]], y=[s["perf"]],
            mode="markers+text",
            marker=dict(size=14, color=meth_col[m], symbol=meth_sym[m],
                        line=dict(color="white",width=1.2)),
            text=[m.replace("Min ","").replace("Max ","")],
            textposition="bottom right",
            textfont=dict(size=8,color=meth_col[m]),
            name=m,
            hovertemplate=(
                f"<b>{m}</b><br>"
                f"Vol:{s['vol']:.2f}% | Perf:{s['perf']:.2f}%<br>"
                f"Sharpe:{s['sharpe']:.3f} | VaR:{s['var99']:.4f}%"
                "<extra></extra>"
            ),
        ))

    fig_fe.update_layout(
        height=590,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(244,247,251,0.65)",
        xaxis=dict(title="Volatilité annualisée (%)",
                   gridcolor=C["lgray"], zeroline=False,
                   showspikes=True, spikecolor=C["gray"],
                   spikedash="dot", spikethickness=1),
        yaxis=dict(title="Rendement annualisé (%)",
                   gridcolor=C["lgray"], zeroline=False,
                   showspikes=True, spikecolor=C["gray"],
                   spikedash="dot", spikethickness=1),
        legend=dict(orientation="v", x=1.14, y=1,
                    bgcolor="rgba(255,255,255,0.88)",
                    bordercolor=C["light"], borderwidth=1,
                    font=dict(size=9)),
        margin=dict(t=20,b=50,l=60,r=190),
        hovermode="closest",
        hoverlabel=dict(bgcolor="white",
                        bordercolor=C["mid"],
                        font=dict(size=11)),
    )
    st.plotly_chart(fig_fe, use_container_width=True)

    st.markdown(f"""
    <div style="background:white;border-radius:8px;padding:10px 15px;
                border:1px solid {C['lgray']};font-size:0.81rem;
                color:{C['dark']}">
      <b>Lecture :</b>&nbsp;
      ⭐ Votre portefeuille courant (se déplace en temps réel) &nbsp;·&nbsp;
      ◼ Portefeuille actuel &nbsp;·&nbsp;
      <span style="color:{C['mid']}">▪ Min Variance</span> &nbsp;·&nbsp;
      <span style="color:{C['navy']}">▲ Max Sharpe</span> &nbsp;·&nbsp;
      <span style="color:{C['red']}">◆ Min CVaR</span> &nbsp;·&nbsp;
      <span style="color:{C['light']}">● α Jensen &gt; 0</span> &nbsp;·&nbsp;
      <span style="color:{C['red']}">● α Jensen &lt; 0</span>
    </div>
    """, unsafe_allow_html=True)

    # Tableau comparatif
    st.markdown('<div class="stitle">📊 Comparaison des portefeuilles</div>',
                unsafe_allow_html=True)
    rows_c = []
    all_p  = {"Actuel": s_ref, "Courant": s_cur,
              **{m: d["s"] for m,d in optim_pts.items()}}
    for nm,s in all_p.items():
        rows_c.append({
            "Portefeuille": nm,
            "Perf %/an":  f"{s['perf']:.2f}%",
            "Vol %/an":   f"{s['vol']:.2f}%",
            "Sharpe":     f"{s['sharpe']:.3f}",
            "Sortino":    f"{s['sortino']:.3f}",
            "VaR 99%/j":  f"{s['var99']:.4f}%",
            "CVaR/j":     f"{s['cvar99']:.4f}%",
            "DD Max":     f"{s['dd_max']:.2f}%",
        })
    st.dataframe(pd.DataFrame(rows_c).set_index("Portefeuille"),
                 use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — OPTIMISATION MARKOWITZ DÉTAILLÉE
# ═══════════════════════════════════════════════════════════════
with tab3:

    st.markdown(f"""
    <div class="opt-box">
      <h4>📐 Principe de l'optimisation Markowitz</h4>
      <p style="color:{C['gray']};font-size:0.88rem">
        La théorie moderne du portefeuille (Markowitz, 1952) cherche la
        <b>combinaison optimale de poids</b> qui maximise le rendement pour un
        niveau de risque donné (ou minimise le risque pour un rendement cible).
        Le résultat est la <b>frontière efficiente</b> — ensemble des portefeuilles
        dominants. Aucun portefeuille en-dehors ne peut être meilleur sur les deux
        dimensions simultanément.
      </p>
      <div style="display:flex;gap:12px;margin-top:8px;flex-wrap:wrap">
        <div style="flex:1;min-width:200px;background:{C['bg']};
             padding:10px;border-radius:7px;border-left:3px solid {C['mid']}">
          <b style="color:{C['dark']}">Min Variance</b><br>
          <code style="font-size:0.8rem">min wᵀΣw</code><br>
          <small>Risque absolu minimal.
                 Favorise les fonds peu volatils et peu corrélés.</small>
        </div>
        <div style="flex:1;min-width:200px;background:{C['bg']};
             padding:10px;border-radius:7px;border-left:3px solid {C['navy']}">
          <b style="color:{C['dark']}">Max Sharpe</b><br>
          <code style="font-size:0.8rem">max (μ−Rf)/σ</code><br>
          <small>Portefeuille tangent. Meilleur rendement
                 ajusté au risque (Rf = 2.25%).</small>
        </div>
        <div style="flex:1;min-width:200px;background:{C['bg']};
             padding:10px;border-radius:7px;border-left:3px solid {C['red']}">
          <b style="color:{C['dark']}">Min CVaR</b><br>
          <code style="font-size:0.8rem">min 𝔼[r | r ≤ VaR₉₉%]</code><br>
          <small>Réduit directement les pertes extrêmes.
                 Contrainte : perf ≥ 95% actuel.</small>
        </div>
        <div style="flex:1;min-width:200px;background:{C['bg']};
             padding:10px;border-radius:7px;border-left:3px solid {C['gold']}">
          <b style="color:{C['dark']}">Max Perf / Min Risque</b><br>
          <code style="font-size:0.8rem">max μ − λ·σ²</code><br>
          <small>Compromis paramétrable.
                 λ = aversion au risque calibrée.</small>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="stitle">⚡ Lancer l\'optimisation et comparer les 4 méthodes</div>',
                unsafe_allow_html=True)

    if st.button("🚀 Calculer les 4 allocations optimales",
                 use_container_width=True, type="primary"):

        w_cur = np.clip(w_norm.copy(), w_min_pct/100, w_max_pct/100)
        w_cur /= w_cur.sum()

        results4 = {}
        progress = st.progress(0, text="Calcul en cours…")
        for i, m in enumerate(["Min Variance","Max Sharpe","Min CVaR",
                                "Max Perf / Min Risque"]):
            progress.progress((i+1)/4, text=f"Optimisation {m}…")
            wo = run_optim(m, w_cur, w_min_pct/100, w_max_pct/100)
            results4[m] = {"w": wo, "s": ptf_stats(wo)}
        progress.empty()

        # ── Tableau des poids recommandés ──────────────────────────────────
        st.markdown(
            '<div class="stitle">📋 Allocations optimales recommandées</div>',
            unsafe_allow_html=True,
        )
        rows_w = []
        for i, nom in enumerate(NOMS):
            d = OPCVM_DATA[nom]
            row = {
                "OPCVM":      nom[:28],
                "Catégorie":  d["cat"],
                "Actuel %":   f"{d['poids']:.2f}%",
                "α Jensen %": f"{d['alpha_j']:+.2f}%",
                "Sharpe":     f"{d['sharpe']:.2f}",
            }
            for m, res in results4.items():
                row[m] = f"{res['w'][i]*100:.2f}%"
            rows_w.append(row)
        df_w = pd.DataFrame(rows_w).set_index("OPCVM")
        st.dataframe(df_w, use_container_width=True, height=440)

        # ── KPIs des 4 méthodes ─────────────────────────────────────────────
        st.markdown(
            '<div class="stitle">📊 Performance des 4 portefeuilles optimaux</div>',
            unsafe_allow_html=True,
        )
        cols_4 = st.columns(4)
        method_labels = list(results4.keys())
        cols_colors   = [C["mid"], C["navy"], C["red"], C["gold"]]
        for col4, (m, color4) in zip(cols_4, zip(method_labels, cols_colors)):
            s = results4[m]["s"]
            delta_sh = s["sharpe"] - s_ref["sharpe"]
            delta_vr = s["var99"]  - s_ref["var99"]
            with col4:
                st.markdown(f"""
                <div style="background:white;border-radius:10px;padding:14px;
                     border-top:5px solid {color4};
                     box-shadow:0 3px 10px rgba(0,0,0,0.07);">
                  <h4 style="color:{C['dark']};margin:0 0 10px;font-size:0.95rem">
                    {m}</h4>
                  <p style="margin:3px 0;font-size:0.88rem">
                    <b>Perf :</b> {s['perf']:.2f}%/an</p>
                  <p style="margin:3px 0;font-size:0.88rem">
                    <b>Vol :</b> {s['vol']:.2f}%/an</p>
                  <p style="margin:3px 0;font-size:0.88rem">
                    <b>Sharpe :</b> {s['sharpe']:.3f}
                    <span style="font-size:0.78rem;color:
                      {'#2C5F2D' if delta_sh>0 else '#C0392B'}">
                      ({'+' if delta_sh>0 else ''}{delta_sh:.3f})</span></p>
                  <p style="margin:3px 0;font-size:0.88rem">
                    <b>VaR 99% :</b> {s['var99']:.4f}%/j
                    <span style="font-size:0.78rem;color:
                      {'#2C5F2D' if delta_vr>0 else '#C0392B'}">
                      ({'+' if delta_vr>0 else ''}{delta_vr:.4f}pp)</span></p>
                  <p style="margin:3px 0;font-size:0.88rem">
                    <b>Sortino :</b> {s['sortino']:.3f}</p>
                  <p style="margin:3px 0;font-size:0.88rem">
                    <b>DD Max :</b> {s['dd_max']:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

        # ── Graphique radar ─────────────────────────────────────────────────
        st.markdown(
            '<div class="stitle">🕸️ Profil rendement/risque (radar)</div>',
            unsafe_allow_html=True,
        )
        cats_r = ["Rendement","Sharpe","Sortino","VaR (inv.)","CVaR (inv.)"]
        all_r  = {"Actuel": s_ref, **{m: results4[m]["s"] for m in results4}}
        col_r  = {
            "Actuel":              C["gray"],
            "Min Variance":        C["mid"],
            "Max Sharpe":          C["navy"],
            "Min CVaR":            C["red"],
            "Max Perf / Min Risque": C["gold"],
        }
        radar_raw = {
            nm: [s["perf"], s["sharpe"], s["sortino"], -s["var99"], -s["cvar99"]]
            for nm, s in all_r.items()
        }
        radar_n = {k: [] for k in all_r}
        for i in range(5):
            vals = {k: radar_raw[k][i] for k in all_r}
            mn, mx = min(vals.values()), max(vals.values())
            for k in all_r:
                radar_n[k].append((vals[k]-mn)/(mx-mn+1e-10))

        fig_r = go.Figure()
        for nm in all_r:
            v = radar_n[nm] + [radar_n[nm][0]]
            fig_r.add_trace(go.Scatterpolar(
                r=v, theta=cats_r+[cats_r[0]],
                fill="toself", name=nm,
                line=dict(color=col_r.get(nm,C["gray"]),width=2),
                fillcolor=col_r.get(nm,C["gray"]),
                opacity=0.12,
            ))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=False,range=[0,1])),
            showlegend=True, height=380,
            legend=dict(font=dict(size=9),x=0.8,y=1.15),
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40,b=20,l=20,r=20),
        )
        st.plotly_chart(fig_r, use_container_width=True)

        # ── Heatmap poids ───────────────────────────────────────────────────
        st.markdown(
            '<div class="stitle">🗺️ Heatmap des allocations recommandées</div>',
            unsafe_allow_html=True,
        )
        short_n = [n[:18] for n in NOMS]
        mths    = ["Actuel"] + list(results4.keys())
        wmat    = np.vstack([
            w_ref*100,
            *[results4[m]["w"]*100 for m in results4],
        ])
        fig_hm = go.Figure(go.Heatmap(
            z=wmat, x=short_n, y=mths,
            colorscale="Greens",
            text=np.round(wmat,1),
            texttemplate="%{text}%",
            textfont=dict(size=8),
            hovertemplate="<b>%{y}</b> — %{x}<br>Poids: %{z:.1f}%<extra></extra>",
            colorbar=dict(title="Poids (%)",thickness=12),
        ))
        fig_hm.update_layout(
            height=250, margin=dict(t=5,b=75,l=130,r=40),
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickangle=-40,tickfont=dict(size=8)),
            yaxis=dict(tickfont=dict(size=9)),
        )
        st.plotly_chart(fig_hm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 4 — DETAIL PAR OPCVM
# ═══════════════════════════════════════════════════════════════
with tab4:

    st.markdown('<div class="stitle">📋 Fiche complète — 14 OPCVM</div>',
                unsafe_allow_html=True)

    rows_d = []
    for i, nom in enumerate(NOMS):
        d  = OPCVM_DATA[nom]
        wp = float(w_norm[i] * 100)
        rows_d.append({
            "OPCVM":         nom,
            "Catégorie":     d["cat"],
            "Benchmark":     d["bench"],
            "Val. marché (MAD)": f"{d['val_mad']:,.0f}",
            "Poids actuel":  f"{d['poids']:.2f}%",
            "Poids courant": f"{wp:.2f}%",
            "Δ (pp)":        f"{wp-d['poids']:+.2f}",
            "Perf %/an":     f"{d['perf']:.2f}",
            "β":             f"{d['beta']:.2f}",
            "α Jensen %":    f"{d['alpha_j']:+.2f}",
            "Sharpe":        f"{d['sharpe']:.3f}",
            "Sortino":       f"{d['sortino']:.3f}",
            "VaR 99% %/j":   f"{d['var99']:.3f}",
            "TE %":          f"{d['te']:.3f}",
            "IR":            f"{d['ir']:.3f}",
            "DD Max %":      f"{d['dd']:.2f}",
        })
    df_d = pd.DataFrame(rows_d).set_index("OPCVM")
    st.dataframe(df_d, use_container_width=True, height=480)

    c_sc1, c_sc2 = st.columns(2)

    with c_sc1:
        st.markdown(
            '<div class="stitle">🔵 α Jensen vs Sharpe</div>',
            unsafe_allow_html=True,
        )
        fig_s = go.Figure()
        for i, nom in enumerate(NOMS):
            d   = OPCVM_DATA[nom]
            wp  = float(w_norm[i]*100)
            col = C["mid"] if d["alpha_j"]>0 else C["red"]
            fig_s.add_trace(go.Scatter(
                x=[d["alpha_j"]], y=[d["sharpe"]],
                mode="markers+text",
                marker=dict(size=max(wp*2.2,6), color=col, opacity=0.72,
                            line=dict(color="white",width=1)),
                text=[nom[:14]], textposition="top center",
                textfont=dict(size=7.5),
                showlegend=False,
                hovertemplate=(
                    f"<b>{nom}</b><br>"
                    f"α Jensen:{d['alpha_j']:+.2f}%<br>"
                    f"Sharpe:{d['sharpe']:.3f}<br>"
                    f"Poids:{wp:.1f}%<extra></extra>"
                ),
            ))
        fig_s.add_vline(x=0, line_dash="dot", line_color=C["red"])
        fig_s.add_hline(y=2, line_dash="dot", line_color=C["navy"])
        fig_s.update_layout(
            height=370, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(244,247,251,0.5)",
            xaxis=dict(title="Alpha Jensen (%)", gridcolor=C["lgray"]),
            yaxis=dict(title="Sharpe Ratio",     gridcolor=C["lgray"]),
            margin=dict(t=10,b=40,l=50,r=10),
        )
        st.plotly_chart(fig_s, use_container_width=True)

    with c_sc2:
        st.markdown(
            '<div class="stitle">📉 VaR 99% vs Perf. annuelle</div>',
            unsafe_allow_html=True,
        )
        fig_v = go.Figure()
        for i, nom in enumerate(NOMS):
            d   = OPCVM_DATA[nom]
            wp  = float(w_norm[i]*100)
            col = C["mid"] if d["alpha_j"]>0 else C["red"]
            fig_v.add_trace(go.Scatter(
                x=[d["var99"]], y=[d["perf"]],
                mode="markers+text",
                marker=dict(size=max(wp*2.2,6), color=col, opacity=0.72,
                            line=dict(color="white",width=1)),
                text=[nom[:14]], textposition="top center",
                textfont=dict(size=7.5),
                showlegend=False,
                hovertemplate=(
                    f"<b>{nom}</b><br>"
                    f"VaR:{d['var99']:.3f}%<br>"
                    f"Perf:{d['perf']:.2f}%<br>"
                    f"Poids:{wp:.1f}%<extra></extra>"
                ),
            ))
        fig_v.update_layout(
            height=370, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(244,247,251,0.5)",
            xaxis=dict(title="VaR 99% (%/jour)", gridcolor=C["lgray"]),
            yaxis=dict(title="Perf. annuelle (%)", gridcolor=C["lgray"]),
            margin=dict(t=10,b=40,l=50,r=10),
        )
        st.plotly_chart(fig_v, use_container_width=True)
