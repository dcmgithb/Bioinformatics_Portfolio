"""
Physiologically-Based Pharmacokinetic (PBPK) Modelling
========================================================
Implements a 2-compartment oral PBPK model with Michaelis-Menten
hepatic metabolism, first-order intestinal absorption, and renal clearance.
Supports dose-response simulation, DDI (drug-drug interactions), and
enzyme saturation/inhibition kinetics.

Model structure
---------------
GI lumen → Central (blood/plasma) ↔ Peripheral (tissue)
                    ↓
               Hepatic metabolism (CYP, Km, Vmax)
               Renal clearance (GFR-based)

Differential equations
----------------------
dA_gut/dt  = -ka * A_gut
dC_c/dt    = (ka * A_gut * F_abs) / Vc
             - (Vmax * C_c) / (Km + C_c) * (1/Vc)     ← hepatic CL (M-M)
             - CL_renal/Vc * C_c
             - ktp * C_c + ktc * C_t/Vt               ← tissue exchange
dC_t/dt    = ktp * C_c * Vc/Vt - ktc * C_t

PK parameters estimated from structure
---------------------------------------
Vd        : volume of distribution (from logP, pKa, MW, protein binding)
CL_int    : intrinsic clearance (from SoM probability, enzyme affinity)
F         : oral bioavailability (absorption × first-pass extraction)
fu_plasma : fraction unbound (from logP, MW)
t1/2      : elimination half-life

References
----------
Rowland & Tozer (2011) Clinical Pharmacokinetics and Pharmacodynamics
Poulin & Theil (2002) A priori prediction of Vss — JPET
Obach et al. (1997) Relationship between in vitro and in vivo clearance
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

warnings.filterwarnings("ignore")
SEED = 42
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

# ── Physiological constants ────────────────────────────────────────────────────

PHYSIOL = {
    "QH":      90.0,    # Hepatic blood flow (L/h) — 70 kg human
    "QR":      75.0,    # Renal blood flow (L/h)
    "Vc_kg":   0.050,   # Central volume / body weight (L/kg)
    "Vt_kg":   0.200,   # Peripheral volume / body weight (L/kg)
    "BW":      70.0,    # Body weight (kg)
    "GFR":     0.125,   # Glomerular filtration rate (L/h per kg)
    "F_blood": 1.0,     # Blood:plasma ratio (simplified)
    "CLH_max": 90.0,    # Max hepatic clearance = QH
}

PHYSIOL["Vc"] = PHYSIOL["Vc_kg"] * PHYSIOL["BW"]
PHYSIOL["Vt"] = PHYSIOL["Vt_kg"] * PHYSIOL["BW"]


# ── PK parameter estimation from structure ─────────────────────────────────────

def estimate_pka_basic(mol: Chem.Mol) -> float:
    """Estimate strongest basic pKa (rules-based)."""
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H1,H2;!$(NC=O);!a]")):
        return 9.5
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[nH]1ccnc1")):
        return 7.0
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[nH]")):
        return 5.5
    return -2.0


def estimate_fu_plasma(logp: float, mw: float) -> float:
    """
    Estimate fraction unbound in plasma.
    Based on Obach (1999) and Austin et al. (2002):
      logfu ≈ 0.072 × (logP - 1.5) - 0.0044 × MW + constant
    Returned as [0.01, 1.0].
    """
    log_fu = -0.072 * max(logp - 1.5, 0) - 0.004 * (mw / 100) + 0.5
    fu     = 10 ** log_fu
    return float(np.clip(fu, 0.01, 1.0))


def estimate_vd(
    logp: float,
    fu_plasma: float,
    mw: float,
    pka_base: float,
) -> float:
    """
    Estimate steady-state volume of distribution (L/kg).
    Modified Poulin-Theil model with basic drug ionisation correction.
    """
    # Tissue:plasma partitioning driven by lipophilicity
    Kp_tissue = 10 ** (0.45 * logp)  # simplified

    # Basic drugs: additional tissue sequestration due to ion trapping
    if pka_base > 7.0:
        ion_trap = 10 ** (pka_base - 7.4)  # ionisation at lysosomal pH ~5.0
        Kp_tissue *= min(ion_trap, 100)

    # fu correction
    Vd = (PHYSIOL["Vc_kg"] + Kp_tissue * PHYSIOL["Vt_kg"]) / fu_plasma
    return float(np.clip(Vd, 0.05, 100.0))


def estimate_cl_intrinsic(
    logp: float,
    mw: float,
    n_aromatic_rings: int,
    cyp3a4_prob: float = 0.7,
) -> float:
    """
    Estimate intrinsic hepatic clearance (L/h/kg).
    Scaled from in vitro microsomal CLint using well-stirred liver model.
    """
    # Base CLint driven by lipophilicity and aromatic complexity
    cl_base = 0.3 * (1 + max(logp - 1.0, 0)) * (1 + 0.2 * n_aromatic_rings)
    cl_base *= cyp3a4_prob   # scale by CYP3A4 involvement

    # MW penalty: large molecules clear more slowly
    if mw > 500:
        cl_base *= 0.5
    elif mw > 400:
        cl_base *= 0.75

    return float(np.clip(cl_base, 0.01, 20.0))


def estimate_f_oral(fu_plasma: float, cl_int: float, logp: float, mw: float) -> float:
    """
    Estimate oral bioavailability F = Fa × Fg × Fh.
    Fa (absorption): based on TPSA/MW/logP rules
    Fg (gut wall):   simplification (CYP3A4 in enterocytes)
    Fh (hepatic FPE): 1 - ER = 1 - CLh/QH
    """
    # Fa: fraction absorbed (Egan et al. rule)
    if mw > 500 or logp < -2 or logp > 6:
        Fa = 0.2
    elif mw < 300 and -1 <= logp <= 4:
        Fa = 0.95
    else:
        Fa = 0.75

    # Fh: well-stirred model CLh = QH * fu * CLint / (QH + fu * CLint)
    QH   = PHYSIOL["QH"] / PHYSIOL["BW"]  # per kg
    CLh  = QH * fu_plasma * cl_int / (QH + fu_plasma * cl_int)
    Fh   = 1 - CLh / QH

    Fg   = 0.95   # gut wall extraction (simplified)
    F    = Fa * Fg * Fh
    return float(np.clip(F, 0.01, 1.0))


@dataclass
class PKParameters:
    """Pharmacokinetic parameters for a compound."""
    smiles:       str
    name:         str       = "Compound"
    dose_mg:      float     = 100.0    # administered dose (mg)
    route:        str       = "oral"   # "oral" or "iv"

    # Estimated structural parameters
    mw:           float     = 300.0
    logp:         float     = 2.0
    fu_plasma:    float     = 0.1
    pka_base:     float     = 8.0

    # Volume and clearance
    Vd:           float     = 0.6      # L/kg
    CL_int:       float     = 1.0      # L/h/kg intrinsic hepatic CLint
    F_oral:       float     = 0.5      # oral bioavailability

    # Renal clearance
    CL_renal:     float     = 0.05     # L/h/kg (GFR × fu × fraction renal)

    # Absorption
    ka:           float     = 1.5      # absorption rate constant (h-1)

    # M-M kinetics for hepatic metabolism
    Km:           float     = 1.0      # Michaelis constant (mg/L)
    Vmax:         float     = 50.0     # Maximum metabolic rate (mg/h)

    # Inhibitor (for DDI)
    inhibitors:   List[Dict] = field(default_factory=list)
                               # [{name, Ki, mechanism}]

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        name: str = "Compound",
        dose_mg: float = 100.0,
        route: str = "oral",
    ) -> "PKParameters":
        """Estimate all PK parameters from SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        mw    = Descriptors.MolWt(mol)
        logp  = Descriptors.MolLogP(mol)
        n_ar  = rdMolDescriptors.CalcNumAromaticRings(mol)
        pka_b = estimate_pka_basic(mol)
        fu    = estimate_fu_plasma(logp, mw)
        vd    = estimate_vd(logp, fu, mw, pka_b)
        cl_i  = estimate_cl_intrinsic(logp, mw, n_ar)
        f_o   = estimate_f_oral(fu, cl_i, logp, mw)

        # Renal: GFR × fu + active secretion (simplified)
        cl_r  = PHYSIOL["GFR"] * fu * (1 + 0.5 * (logp < 1))

        # Km and Vmax from CLint (steady-state approximation)
        # CL_int = Vmax/Km → Vmax = CL_int × Km
        Km   = max(0.1, 2.0 - 0.3 * logp)   # μg/mL — less lipophilic → higher Km
        Vmax = cl_i * PHYSIOL["BW"] * Km     # mg/h

        return cls(
            smiles     = smiles,
            name       = name,
            dose_mg    = dose_mg,
            route      = route,
            mw         = mw,
            logp       = logp,
            fu_plasma  = fu,
            pka_base   = pka_b,
            Vd         = vd,
            CL_int     = cl_i,
            F_oral     = f_o,
            CL_renal   = cl_r,
            ka         = 1.5 if logp > 0 else 0.6,  # high logP → faster absorption
            Km         = Km,
            Vmax       = Vmax,
        )

    def t_half_elim(self) -> float:
        """Terminal elimination half-life (h)."""
        Vc = PHYSIOL["Vc"]
        Vt = PHYSIOL["Vt"]
        BW = PHYSIOL["BW"]
        # Apparent CL at low concentrations (linear portion)
        QH    = PHYSIOL["QH"]
        CL_h  = QH * self.fu_plasma * self.CL_int * BW / (
            QH + self.fu_plasma * self.CL_int * BW
        )
        CL_r  = self.CL_renal * BW
        CL_tot = CL_h + CL_r
        if CL_tot <= 0:
            return float("inf")
        return float(0.693 * self.Vd * BW / CL_tot)


# ── ODE system ────────────────────────────────────────────────────────────────

def pk_odes(
    t: float,
    y: np.ndarray,
    params: PKParameters,
    bw: float = 70.0,
) -> np.ndarray:
    """
    2-compartment PBPK ODE system with Michaelis-Menten hepatic metabolism.

    State variables
    ---------------
    y[0] : A_gut   — drug amount in GI tract (mg)
    y[1] : C_c     — drug concentration in central compartment (mg/L)
    y[2] : A_t     — drug amount in peripheral/tissue compartment (mg)

    Returns dy/dt.
    """
    A_gut, C_c, A_t = y
    C_c = max(C_c, 0.0)

    Vc  = PHYSIOL["Vc"]    # L
    Vt  = PHYSIOL["Vt"]    # L
    QH  = PHYSIOL["QH"]    # L/h

    # Kinetic transfer rate constants
    CL_d    = 10.0          # distributional clearance (L/h)
    ktp     = CL_d / Vc     # central → tissue rate constant
    kct     = CL_d / Vt     # tissue → central rate constant

    # Hepatic metabolism (Michaelis-Menten, operates on unbound Cc)
    C_u     = C_c * params.fu_plasma
    MM_met  = (params.Vmax * C_u) / (params.Km + C_u) if (params.Km + C_u) > 0 else 0.0

    # Inhibitor DDI: competitive inhibition → increase effective Km
    for inh in params.inhibitors:
        Ki  = inh.get("Ki", 10.0)
        Ci  = inh.get("concentration", 1.0)  # inhibitor plasma concentration
        MM_met *= params.Km / (params.Km + Ci / Ki)  # reduce CLint

    # Renal clearance
    CL_r    = params.CL_renal * bw   # L/h

    # ODEs
    dA_gut = -params.ka * A_gut

    if params.route == "oral":
        input_rate = params.ka * A_gut * params.F_oral / Vc
    else:  # iv bolus — handled via initial conditions
        input_rate = 0.0

    dC_c = (input_rate
            - MM_met / Vc
            - CL_r / Vc * C_c
            - ktp * C_c
            + kct * (A_t / Vt))

    dA_t = ktp * C_c * Vc - kct * A_t

    return [dA_gut, dC_c, dA_t]


# ── PK simulation ──────────────────────────────────────────────────────────────

def simulate_pk(
    params: PKParameters,
    t_end: float = 48.0,
    n_points: int = 500,
) -> pd.DataFrame:
    """
    Simulate plasma concentration-time profile.

    Returns
    -------
    DataFrame: time [h], Cc [mg/L], Ct [mg/L], A_gut [mg]
    """
    t_eval = np.linspace(0, t_end, n_points)

    # Initial conditions
    if params.route == "oral":
        y0 = [params.dose_mg, 0.0, 0.0]   # all dose in gut
    else:
        # IV bolus: dose directly into central compartment
        y0 = [0.0, params.dose_mg / PHYSIOL["Vc"], 0.0]

    sol = solve_ivp(
        fun=lambda t, y: pk_odes(t, y, params),
        t_span=(0, t_end),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
        max_step=0.1,
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    Cc = np.maximum(sol.y[1], 0.0)
    At = np.maximum(sol.y[2], 0.0)
    Ct = At / PHYSIOL["Vt"]

    return pd.DataFrame({
        "time_h":   sol.t,
        "Cc_mgL":   Cc,
        "Ct_mgL":   Ct,
        "A_gut_mg": np.maximum(sol.y[0], 0.0),
    })


def compute_nca_parameters(pk_df: pd.DataFrame, dose_mg: float) -> Dict[str, float]:
    """
    Non-compartmental analysis: Cmax, Tmax, AUC, t1/2, MRT.
    """
    t   = pk_df["time_h"].values
    Cc  = pk_df["Cc_mgL"].values

    Cmax_idx = int(np.argmax(Cc))
    Cmax     = float(Cc[Cmax_idx])
    Tmax     = float(t[Cmax_idx])

    # AUC by trapezoidal rule
    AUC      = float(np.trapz(Cc, t))

    # t1/2 from terminal slope (last 30% of time)
    term_idx = int(len(t) * 0.7)
    if term_idx < len(t) - 3 and Cc[term_idx] > 1e-9:
        try:
            log_Cc = np.log(Cc[term_idx:] + 1e-12)
            slope, _ = np.polyfit(t[term_idx:], log_Cc, 1)
            t_half   = float(-0.693 / slope) if slope < 0 else float("nan")
        except Exception:
            t_half   = float("nan")
    else:
        t_half = float("nan")

    # Clearance from NCA: CL = Dose / AUC
    CL_obs = dose_mg / AUC if AUC > 0 else float("nan")

    return {
        "Cmax_mgL":  round(Cmax, 4),
        "Tmax_h":    round(Tmax, 2),
        "AUC_mgLh":  round(AUC, 3),
        "t_half_h":  round(t_half, 2) if np.isfinite(t_half) else float("nan"),
        "CL_Lh":     round(CL_obs, 3),
    }


# ── Multi-dose simulation ──────────────────────────────────────────────────────

def simulate_multiple_doses(
    params: PKParameters,
    n_doses: int  = 5,
    interval_h: float = 12.0,
    t_end: Optional[float] = None,
) -> pd.DataFrame:
    """Simulate multiple oral doses until near steady-state."""
    if t_end is None:
        t_end = n_doses * interval_h + 24.0

    t_eval   = np.linspace(0, t_end, n_doses * 200 + 200)
    dose_times = [i * interval_h for i in range(n_doses)]

    # Event-driven simulation: restart at each dose
    all_t, all_Cc = [], []
    y = [params.dose_mg, 0.0, 0.0]  # initial dose
    t_start = 0.0

    for di, dose_t in enumerate(dose_times[1:] + [t_end], 1):
        t_span  = (t_start, dose_t)
        t_pts   = np.linspace(t_start, dose_t, 200)
        sol = solve_ivp(lambda t, y: pk_odes(t, y, params),
                        t_span, y, t_eval=t_pts, method="RK45",
                        rtol=1e-6, atol=1e-9)
        if sol.success:
            all_t.extend(sol.t)
            all_Cc.extend(np.maximum(sol.y[1], 0.0))
            # Add dose at end of interval
            if di < len(dose_times):
                y = [sol.y[0, -1] + params.dose_mg,
                     sol.y[1, -1],
                     sol.y[2, -1]]
            else:
                y = [sol.y[0, -1], sol.y[1, -1], sol.y[2, -1]]
        t_start = dose_t

    return pd.DataFrame({"time_h": all_t, "Cc_mgL": all_Cc})


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_pk_profile(
    pk_dfs: Dict[str, pd.DataFrame],
    nca: Dict[str, Dict],
    title: str = "PK Profile",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """Multi-panel PK visualisation."""
    n     = len(pk_dfs)
    fig   = plt.figure(figsize=(14, 5 * ((n + 1) // 2)))
    axes  = fig.subplots((n + 1) // 2, min(n, 2))
    axes  = axes.flatten() if hasattr(axes, "flatten") else [axes]

    colours = plt.cm.tab10(np.linspace(0, 0.8, n))

    for ax, (name, df), col in zip(axes, pk_dfs.items(), colours):
        ax.plot(df["time_h"], df["Cc_mgL"], lw=2.5, color=col, label="Central")
        if "Ct_mgL" in df.columns:
            ax.plot(df["time_h"], df["Ct_mgL"], lw=1.5, color=col,
                    ls="--", alpha=0.6, label="Tissue")

        params_nca = nca.get(name, {})
        if params_nca:
            cmax = params_nca.get("Cmax_mgL", None)
            tmax = params_nca.get("Tmax_h",    None)
            if cmax and tmax:
                ax.axhline(cmax, ls=":", color="red", alpha=0.5, lw=1)
                ax.axvline(tmax, ls=":", color="grey", alpha=0.5, lw=1)
                ax.annotate(
                    f"Cmax={cmax:.2f}\nTmax={tmax:.1f}h\n"
                    f"AUC={params_nca.get('AUC_mgLh', 0):.1f}\n"
                    f"t½={params_nca.get('t_half_h', 0):.1f}h",
                    xy=(tmax, cmax), xytext=(tmax * 1.2, cmax * 0.8),
                    fontsize=7, color="black",
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="grey"),
                    bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.2"),
                )

        ax.set_xlabel("Time (h)", fontsize=10)
        ax.set_ylabel("Plasma concentration (mg/L)", fontsize=10)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ddi_comparison(
    pk_ctrl: pd.DataFrame,
    pk_ddi: pd.DataFrame,
    drug_name: str,
    inhibitor_name: str,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """Overlay PK profiles with and without DDI."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(pk_ctrl["time_h"], pk_ctrl["Cc_mgL"], "b-", lw=2.5, label="Control")
    axes[0].plot(pk_ddi["time_h"],  pk_ddi["Cc_mgL"],  "r-", lw=2.5, label=f"+ {inhibitor_name}")
    axes[0].set_xlabel("Time (h)"); axes[0].set_ylabel("Plasma Conc. (mg/L)")
    axes[0].set_title(f"DDI: {drug_name} ± {inhibitor_name}", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.25)

    # AUC ratio
    auc_ctrl = np.trapz(pk_ctrl["Cc_mgL"], pk_ctrl["time_h"])
    auc_ddi  = np.trapz(pk_ddi["Cc_mgL"],  pk_ddi["time_h"])
    auc_ratio = auc_ddi / max(auc_ctrl, 1e-9)
    cmax_ratio = pk_ddi["Cc_mgL"].max() / max(pk_ctrl["Cc_mgL"].max(), 1e-9)

    categories = ["AUC ratio", "Cmax ratio"]
    values     = [auc_ratio, cmax_ratio]
    colours    = ["#E74C3C" if v > 2 else "#F39C12" if v > 1.25 else "#2ECC71"
                  for v in values]
    axes[1].bar(categories, values, color=colours, alpha=0.85, width=0.4)
    axes[1].axhline(1.0, ls="--", color="black", lw=1, alpha=0.5, label="No DDI")
    axes[1].axhline(2.0, ls="--", color="red",   lw=1, alpha=0.5, label="2× threshold")
    axes[1].set_ylabel("Ratio (DDI / Control)", fontsize=10)
    axes[1].set_title("DDI Magnitude", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=8)
    for i, (v, c) in enumerate(zip(values, categories)):
        axes[1].text(i, v + 0.05, f"{v:.2f}×", ha="center", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 65)
    print("  PBPK / Pharmacokinetic Modelling Engine")
    print("=" * 65)

    test_drugs = {
        "Paracetamol (500 mg)": ("CC(=O)Nc1ccc(O)cc1", 500.0, "oral"),
        "Ibuprofen (400 mg)":   ("CC(C)Cc1ccc(C(C)C(=O)O)cc1", 400.0, "oral"),
        "Lidocaine (100 mg IV)":("CCN(CC)CC(=O)Nc1c(C)cccc1C", 100.0, "iv"),
        "Caffeine (200 mg)":    ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", 200.0, "oral"),
    }

    pk_dfs, nca_params = {}, {}

    print("\n[1/4] Estimating PK parameters from structure...")
    for drug_name, (smi, dose, route) in test_drugs.items():
        pk = PKParameters.from_smiles(smi, name=drug_name, dose_mg=dose, route=route)
        print(f"\n  {drug_name}")
        print(f"    MW={pk.mw:.0f}  LogP={pk.logp:.2f}  fu={pk.fu_plasma:.3f}")
        print(f"    Vd={pk.Vd:.2f} L/kg  CLint={pk.CL_int:.2f} L/h/kg  F={pk.F_oral:.2f}")
        print(f"    t½(est)={pk.t_half_elim():.1f} h  ka={pk.ka:.2f} h⁻¹")

    print("\n[2/4] Running PK simulations...")
    for drug_name, (smi, dose, route) in test_drugs.items():
        pk  = PKParameters.from_smiles(smi, name=drug_name, dose_mg=dose, route=route)
        df  = simulate_pk(pk, t_end=36.0)
        nca = compute_nca_parameters(df, dose)
        pk_dfs[drug_name]    = df
        nca_params[drug_name] = nca
        print(f"  {drug_name}: Cmax={nca['Cmax_mgL']:.3f} mg/L  "
              f"Tmax={nca['Tmax_h']:.1f}h  AUC={nca['AUC_mgLh']:.2f}  "
              f"t½={nca['t_half_h']:.1f}h")

    # Plot PK profiles
    fig = plot_pk_profile(pk_dfs, nca_params, title="PBPK Simulation Results",
                           out_path=OUT_DIR / "pk_profiles.png")
    plt.close(fig)

    # 3. Multi-dose simulation (Paracetamol q12h × 5 doses)
    print("\n[3/4] Multi-dose simulation (Paracetamol 500 mg q12h × 5)...")
    pk_para = PKParameters.from_smiles(
        "CC(=O)Nc1ccc(O)cc1", name="Paracetamol", dose_mg=500, route="oral"
    )
    md_df = simulate_multiple_doses(pk_para, n_doses=5, interval_h=12.0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(md_df["time_h"], md_df["Cc_mgL"], "b-", lw=2.5)
    for i in range(5):
        ax.axvline(i * 12, ls="--", color="grey", alpha=0.4, lw=1)
        ax.text(i * 12 + 0.3, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 0.5,
                f"Dose {i+1}", fontsize=7, color="grey")
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Plasma Conc. (mg/L)")
    ax.set_title("Paracetamol 500 mg q12h — Multiple Dose Simulation", fontsize=11)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "multidose_simulation.png", dpi=150)
    plt.close()
    print(f"  Saved multi-dose profile to {OUT_DIR}/multidose_simulation.png")

    # 4. DDI simulation: Lidocaine + CYP3A4 inhibitor (like ketoconazole)
    print("\n[4/4] DDI simulation (Lidocaine + CYP3A4 inhibitor)...")
    pk_lido_ctrl = PKParameters.from_smiles(
        "CCN(CC)CC(=O)Nc1c(C)cccc1C", name="Lidocaine_ctrl", dose_mg=100, route="iv"
    )
    pk_lido_ddi = PKParameters.from_smiles(
        "CCN(CC)CC(=O)Nc1c(C)cccc1C", name="Lidocaine_DDI", dose_mg=100, route="iv"
    )
    # Add ketoconazole-like inhibitor: Ki=0.005 μg/mL, Ci=0.5 μg/mL
    pk_lido_ddi.inhibitors = [{"name": "CYP3A4_inhibitor", "Ki": 0.005, "concentration": 0.5}]

    df_ctrl = simulate_pk(pk_lido_ctrl, t_end=24.0)
    df_ddi  = simulate_pk(pk_lido_ddi,  t_end=24.0)

    auc_ctrl  = np.trapz(df_ctrl["Cc_mgL"], df_ctrl["time_h"])
    auc_ddi   = np.trapz(df_ddi["Cc_mgL"],  df_ddi["time_h"])
    print(f"  AUC ratio (DDI/control) = {auc_ddi/max(auc_ctrl, 1e-9):.2f}× "
          f"({'⚠ significant DDI' if auc_ddi/auc_ctrl > 2 else 'moderate DDI'})")

    fig = plot_ddi_comparison(df_ctrl, df_ddi, "Lidocaine", "CYP3A4 inhibitor",
                               out_path=OUT_DIR / "ddi_simulation.png")
    plt.close(fig)

    # Export NCA summary
    nca_df = pd.DataFrame([
        {"Drug": name, **params} for name, params in nca_params.items()
    ])
    nca_df.to_csv(OUT_DIR / "pk_nca_summary.csv", index=False)
    print(f"\nSaved PK results to {OUT_DIR}/")
