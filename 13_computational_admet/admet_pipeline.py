"""
admet_pipeline.py — Integrated ADMET Pipeline
==============================================
Ties together all five ADMET modules into a single end-to-end profiling
workflow: atom-level SoM prediction → metabolic tree generation → CYP
isoform classification → metabolite toxicity profiling → PBPK simulation.

Falls back gracefully to rule-based StandaloneADMET predictor when the
individual ML modules are not importable (e.g. first-run without training).
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, AllChem

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

# ── Optional module imports ──────────────────────────────────────────────────
try:
    from som_prediction import SoMPredictor, predict_som
    HAS_SOM = True
except ImportError:
    HAS_SOM = False

try:
    from metabolite_generation import MetabolicTreeGenerator
    HAS_METAB = True
except ImportError:
    HAS_METAB = False

try:
    from cyp_isoform_classifier import CYPIsoformClassifier, mol_to_features as cyp_features
    HAS_CYP = True
except ImportError:
    HAS_CYP = False

try:
    from metabolite_toxicity import ToxicityProfiler, generate_tox_dataset
    HAS_TOX = True
except ImportError:
    HAS_TOX = False

try:
    from pbpk_model import PKParameters, simulate_pk, compute_nca_parameters
    HAS_PBPK = True
except ImportError:
    HAS_PBPK = False


# ── ADMETResult dataclass ────────────────────────────────────────────────────

@dataclass
class ADMETResult:
    smiles: str
    name: str

    # SoM
    som_atoms: List[int] = field(default_factory=list)
    som_scores: Dict[int, float] = field(default_factory=dict)
    top_som_class: str = ""

    # Metabolites
    metabolites: List[Dict] = field(default_factory=list)
    n_metabolites: int = 0
    reactive_metabolites: int = 0

    # CYP
    cyp_profile: Dict[str, float] = field(default_factory=dict)
    primary_cyp: str = ""

    # Toxicity
    tox_probs: Dict[str, float] = field(default_factory=dict)
    structural_alerts: List[Dict] = field(default_factory=list)
    safety_score: float = 100.0

    # PK
    pk_params: Dict[str, float] = field(default_factory=dict)

    # Composite
    overall_score: float = 0.0
    soft_spots: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# ── Standalone rule-based ADMET predictor ───────────────────────────────────

class StandaloneADMET:
    """
    Pure-RDKit rule-based ADMET predictor used when ML modules are unavailable.
    Returns the same dict keys that the full modules would populate.
    """

    # Minimal structural alert SMARTS
    _ALERT_SMARTS = {
        "quinone":          "[#6]1(=O)[#6]=[#6][#6](=O)[#6]=[#6]1",
        "michael_acceptor": "[C]=[C]-[C]=O",
        "aromatic_amine":   "[NH2,NH1;$([N]c)]",
        "nitro":            "[$([N+](=O)[O-]),$([N](=O)=O)]",
        "epoxide":          "[C;X3]1[O]1",
        "aldehyde":         "[CH]=O",
        "hydrazine":        "[NH][NH]",
    }

    def __init__(self) -> None:
        self._pats = {}
        for k, s in self._ALERT_SMARTS.items():
            pat = Chem.MolFromSmarts(s)
            if pat:
                self._pats[k] = pat

    # ------------------------------------------------------------------
    def predict(self, smiles: str) -> Dict:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        mw    = Descriptors.ExactMolWt(mol)
        logp  = Crippen.MolLogP(mol)
        tpsa  = rdMolDescriptors.CalcTPSA(mol)
        hbd   = rdMolDescriptors.CalcNumHBD(mol)
        hba   = rdMolDescriptors.CalcNumHBA(mol)
        nrotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
        n_ar  = rdMolDescriptors.CalcNumAromaticRings(mol)

        # Lipinski
        lip_pass = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)

        # Absorption proxy (Veber rules: TPSA<=140, rotB<=10)
        absorption = 1.0 if (tpsa <= 140 and nrotb <= 10) else 0.5

        # BBB
        bbb = (mw < 400 and 1.0 <= logp <= 3.0 and tpsa < 90 and hbd < 3)

        # CYP3A4 substrate likelihood
        cyp3a4 = mw > 400 or n_ar >= 3 or logp > 3

        # Structural alerts
        alerts = []
        for name, pat in self._pats.items():
            if mol.HasSubstructMatch(pat):
                alerts.append(name)

        # Safety score (simple)
        safety = max(0.0, 100.0 - len(alerts) * 15.0 - (0 if lip_pass else 10.0))

        # Simple PK estimates
        f_oral = min(0.95, absorption * (0.7 if cyp3a4 else 0.9))
        t_half = 4.0 + logp * 0.8  # rough estimate
        vd     = 0.5 + logp * 0.2

        return {
            "MW": mw, "LogP": logp, "TPSA": tpsa,
            "HBD": hbd, "HBA": hba, "nRotB": nrotb,
            "Lipinski_pass": lip_pass,
            "BBB_penetrant": bbb,
            "CYP3A4_substrate": cyp3a4,
            "absorption": absorption,
            "structural_alerts": alerts,
            "safety_score": safety,
            "F_oral": f_oral,
            "t_half_h": t_half,
            "Vd_L_kg": vd,
        }


# ── Global singletons (lazy init) ───────────────────────────────────────────

_standalone = StandaloneADMET()
_som_model: Optional[Any] = None
_cyp_model: Optional[Any] = None
_tox_model: Optional[ToxicityProfiler] = None


def _get_tox_model() -> Optional[ToxicityProfiler]:
    global _tox_model
    if _tox_model is None and HAS_TOX:
        try:
            df, X = generate_tox_dataset(n=400, seed=0)
            profiler = ToxicityProfiler()
            profiler.fit(df["smiles"].tolist(), df[ToxicityProfiler.ENDPOINTS])
            _tox_model = profiler
        except Exception:
            pass
    return _tox_model


def _get_som_model() -> Optional[Any]:
    global _som_model
    if _som_model is None and HAS_SOM:
        try:
            _som_model = SoMPredictor(n_layers=3, hidden=128)
        except Exception:
            pass
    return _som_model


# ── Pipeline class ───────────────────────────────────────────────────────────

class ADMETPipeline:
    """
    End-to-end ADMET profiling pipeline.

    Parameters
    ----------
    use_ml : bool
        If True, attempt to load and use the ML sub-modules.
        If False (or modules not installed), fall back to StandaloneADMET.
    """

    def __init__(self, use_ml: bool = True) -> None:
        self.use_ml = use_ml
        self._tox = _get_tox_model() if use_ml else None
        self._som = _get_som_model() if use_ml else None

    # ------------------------------------------------------------------
    def run(
        self,
        smiles: str,
        name: str = "",
        dose_mg: float = 100.0,
    ) -> ADMETResult:
        """Full pipeline: SoM → metabolites → CYP → tox → PK → scoring."""
        result = ADMETResult(smiles=smiles, name=name or smiles[:20])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result.soft_spots = ["Invalid SMILES"]
            return result

        # ── 1. Sites of Metabolism ──────────────────────────────────
        if self.use_ml and HAS_SOM and self._som is not None:
            try:
                som_result = predict_som(self._som, smiles)
                if som_result:
                    top_sites = som_result["top_sites"][:3]
                    result.som_atoms     = [s[0] for s in top_sites]
                    result.som_scores    = {s[0]: s[2] for s in top_sites}
                    result.top_som_class = top_sites[0][1] if top_sites else ""
            except Exception:
                pass

        if not result.som_atoms:
            # Fallback: flag aromatic carbons as putative SoM
            result.som_atoms = [
                a.GetIdx() for a in mol.GetAtoms()
                if a.GetIsAromatic() and a.GetAtomicNum() == 6
            ][:3]
            result.top_som_class = "aromatic_hydroxylation"

        # ── 2. Metabolite generation ────────────────────────────────
        if self.use_ml and HAS_METAB:
            try:
                gen = MetabolicTreeGenerator(max_depth=2, max_metabolites=20)
                metabolites = gen.generate(smiles)
                result.metabolites      = [{"smiles": m.smiles, "phase": m.phase}
                                           for m in metabolites]
                result.n_metabolites    = len(metabolites)
                result.reactive_metabolites = sum(
                    1 for m in metabolites if m.is_reactive
                )
            except Exception:
                pass

        if result.n_metabolites == 0:
            # Estimate from molecular complexity
            result.n_metabolites = max(2, int(mol.GetNumAtoms() / 8))

        # ── 3. CYP isoform classification ───────────────────────────
        if self.use_ml and HAS_CYP:
            try:
                feat = cyp_features(smiles)
                if feat is not None and _cyp_model is not None:
                    probs = _cyp_model.predict_proba(smiles)
                    result.cyp_profile = probs
                    if probs:
                        result.primary_cyp = max(probs, key=probs.get)
            except Exception:
                pass

        if not result.cyp_profile:
            # Rule-based CYP assignment
            mw   = Descriptors.ExactMolWt(mol)
            logp = Crippen.MolLogP(mol)
            n_ar = rdMolDescriptors.CalcNumAromaticRings(mol)
            result.cyp_profile = self._rule_based_cyp(smiles, mw, logp, n_ar)
            result.primary_cyp = max(result.cyp_profile, key=result.cyp_profile.get)

        # ── 4. Toxicity profiling ───────────────────────────────────
        tox = self._tox
        if tox is not None:
            try:
                profile = tox.full_toxicity_profile(smiles)
                result.tox_probs         = profile["tox_probs"]
                result.structural_alerts = profile["structural_alerts"]
                result.safety_score      = profile["safety_score"]
            except Exception:
                tox = None

        if tox is None:
            sa = _standalone.predict(smiles)
            result.structural_alerts = [{"name": a} for a in sa.get("structural_alerts", [])]
            result.safety_score      = sa.get("safety_score", 80.0)
            result.tox_probs = {
                "DILI":            0.3 if "quinone" in sa.get("structural_alerts", []) else 0.1,
                "mutagenicity":    0.3 if "nitro" in sa.get("structural_alerts", []) else 0.05,
                "hERG_inhibition": 0.2,
                "hepatotoxicity":  0.15,
                "GSH_trapping":    0.2 if "michael_acceptor" in sa.get("structural_alerts", []) else 0.05,
            }

        # ── 5. PBPK / PK parameters ────────────────────────────────
        if self.use_ml and HAS_PBPK:
            try:
                pk    = PKParameters.from_smiles(smiles, dose_mg=dose_mg)
                df_pk = simulate_pk(pk, t_end=24.0)
                nca   = compute_nca_parameters(df_pk, dose_mg)
                result.pk_params = {
                    "Cmax_ugmL":   nca.get("Cmax_mgL", 0.0),
                    "Tmax_h":      nca.get("Tmax_h", 0.0),
                    "AUC_ughmL":   nca.get("AUC_mgLh", 0.0),
                    "t_half_h":    nca.get("t_half_h", 0.0),
                    "F_oral":      pk.F_oral,
                    "Vd_L_kg":     pk.Vd,
                    "CL_Lh":       nca.get("CL_Lh", 0.0),
                }
            except Exception:
                pass

        if not result.pk_params:
            sa = _standalone.predict(smiles)
            result.pk_params = {
                "Cmax_ugmL":  dose_mg * sa.get("F_oral", 0.6) / 10.0,
                "Tmax_h":     1.5,
                "AUC_ughmL":  dose_mg * sa.get("F_oral", 0.6) * sa.get("t_half_h", 4.0) / 5.0,
                "t_half_h":   sa.get("t_half_h", 4.0),
                "F_oral":     sa.get("F_oral", 0.6),
                "Vd_L_kg":    sa.get("Vd_L_kg", 0.7),
            }

        # ── 6. Scoring + soft spots + recommendations ───────────────
        result.overall_score    = self._compute_overall_score(result)
        result.soft_spots       = self._identify_soft_spots(result)
        result.recommendations  = self._generate_recommendations(result)

        return result

    # ------------------------------------------------------------------
    def _rule_based_cyp(self, smiles: str, mw: float, logp: float, n_ar: int) -> Dict[str, float]:
        """Assign CYP substrate probability from physicochemical rules."""
        mol = Chem.MolFromSmiles(smiles)
        hba = rdMolDescriptors.CalcNumHBA(mol) if mol else 4
        return {
            "CYP3A4": min(0.95, 0.3 + (mw > 400) * 0.3 + (n_ar >= 3) * 0.2 + (logp > 3) * 0.15),
            "CYP2D6": min(0.90, 0.1 + (n_ar >= 2) * 0.3 + (logp > 2) * 0.2),
            "CYP2C9": min(0.90, 0.1 + (hba >= 2) * 0.25 + (logp > 2) * 0.2),
            "CYP2C19": min(0.85, 0.1 + (hba >= 3) * 0.3 + (n_ar >= 2) * 0.15),
            "CYP1A2": min(0.85, 0.1 + (n_ar >= 3) * 0.35 + (mw < 300) * 0.2),
        }

    # ------------------------------------------------------------------
    def _compute_overall_score(self, r: ADMETResult) -> float:
        """Weighted composite score 0–100."""
        # Safety component (30%)
        safety_c = r.safety_score * 0.30

        # PK component (25%) — penalise very low bioavailability or t½ extremes
        f_oral  = r.pk_params.get("F_oral", 0.5)
        t_half  = r.pk_params.get("t_half_h", 4.0)
        pk_ok   = min(1.0, f_oral) * min(1.0, 1.0 - abs(t_half - 8) / 20)
        pk_c    = max(0.0, pk_ok) * 100 * 0.25

        # CYP component (25%) — prefer fewer high-probability substrates
        top_cyp = max(r.cyp_profile.values()) if r.cyp_profile else 0.5
        cyp_c   = (1.0 - top_cyp * 0.5) * 100 * 0.25

        # Metabolite risk (20%) — penalise reactive metabolites
        react_frac = r.reactive_metabolites / max(1, r.n_metabolites)
        met_c = (1.0 - react_frac) * 100 * 0.20

        return round(safety_c + pk_c + cyp_c + met_c, 1)

    # ------------------------------------------------------------------
    def _identify_soft_spots(self, r: ADMETResult) -> List[str]:
        spots = []

        # High-priority SoM atoms
        if r.som_atoms:
            top_score = max(r.som_scores.values()) if r.som_scores else 0.0
            if top_score > 0.6:
                spots.append(
                    f"High SoM at atom {r.som_atoms[0]} "
                    f"(score {top_score:.2f}, {r.top_som_class})"
                )

        # Reactive metabolites
        if r.reactive_metabolites > 0:
            spots.append(
                f"{r.reactive_metabolites} reactive metabolite(s) detected "
                f"(GSH/protein trapping risk)"
            )

        # Low bioavailability
        if r.pk_params.get("F_oral", 1.0) < 0.30:
            spots.append(
                f"Low oral bioavailability ({r.pk_params['F_oral']:.0%}) "
                "— high first-pass metabolism"
            )

        # Short half-life
        if r.pk_params.get("t_half_h", 10.0) < 2.0:
            spots.append(
                f"Short half-life ({r.pk_params['t_half_h']:.1f} h) "
                "— rapid metabolic clearance"
            )

        # Toxicity flags
        high_tox = [ep for ep, p in r.tox_probs.items() if p > 0.5]
        if high_tox:
            spots.append(f"High predicted {', '.join(high_tox)} risk")

        # Structural alerts
        if len(r.structural_alerts) >= 2:
            alert_names = ", ".join(a.get("name", a.get("id", "?"))
                                    for a in r.structural_alerts[:3])
            spots.append(f"Structural alerts: {alert_names}")

        # Dominant CYP
        if r.primary_cyp:
            p = r.cyp_profile.get(r.primary_cyp, 0.0)
            if p > 0.75:
                spots.append(
                    f"Strong {r.primary_cyp} substrate (p={p:.2f}) "
                    "— DDI liability"
                )

        return spots if spots else ["No major soft spots identified"]

    # ------------------------------------------------------------------
    def _generate_recommendations(self, r: ADMETResult) -> List[str]:
        recs = []
        probs = r.tox_probs
        pk    = r.pk_params

        if r.reactive_metabolites > 0:
            recs.append(
                "Block reactive SoM: introduce F/Cl at aromatic soft spot "
                "or replace phenol with fluorophenol to prevent quinone formation"
            )

        if probs.get("hERG_inhibition", 0) > 0.4:
            recs.append(
                "Reduce hERG risk: decrease basicity of amine (pKa target <8), "
                "add polar group to reduce logP, consider truncated lipophilic tail"
            )

        if pk.get("F_oral", 1.0) < 0.35:
            recs.append(
                "Improve oral bioavailability: add solubilising group (morpholine/piperazine), "
                "reduce rotatable bonds, check permeability vs solubility balance"
            )

        if probs.get("DILI", 0) > 0.4:
            recs.append(
                "Mitigate DILI risk: remove/replace structural alerts (quinone, michael acceptor), "
                "reduce daily dose burden, investigate metabolite profiling in hepatocytes"
            )

        if r.primary_cyp == "CYP3A4" and r.cyp_profile.get("CYP3A4", 0) > 0.75:
            recs.append(
                "CYP3A4 DDI liability: consider deuterium labelling at primary SoM, "
                "add steric bulk at metabolic hotspot, or design for renal excretion pathway"
            )

        if pk.get("t_half_h", 10.0) < 3.0:
            recs.append(
                "Extend half-life: block primary SoM with fluorine/methyl, "
                "or use prodrug strategy for sustained release"
            )

        if not recs:
            recs.append("Profile looks acceptable — advance to in vitro metabolic stability assay")

        return recs

    # ------------------------------------------------------------------
    def screen_library(
        self,
        smiles_list: List[str],
        names: Optional[List[str]] = None,
        dose_mg: float = 100.0,
    ) -> pd.DataFrame:
        """Screen multiple compounds. Returns ranked DataFrame."""
        if names is None:
            names = [f"Cpd_{i+1}" for i in range(len(smiles_list))]

        rows = []
        for smi, name in zip(smiles_list, names):
            try:
                r = self.run(smi, name=name, dose_mg=dose_mg)
                rows.append({
                    "name":                name,
                    "smiles":              smi,
                    "overall_score":       r.overall_score,
                    "safety_score":        r.safety_score,
                    "n_metabolites":       r.n_metabolites,
                    "reactive_metabolites":r.reactive_metabolites,
                    "primary_cyp":         r.primary_cyp,
                    "F_oral":              r.pk_params.get("F_oral", 0.0),
                    "t_half_h":            r.pk_params.get("t_half_h", 0.0),
                    "DILI_prob":           r.tox_probs.get("DILI", 0.0),
                    "hERG_prob":           r.tox_probs.get("hERG_inhibition", 0.0),
                    "n_alerts":            len(r.structural_alerts),
                    "top_soft_spot":       r.soft_spots[0] if r.soft_spots else "",
                })
            except Exception as e:
                rows.append({"name": name, "smiles": smi, "overall_score": 0.0,
                             "error": str(e)})

        df = pd.DataFrame(rows).sort_values("overall_score", ascending=False)
        df.index = range(1, len(df) + 1)
        return df


# ── Text report ───────────────────────────────────────────────────────────────

def report_to_text(r: ADMETResult) -> str:
    sep = "═" * 62
    lines = [
        sep,
        f"  ADMET PROFILE: {r.name}",
        sep,
        f"  SMILES : {r.smiles}",
        f"  Overall ADMET Score : {r.overall_score:.1f} / 100",
        "",
        "  [SITES OF METABOLISM]",
    ]

    if r.som_atoms:
        atoms_str = ", ".join(str(a) for a in r.som_atoms[:5])
        lines.append(f"    Top SoM atoms  : {atoms_str}")
        lines.append(f"    Reaction class : {r.top_som_class or '—'}")
    else:
        lines.append("    No SoM data available")

    lines += [
        "",
        "  [METABOLIC FATE]",
        f"    Metabolites generated  : {r.n_metabolites}",
        f"    Reactive intermediates : {r.reactive_metabolites}",
        f"    Primary CYP            : {r.primary_cyp or '—'}",
    ]

    if r.cyp_profile:
        top3 = sorted(r.cyp_profile.items(), key=lambda x: x[1], reverse=True)[:3]
        for cyp, p in top3:
            lines.append(f"      {cyp:<12} {p:.2f}")

    lines += [
        "",
        "  [TOXICITY PROFILE]",
        f"    Safety Score    : {r.safety_score:.1f} / 100",
        f"    Structural alerts: {len(r.structural_alerts)}",
    ]
    for ep, p in r.tox_probs.items():
        flag = " ⚠" if p > 0.5 else ""
        lines.append(f"    {ep:<22} {p:.2f}{flag}")

    lines += [
        "",
        "  [PHARMACOKINETICS]",
        f"    Oral bioavailability : {r.pk_params.get('F_oral', 0)*100:.0f}%",
        f"    Predicted Cmax       : {r.pk_params.get('Cmax_ugmL', 0):.2f} µg/mL",
        f"    Predicted AUC        : {r.pk_params.get('AUC_ughmL', 0):.1f} µg·h/mL",
        f"    Half-life            : {r.pk_params.get('t_half_h', 0):.1f} h",
        "",
        "  [SOFT SPOTS]",
    ]
    for spot in r.soft_spots:
        lines.append(f"    • {spot}")

    lines += ["", "  [RECOMMENDATIONS]"]
    for rec in r.recommendations:
        lines.append(f"    • {rec}")

    lines.append(sep)
    return "\n".join(lines)


# ── Dashboard visualisation ───────────────────────────────────────────────────

def plot_admet_dashboard(
    results: List[ADMETResult],
    out_path: str = "admet_dashboard.png",
) -> None:
    """3-panel comparison dashboard for a list of ADMETResult objects."""
    names  = [r.name for r in results]
    scores = [r.overall_score for r in results]
    safety = [r.safety_score for r in results]
    f_oral = [r.pk_params.get("F_oral", 0.5) * 100 for r in results]
    n_met  = [r.n_metabolites for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("Integrated ADMET Comparison Dashboard", fontsize=15,
                 fontweight="bold", y=1.01)

    # ── Panel 1: Overall ADMET scores ───────────────────────────────
    ax = axes[0]
    colors = ["#388e3c" if s >= 70 else "#f57c00" if s >= 40 else "#d32f2f"
              for s in scores]
    bars = ax.barh(names, scores, color=colors, alpha=0.85, edgecolor="white")
    ax.set_xlim(0, 100)
    ax.axvline(70, color="#388e3c", lw=1.5, ls="--", alpha=0.6, label="Good (70)")
    ax.axvline(40, color="#f57c00", lw=1.5, ls="--", alpha=0.6, label="Acceptable (40)")
    for bar, val in zip(bars, scores):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", fontsize=9)
    ax.set_xlabel("Overall ADMET Score (0–100)")
    ax.set_title("Overall ADMET Score", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    # ── Panel 2: Safety vs bioavailability scatter ───────────────────
    ax = axes[1]
    sc = ax.scatter(f_oral, safety, s=[max(30, m * 15) for m in n_met],
                    c=scores, cmap="RdYlGn", vmin=0, vmax=100,
                    alpha=0.8, edgecolors="grey", linewidths=0.5)
    for r, x, y in zip(results, f_oral, safety):
        ax.annotate(r.name, (x, y), textcoords="offset points",
                    xytext=(4, 3), fontsize=7, alpha=0.85)
    ax.axhline(70, color="grey", ls="--", lw=1, alpha=0.5)
    ax.axvline(30, color="grey", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Oral Bioavailability (%)")
    ax.set_ylabel("Safety Score (0–100)")
    ax.set_title("Safety vs Bioavailability\n(bubble size = n_metabolites)",
                 fontsize=11, fontweight="bold")
    plt.colorbar(sc, ax=ax, label="Overall Score")

    # ── Panel 3: Radar overlay for top-3 ────────────────────────────
    ax3 = fig.add_subplot(1, 3, 3, polar=True)
    top3 = sorted(results, key=lambda r: r.overall_score, reverse=True)[:3]
    axes_labels = ["Absorption", "Metabolism\nSafety", "Tox\nSafety",
                   "PK\nProfile", "Selectivity"]
    N = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    pal = ["#1565c0", "#c62828", "#2e7d32"]
    for r, color in zip(top3, pal):
        fa   = r.pk_params.get("F_oral", 0.5)
        react_frac = r.reactive_metabolites / max(1, r.n_metabolites)
        tox_risk = max(r.tox_probs.values()) if r.tox_probs else 0.3
        pk_score = min(1.0, r.pk_params.get("t_half_h", 4.0) / 12.0)
        top_cyp  = max(r.cyp_profile.values()) if r.cyp_profile else 0.5

        vals = [
            fa,
            1.0 - react_frac,
            r.safety_score / 100.0,
            pk_score,
            1.0 - top_cyp * 0.5,
        ]
        vals += vals[:1]
        ax3.plot(angles, vals, "o-", lw=2, color=color, label=r.name)
        ax3.fill(angles, vals, alpha=0.08, color=color)

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(axes_labels, fontsize=9)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([0.25, 0.5, 0.75])
    ax3.set_yticklabels(["0.25", "0.5", "0.75"], fontsize=7)
    ax3.set_title("ADMET Radar — Top 3 Compounds", fontsize=11,
                  fontweight="bold", pad=20)
    ax3.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Dashboard saved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  INTEGRATED ADMET PIPELINE")
    print("=" * 62)

    BENCHMARK = [
        ("CC(=O)Nc1ccc(O)cc1",                          "Acetaminophen"),
        ("O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",              "Diclofenac"),
        ("OC=1C=CC(=CC=1/C=C/c1ccc(O)cc1)O",            "Resveratrol"),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C",               "Caffeine"),
        ("CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",         "Testosterone"),
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O",                  "Ibuprofen"),
        ("COc1ccc2[nH]cc(CCNC(C)=O)c2c1",               "Melatonin"),
        ("CN(C)CCOC(c1ccccc1)c1ccccc1",                  "Diphenhydramine"),
    ]

    pipeline = ADMETPipeline(use_ml=True)
    results  = []

    print(f"\n{'Compound':<20} {'Score':>6} {'Safety':>7} {'F_oral':>7} "
          f"{'t½(h)':>6} {'CYP':>8} {'Reactive':>9}")
    print("-" * 70)

    for smi, name in BENCHMARK:
        r = pipeline.run(smi, name=name, dose_mg=100.0)
        results.append(r)
        print(
            f"{name:<20} {r.overall_score:>6.1f} {r.safety_score:>7.1f} "
            f"{r.pk_params.get('F_oral', 0)*100:>6.0f}% "
            f"{r.pk_params.get('t_half_h', 0):>6.1f} "
            f"{r.primary_cyp:>8} "
            f"{r.reactive_metabolites:>9}"
        )

    # Print full report for top compound
    best = max(results, key=lambda r: r.overall_score)
    print(f"\n\nFull report for highest-scoring compound:")
    print(report_to_text(best))

    # Screen as library
    print("\n\nLibrary screening results (ranked):")
    df = pipeline.screen_library(
        [s for s, _ in BENCHMARK],
        names=[n for _, n in BENCHMARK],
    )
    print(df[["name", "overall_score", "safety_score", "F_oral",
              "t_half_h", "primary_cyp", "reactive_metabolites",
              "DILI_prob", "hERG_prob"]].to_string(index=True))

    # Dashboard
    out = os.path.join(os.path.dirname(__file__) or ".", "admet_dashboard.png")
    print("\nGenerating ADMET dashboard…")
    plot_admet_dashboard(results, out_path=out)
    print("\nDone.")
