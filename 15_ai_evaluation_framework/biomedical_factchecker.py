"""
biomedical_factchecker.py — Entity extraction, knowledge-base verification,
and hallucination rate metrics for AI genomics/clinical outputs (Project 15).
"""

from __future__ import annotations

import re
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from utils.common_functions import set_global_seed, PALETTES
except ImportError:
    def set_global_seed(s=42):
        np.random.seed(s)
    PALETTES = {"young": "#2196F3", "aged": "#F44336", "accent": "#4CAF50"}

# ──────────────────────────────────────────────────────────────────────────────
# Curated knowledge base
# ──────────────────────────────────────────────────────────────────────────────

GENE_KB: Dict[str, Dict] = {
    "BRCA1": {"function": "DNA repair, homologous recombination", "chromosome": "17q21",
              "associated_conditions": ["breast cancer", "ovarian cancer"],
              "pathogenic_variants": ["BRCA1:c.5266dupC", "BRCA1:c.68_69del"]},
    "BRCA2": {"function": "DNA repair, homologous recombination", "chromosome": "13q12",
              "associated_conditions": ["breast cancer", "pancreatic cancer"],
              "pathogenic_variants": ["BRCA2:c.5946delT", "BRCA2:c.6174delT"]},
    "APOE":  {"function": "lipid transport and metabolism", "chromosome": "19q13",
              "associated_conditions": ["Alzheimer disease", "cardiovascular disease"],
              "pathogenic_variants": ["APOE:p.Arg158Cys", "APOE:p.Cys112Arg"]},
    "TP53":  {"function": "tumour suppressor, cell cycle regulation", "chromosome": "17p13",
              "associated_conditions": ["Li-Fraumeni syndrome", "multiple cancers"],
              "pathogenic_variants": ["TP53:p.Arg175His", "TP53:p.Arg273His"]},
    "CFTR":  {"function": "chloride channel, ion transport", "chromosome": "7q31",
              "associated_conditions": ["cystic fibrosis"],
              "pathogenic_variants": ["CFTR:p.Phe508del", "CFTR:p.Gly542X"]},
    "KRAS":  {"function": "RAS GTPase signalling", "chromosome": "12p12",
              "associated_conditions": ["NSCLC", "colorectal cancer", "pancreatic cancer"],
              "pathogenic_variants": ["KRAS:p.Gly12Val", "KRAS:p.Gly12Asp"]},
    "EGFR":  {"function": "receptor tyrosine kinase, EGF signalling", "chromosome": "7p11",
              "associated_conditions": ["NSCLC"],
              "pathogenic_variants": ["EGFR:p.Leu858Arg", "EGFR:p.Glu746_Ala750del"]},
    "BRAF":  {"function": "serine/threonine kinase, MAPK pathway", "chromosome": "7q34",
              "associated_conditions": ["melanoma", "thyroid cancer"],
              "pathogenic_variants": ["BRAF:p.Val600Glu"]},
    "MLH1":  {"function": "DNA mismatch repair", "chromosome": "3p22",
              "associated_conditions": ["Lynch syndrome", "colorectal cancer"],
              "pathogenic_variants": ["MLH1:c.1A>T", "MLH1:c.676C>T"]},
    "PTEN":  {"function": "phosphatase, PI3K/AKT pathway regulation", "chromosome": "10q23",
              "associated_conditions": ["Cowden syndrome", "endometrial cancer"],
              "pathogenic_variants": ["PTEN:p.Arg130Gln", "PTEN:c.802-2A>G"]},
}

VARIANT_PATHOGENICITY_KB: Dict[str, str] = {
    "BRCA1:c.5266dupC":       "pathogenic",
    "BRCA1:c.68_69del":       "pathogenic",
    "BRCA2:c.5946delT":       "pathogenic",
    "BRCA2:c.6174delT":       "pathogenic",
    "APOE:p.Arg158Cys":       "risk_factor",
    "APOE:p.Cys112Arg":       "risk_factor",
    "TP53:p.Arg175His":       "pathogenic",
    "TP53:p.Arg273His":       "pathogenic",
    "CFTR:p.Phe508del":       "pathogenic",
    "CFTR:p.Gly542X":         "pathogenic",
    "KRAS:p.Gly12Val":        "pathogenic",
    "KRAS:p.Gly12Asp":        "pathogenic",
    "EGFR:p.Leu858Arg":       "pathogenic",
    "EGFR:p.Glu746_Ala750del":"pathogenic",
    "BRAF:p.Val600Glu":       "pathogenic",
    "MLH1:c.1A>T":            "pathogenic",
    "PTEN:p.Arg130Gln":       "likely_pathogenic",
}

DRUG_GENE_KB: Dict[str, Dict] = {
    "warfarin":    {"primary_targets": ["CYP2C9", "VKORC1"], "pgx_level": "1A",
                   "interaction_type": "metabolism", "effect": "altered anticoagulation"},
    "clopidogrel": {"primary_targets": ["CYP2C19"], "pgx_level": "1A",
                   "interaction_type": "activation", "effect": "reduced antiplatelet effect"},
    "tamoxifen":   {"primary_targets": ["CYP2D6"], "pgx_level": "1A",
                   "interaction_type": "metabolism", "effect": "reduced efficacy"},
    "codeine":     {"primary_targets": ["CYP2D6"], "pgx_level": "1A",
                   "interaction_type": "activation", "effect": "morphine conversion"},
    "simvastatin": {"primary_targets": ["SLCO1B1"], "pgx_level": "1A",
                   "interaction_type": "transport", "effect": "myopathy risk"},
    "irinotecan":  {"primary_targets": ["UGT1A1"], "pgx_level": "1A",
                   "interaction_type": "metabolism", "effect": "toxicity risk"},
    "olaparib":    {"primary_targets": ["BRCA1", "BRCA2"], "pgx_level": "1A",
                   "interaction_type": "synthetic_lethality", "effect": "PARP inhibitor response"},
    "erlotinib":   {"primary_targets": ["EGFR"], "pgx_level": "1A",
                   "interaction_type": "targeted_therapy", "effect": "EGFR TKI response"},
    "vemurafenib": {"primary_targets": ["BRAF"], "pgx_level": "1A",
                   "interaction_type": "targeted_therapy", "effect": "BRAF V600E inhibition"},
    "pembrolizumab":{"primary_targets": ["PDCD1"], "pgx_level": "2A",
                   "interaction_type": "immune_checkpoint", "effect": "MSI-H/TMB-H response"},
}

BIOMARKER_RANGES_KB: Dict[str, Dict] = {
    "HbA1c":       {"unit": "%",       "normal": (4.0, 5.6),  "prediabetes": (5.7, 6.4), "diabetes": (6.5, 15.0)},
    "LDL":         {"unit": "mg/dL",   "optimal": (0, 99),    "borderline": (100, 129),  "high": (130, 500)},
    "HDL":         {"unit": "mg/dL",   "low": (0, 39),        "normal": (40, 59),        "high": (60, 120)},
    "eGFR":        {"unit": "mL/min",  "normal": (60, 130),   "CKD3": (30, 59),          "CKD4": (15, 29)},
    "creatinine":  {"unit": "mg/dL",   "normal_m": (0.74, 1.35), "normal_f": (0.59, 1.04)},
    "PSA":         {"unit": "ng/mL",   "normal": (0, 4.0),    "elevated": (4.0, 10.0),   "high": (10.0, 1000)},
    "troponin_I":  {"unit": "ng/mL",   "normal": (0, 0.04),   "elevated": (0.04, 2.0)},
    "CA_125":      {"unit": "U/mL",    "normal": (0, 35),     "elevated": (35, 10000)},
    "AFP":         {"unit": "ng/mL",   "normal": (0, 10),     "elevated": (10, 500)},
    "TSH":         {"unit": "mIU/L",   "normal": (0.4, 4.0),  "hypo": (4.0, 100), "hyper": (0.0, 0.4)},
}

CLINICAL_GUIDELINES_KB: Dict[str, str] = {
    "BRCA1 pathogenic breast cancer screening":
        "Annual mammogram and MRI starting age 25; consider prophylactic mastectomy (NCCN 2024)",
    "EGFR L858R NSCLC first-line treatment":
        "Osimertinib preferred over erlotinib/gefitinib as first-line (NCCN NSCLC 2024)",
    "BRAF V600E melanoma treatment":
        "BRAF+MEK inhibitor combination (dabrafenib+trametinib) preferred (ASCO 2023)",
    "Lynch syndrome MSI-H colorectal cancer immunotherapy":
        "Pembrolizumab approved for MSI-H/dMMR tumors regardless of tissue of origin (FDA 2017)",
    "HbA1c diabetes diagnosis":
        "HbA1c ≥6.5% on two separate occasions confirms diabetes (ADA 2024)",
    "LDL cardiovascular risk reduction":
        "High-intensity statin for ASCVD risk >20%; target LDL <70 mg/dL (ACC/AHA 2019)",
    "CKD eGFR staging":
        "CKD Stage 3a: eGFR 45-59; Stage 3b: 30-44; Stage 4: 15-29 (KDIGO 2022)",
    "APOE4 Alzheimer risk":
        "APOE ε4 homozygotes have 8-12x increased AD risk; not diagnostic alone (NIA-AA 2023)",
}

# ──────────────────────────────────────────────────────────────────────────────
# Entity extraction patterns
# ──────────────────────────────────────────────────────────────────────────────

_GENE_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(g) for g in GENE_KB) + r')\b'
)

_VARIANT_PATTERN = re.compile(
    r'\b[A-Z][A-Z0-9]+:'           # gene prefix
    r'(?:c\.[0-9_+\-*/delinsdup[A-Z>]+|'  # cDNA notation
    r'p\.[A-Za-z]{3}[0-9]+[A-Za-z]{3}|'   # protein 3-letter
    r'p\.[A-Z][0-9]+[A-Z*])'              # protein 1-letter
    r'\b'
)

_DRUG_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(d) for d in DRUG_GENE_KB) + r')\b',
    re.IGNORECASE,
)

_BIOMARKER_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(b) for b in BIOMARKER_RANGES_KB) + r')\b',
    re.IGNORECASE,
)

_NUMBER_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(%|mg/dL|ng/mL|mIU/L|U/mL|mL/min)')


@dataclass
class ExtractedEntities:
    genes: List[str]
    variants: List[str]
    drugs: List[str]
    biomarkers: List[str]
    numeric_values: List[Tuple[str, float, str]]   # (context, value, unit)


def extract_entities(text: str) -> ExtractedEntities:
    genes = list(dict.fromkeys(_GENE_PATTERN.findall(text)))
    variants = list(dict.fromkeys(_VARIANT_PATTERN.findall(text)))
    drugs = list(dict.fromkeys(d.lower() for d in _DRUG_PATTERN.findall(text)))
    biomarkers = list(dict.fromkeys(_BIOMARKER_PATTERN.findall(text)))

    numeric_values = []
    for m in _NUMBER_PATTERN.finditer(text):
        val = float(m.group(1))
        unit = m.group(2)
        start = max(0, m.start() - 20)
        ctx = text[start:m.start()].strip().split()[-3:]
        numeric_values.append((" ".join(ctx), val, unit))

    return ExtractedEntities(
        genes=genes, variants=variants, drugs=drugs,
        biomarkers=biomarkers, numeric_values=numeric_values,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Claim verification
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ClaimResult:
    claim_type: str
    entity: str
    stated_value: str
    kb_value: str
    status: str          # "verified" | "contradicted" | "not_found" | "partial"
    confidence: float    # 0.0 – 1.0
    explanation: str


def verify_gene_claim(gene: str, stated_function: str = "") -> ClaimResult:
    if gene not in GENE_KB:
        return ClaimResult("gene_function", gene, stated_function, "",
                           "not_found", 0.0, f"{gene} not in knowledge base")

    kb_func = GENE_KB[gene]["function"].lower()
    stated_lower = stated_function.lower()

    if not stated_function:
        return ClaimResult("gene_function", gene, "", kb_func,
                           "verified", 1.0, "Gene exists in KB")

    # Keyword overlap
    kb_words = set(re.split(r'\W+', kb_func))
    stated_words = set(re.split(r'\W+', stated_lower))
    overlap = kb_words & stated_words
    score = len(overlap) / max(len(kb_words), 1)

    if score >= 0.5:
        status, conf = "verified", min(0.9, 0.5 + score)
    elif score >= 0.2:
        status, conf = "partial", 0.5
    else:
        status, conf = "contradicted", max(0.1, score)

    return ClaimResult("gene_function", gene, stated_function, kb_func,
                       status, conf, f"Keyword overlap {score:.0%}")


def verify_variant_claim(variant: str, stated_pathogenicity: str = "") -> ClaimResult:
    if variant not in VARIANT_PATHOGENICITY_KB:
        return ClaimResult("variant_pathogenicity", variant, stated_pathogenicity, "",
                           "not_found", 0.0, f"Variant {variant} not in KB")

    kb_path = VARIANT_PATHOGENICITY_KB[variant]
    if not stated_pathogenicity:
        return ClaimResult("variant_pathogenicity", variant, "", kb_path,
                           "verified", 1.0, "Variant exists in KB")

    sp = stated_pathogenicity.lower().replace(" ", "_")
    if sp == kb_path:
        return ClaimResult("variant_pathogenicity", variant, stated_pathogenicity, kb_path,
                           "verified", 0.95, "Exact pathogenicity match")
    elif ("pathogenic" in sp and "pathogenic" in kb_path) or \
         ("benign" in sp and "benign" in kb_path):
        return ClaimResult("variant_pathogenicity", variant, stated_pathogenicity, kb_path,
                           "partial", 0.70, "Partial pathogenicity match")
    else:
        return ClaimResult("variant_pathogenicity", variant, stated_pathogenicity, kb_path,
                           "contradicted", 0.10, f"Stated '{sp}' but KB has '{kb_path}'")


def verify_drug_gene_claim(drug: str, stated_gene: str = "") -> ClaimResult:
    drug_lower = drug.lower()
    if drug_lower not in DRUG_GENE_KB:
        return ClaimResult("drug_gene_interaction", drug, stated_gene, "",
                           "not_found", 0.0, f"Drug {drug} not in KB")

    kb_targets = DRUG_GENE_KB[drug_lower]["primary_targets"]
    if not stated_gene:
        return ClaimResult("drug_gene_interaction", drug, "", str(kb_targets),
                           "verified", 1.0, "Drug exists in KB")

    stated_upper = stated_gene.upper()
    if stated_upper in kb_targets:
        return ClaimResult("drug_gene_interaction", drug, stated_gene, str(kb_targets),
                           "verified", 0.95, "Gene-drug interaction confirmed")
    else:
        return ClaimResult("drug_gene_interaction", drug, stated_gene, str(kb_targets),
                           "contradicted", 0.15,
                           f"{stated_gene} not among known targets {kb_targets}")


def verify_biomarker_value(
    biomarker: str, value: float, unit: str, stated_interpretation: str = ""
) -> ClaimResult:
    bm_upper = biomarker.upper().replace(" ", "_")
    match = next((k for k in BIOMARKER_RANGES_KB if k.upper() == bm_upper), None)

    if match is None:
        return ClaimResult("biomarker_range", biomarker, f"{value} {unit}", "",
                           "not_found", 0.0, f"Biomarker {biomarker} not in KB")

    ranges = BIOMARKER_RANGES_KB[match]
    interpretation = "unknown"
    for range_name, rng in ranges.items():
        if range_name == "unit":
            continue
        if isinstance(rng, tuple) and len(rng) == 2 and rng[0] <= value <= rng[1]:
            interpretation = range_name
            break

    if not stated_interpretation:
        return ClaimResult("biomarker_range", biomarker, f"{value} {unit}",
                           f"{interpretation} per KB", "verified", 0.9, "Value categorised")

    si = stated_interpretation.lower()
    if interpretation in si or si in interpretation:
        return ClaimResult("biomarker_range", biomarker, f"{value} {unit}",
                           interpretation, "verified", 0.90, "Interpretation matches KB")
    else:
        return ClaimResult("biomarker_range", biomarker, f"{value} {unit}",
                           interpretation, "contradicted", 0.10,
                           f"Stated '{si}' but KB says '{interpretation}'")


# ──────────────────────────────────────────────────────────────────────────────
# Main fact-checker class
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FactCheckResult:
    text_id: str
    entities: ExtractedEntities
    claim_results: List[ClaimResult]
    hallucination_rate: float       # fraction of contradicted claims
    verified_count: int
    contradicted_count: int
    not_found_count: int
    overall_confidence: float
    flags: List[str]


class BiomedicalFactChecker:
    def __init__(self) -> None:
        self._gene_kb = GENE_KB
        self._variant_kb = VARIANT_PATHOGENICITY_KB
        self._drug_kb = DRUG_GENE_KB
        self._biomarker_kb = BIOMARKER_RANGES_KB

    def check(self, text: str, text_id: str = "item") -> FactCheckResult:
        entities = extract_entities(text)
        claims: List[ClaimResult] = []

        # Verify all extracted genes
        for gene in entities.genes:
            claims.append(verify_gene_claim(gene))

        # Verify all extracted variants
        for variant in entities.variants:
            claims.append(verify_variant_claim(variant))

        # Verify all extracted drugs
        for drug in entities.drugs:
            claims.append(verify_drug_gene_claim(drug))

        # Verify numeric biomarker values
        for ctx, val, unit in entities.numeric_values:
            for bm in entities.biomarkers:
                if bm.lower() in ctx.lower() or not ctx:
                    claims.append(verify_biomarker_value(bm, val, unit))

        if not claims:
            return FactCheckResult(
                text_id=text_id, entities=entities, claim_results=[],
                hallucination_rate=0.0, verified_count=0,
                contradicted_count=0, not_found_count=0,
                overall_confidence=0.5, flags=["no_verifiable_claims"],
            )

        verified = [c for c in claims if c.status == "verified"]
        contradicted = [c for c in claims if c.status == "contradicted"]
        not_found = [c for c in claims if c.status == "not_found"]

        total = len(claims)
        hal_rate = len(contradicted) / total
        avg_conf = float(np.mean([c.confidence for c in claims]))

        flags = []
        if hal_rate > 0.30:
            flags.append("high_hallucination_rate")
        if len(not_found) / total > 0.50:
            flags.append("low_kb_coverage")
        if any(c.status == "contradicted" and c.claim_type == "variant_pathogenicity"
               for c in claims):
            flags.append("variant_pathogenicity_error")
        if any(c.status == "contradicted" and c.claim_type == "drug_gene_interaction"
               for c in claims):
            flags.append("drug_interaction_error")

        return FactCheckResult(
            text_id=text_id,
            entities=entities,
            claim_results=claims,
            hallucination_rate=hal_rate,
            verified_count=len(verified),
            contradicted_count=len(contradicted),
            not_found_count=len(not_found),
            overall_confidence=avg_conf,
            flags=flags,
        )

    def check_batch(self, texts: List[Dict]) -> pd.DataFrame:
        rows = []
        for item in texts:
            result = self.check(item["text"], text_id=item.get("item_id", "?"))
            rows.append({
                "item_id": result.text_id,
                "topic": item.get("topic", "unknown"),
                "n_claims": len(result.claim_results),
                "verified": result.verified_count,
                "contradicted": result.contradicted_count,
                "not_found": result.not_found_count,
                "hallucination_rate": result.hallucination_rate,
                "overall_confidence": result.overall_confidence,
                "flags": "|".join(result.flags),
            })
        return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic test responses
# ──────────────────────────────────────────────────────────────────────────────

RESPONSE_TEMPLATES = {
    "variant_interpretation": [
        "The variant BRCA1:c.5266dupC is classified as pathogenic and associated with "
        "hereditary breast and ovarian cancer syndrome. BRCA1 plays a role in DNA repair.",
        "KRAS:p.Gly12Val is a benign polymorphism with no clinical significance.",   # hallucination
        "TP53:p.Arg175His is a well-known pathogenic hotspot mutation affecting DNA binding.",
    ],
    "drug_gene_interaction": [
        "Warfarin metabolism is primarily mediated by CYP2C9, and VKORC1 variants affect "
        "dose requirements significantly.",
        "Clopidogrel is activated by BRCA1, making BRCA1 testing essential before prescribing.",  # hallucination
        "Tamoxifen efficacy is influenced by CYP2D6 metabolizer status.",
    ],
    "biomarker_reference": [
        "An HbA1c of 7.2% indicates diabetes according to current ADA guidelines.",
        "A PSA of 2.5 ng/mL is highly elevated and diagnostic for prostate cancer.",  # hallucination
        "eGFR of 45 mL/min places the patient in CKD stage 3a.",
    ],
    "clinical_guideline": [
        "Patients with EGFR L858R NSCLC should receive osimertinib as first-line therapy "
        "per current NCCN guidelines.",
        "BRAF V600E melanoma is treated with pembrolizumab monotherapy as standard of care.",  # hallucination
        "Lynch syndrome patients with MSI-H colorectal cancer may benefit from pembrolizumab.",
    ],
    "phenotype_association": [
        "APOE ε4 carriers have significantly increased risk of Alzheimer disease, "
        "though APOE status alone is not diagnostic.",
        "CFTR p.Phe508del causes a gain-of-function in the chloride channel.",  # hallucination
        "BRCA2 pathogenic variants are associated with breast and pancreatic cancer risk.",
    ],
}

TOPICS = list(RESPONSE_TEMPLATES.keys())


def generate_test_responses(n: int = 50, seed: int = 42) -> List[Dict]:
    rng = np.random.default_rng(seed)
    responses = []
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        templates = RESPONSE_TEMPLATES[topic]
        text = str(rng.choice(templates))
        responses.append({"item_id": f"resp_{i:03d}", "text": text, "topic": topic})
    return responses


# ──────────────────────────────────────────────────────────────────────────────
# Hallucination rate metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_hallucination_metrics(results_df: pd.DataFrame) -> Dict:
    metrics = {
        "overall_hallucination_rate": float(results_df["hallucination_rate"].mean()),
        "per_topic": results_df.groupby("topic")["hallucination_rate"].mean().to_dict(),
        "high_risk_items": results_df[results_df["hallucination_rate"] > 0.30]["item_id"].tolist(),
        "flagged_drug_errors": int(results_df["flags"].str.contains("drug_interaction_error").sum()),
        "flagged_variant_errors": int(results_df["flags"].str.contains("variant_pathogenicity_error").sum()),
        "mean_confidence": float(results_df["overall_confidence"].mean()),
        "items_with_no_claims": int((results_df["n_claims"] == 0).sum()),
    }
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_factcheck_results(
    results_df: pd.DataFrame,
    out_path: str = "figures/factcheck_results.png",
) -> str:
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#FAFAFA")

    # Panel 1: Hallucination rate by topic
    ax = axes[0]
    by_topic = results_df.groupby("topic")["hallucination_rate"].mean()
    colors = [PALETTES.get("accent", "#4CAF50") if r <= 0.20 else PALETTES.get("aged", "#F44336")
              for r in by_topic.values]
    ax.barh(by_topic.index, by_topic.values * 100, color=colors, edgecolor="white")
    ax.axvline(20, color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("Hallucination Rate (%)")
    ax.set_title("Hallucination Rate by Topic")
    ax.set_facecolor("#F5F5F5")

    # Panel 2: Claim status distribution (stacked bar per topic)
    ax = axes[1]
    topics = results_df["topic"].unique()
    verified_means = [results_df[results_df["topic"] == t]["verified"].mean() for t in topics]
    contradicted_means = [results_df[results_df["topic"] == t]["contradicted"].mean() for t in topics]
    not_found_means = [results_df[results_df["topic"] == t]["not_found"].mean() for t in topics]
    x = np.arange(len(topics))
    ax.bar(x, verified_means, label="Verified", color=PALETTES.get("accent", "#4CAF50"))
    ax.bar(x, contradicted_means, bottom=verified_means,
           label="Contradicted", color=PALETTES.get("aged", "#F44336"))
    ax.bar(x, not_found_means,
           bottom=[v + c for v, c in zip(verified_means, contradicted_means)],
           label="Not Found", color=PALETTES.get("neutral", "#9E9E9E"))
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in topics], fontsize=7)
    ax.set_ylabel("Mean Claims per Item")
    ax.set_title("Claim Verification Status by Topic")
    ax.legend(fontsize=8)
    ax.set_facecolor("#F5F5F5")

    # Panel 3: Confidence score distribution
    ax = axes[2]
    ax.hist(results_df["overall_confidence"], bins=15,
            color=PALETTES.get("young", "#2196F3"), edgecolor="white", alpha=0.8)
    ax.axvline(results_df["overall_confidence"].mean(), color=PALETTES.get("aged", "#F44336"),
               linestyle="--", linewidth=1.5, label=f"Mean={results_df['overall_confidence'].mean():.2f}")
    ax.set_xlabel("Overall Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Score Distribution")
    ax.legend(fontsize=8)
    ax.set_facecolor("#F5F5F5")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_global_seed(42)

    print("Generating test responses …")
    responses = generate_test_responses(n=50, seed=42)

    print("Running fact-checker …")
    checker = BiomedicalFactChecker()
    results_df = checker.check_batch(responses)

    metrics = compute_hallucination_metrics(results_df)
    print(f"\nOverall hallucination rate : {metrics['overall_hallucination_rate']:.1%}")
    print(f"Mean confidence            : {metrics['mean_confidence']:.2f}")
    print(f"Drug interaction errors    : {metrics['flagged_drug_errors']}")
    print(f"Variant pathogenicity errs : {metrics['flagged_variant_errors']}")
    print("\nPer-topic hallucination rates:")
    for topic, rate in sorted(metrics["per_topic"].items(), key=lambda x: -x[1]):
        bar = "█" * int(rate * 20)
        print(f"  {topic:<30}  {rate:.1%}  {bar}")

    os.makedirs("figures", exist_ok=True)
    img_path = plot_factcheck_results(results_df, out_path="figures/factcheck_results.png")
    print(f"\nPlot saved → {img_path}")
