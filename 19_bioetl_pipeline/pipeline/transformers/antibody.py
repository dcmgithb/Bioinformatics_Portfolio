"""Antibody sequence transformation functions for the bioetl-pipeline.

Provides sequence canonicalisation, VH gene normalisation, CDR3 length
parsing, and per-source DataFrame construction functions.

All functions are pure (no side effects) and fully typed.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any

import pandas as pd

from pipeline.logger import get_logger

_log = get_logger(__name__)

# Amino acids valid in antibody sequences (IUPAC single-letter, no ambiguity codes)
VALID_AA: set[str] = set("ACDEFGHIKLMNPQRSTVWY")

# Regex matching standard IGHV gene notation: IGHV#-##[*##]
_IGHV_PATTERN: re.Pattern[str] = re.compile(
    r"IGHV(\d{1,2})-(\d{1,3}[A-Z]?)(?:\*\d+)?",
    re.IGNORECASE,
)

# Regex for cleaning non-alpha characters from sequences
_NON_ALPHA: re.Pattern[str] = re.compile(r"[^A-Za-z]")


def canonicalise_sequence(seq: str | None) -> str | None:
    """Uppercase, strip whitespace and non-alphabetic characters from a sequence.

    Args:
        seq: Raw amino acid or nucleotide sequence string, or ``None``.

    Returns:
        Cleaned, uppercased sequence string, or ``None`` if the input is
        ``None`` or results in an empty string after cleaning.
    """
    if seq is None:
        return None
    cleaned = _NON_ALPHA.sub("", seq.strip()).upper()
    return cleaned if cleaned else None


def parse_cdr3_length(cdr3_aa: str | None) -> int | None:
    """Return the length of a CDR3 amino acid sequence, or ``None`` if invalid.

    Args:
        cdr3_aa: CDR3 amino acid sequence string, or ``None``.

    Returns:
        Integer length of the CDR3 sequence, or ``None`` if the input is
        ``None`` or empty.
    """
    if cdr3_aa is None:
        return None
    cleaned = canonicalise_sequence(cdr3_aa)
    if not cleaned:
        return None
    return len(cleaned)


def normalise_vh_gene(vh_gene: str | None) -> str | None:
    """Standardise VH gene name to IGHV#-## format.

    Strips allele suffixes (e.g. ``*01``), handles case insensitivity,
    and validates the result against the expected IGHV format.

    Args:
        vh_gene: Raw VH gene string from a data source (e.g. ``"IGHV3-23*01"``,
            ``"ighv1-69"``, ``"V3-23"``).

    Returns:
        Normalised IGHV gene name (e.g. ``"IGHV3-23"``), or ``None`` if the
        input cannot be recognised as a valid IGHV gene.
    """
    if vh_gene is None:
        return None

    candidate = vh_gene.strip().upper()

    # Try direct match first (with optional allele suffix)
    match = _IGHV_PATTERN.search(candidate)
    if match:
        family = match.group(1)
        gene_num = match.group(2)
        return f"IGHV{family}-{gene_num}"

    # Handle short form like "V3-23" or "3-23"
    short_match = re.search(r"V?(\d{1,2})-(\d{1,3}[A-Z]?)", candidate)
    if short_match:
        family = short_match.group(1)
        gene_num = short_match.group(2)
        return f"IGHV{family}-{gene_num}"

    _log.debug("vh_gene_normalisation_failed", raw=vh_gene)
    return None


def transform_oas_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Transform raw OAS API records to a standardised antibody DataFrame.

    Applies canonicalisation, VH gene normalisation, CDR3 length calculation,
    and assigns a ``source`` column.

    Args:
        records: Raw list of dicts from :class:`OASExtractor.extract`.

    Returns:
        A :class:`pd.DataFrame` with columns:
        ``source``, ``vh_gene``, ``dh_gene``, ``jh_gene``, ``cdr3_aa``,
        ``cdr3_length``, ``full_vh_aa``, ``isotype``, ``species``, ``study_id``.
    """
    if not records:
        _log.warning("transform_oas_records_empty_input")
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for rec in records:
        vh_raw = rec.get("VH") or rec.get("vh_gene")
        dh_raw = rec.get("DH") or rec.get("dh_gene")
        jh_raw = rec.get("JH") or rec.get("jh_gene")
        cdr3_raw = rec.get("CDRH3_aa") or rec.get("CDR3") or rec.get("cdr3_aa")
        full_vh_raw = rec.get("full_vh_aa") or rec.get("sequence_vh")

        cdr3_clean = canonicalise_sequence(cdr3_raw)
        full_vh_clean = canonicalise_sequence(full_vh_raw)

        rows.append(
            {
                "source": "OAS",
                "vh_gene": normalise_vh_gene(vh_raw),
                "dh_gene": dh_raw.strip().upper() if dh_raw else None,
                "jh_gene": jh_raw.strip().upper() if jh_raw else None,
                "cdr3_aa": cdr3_clean,
                "cdr3_length": parse_cdr3_length(cdr3_raw),
                "full_vh_aa": full_vh_clean,
                "isotype": rec.get("isotype"),
                "species": rec.get("species"),
                "study_id": rec.get("study_id"),
            }
        )

    df = pd.DataFrame(rows)
    _log.info("transform_oas_records_complete", n_rows=len(df))
    return df


def transform_chembl_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Transform raw ChEMBL API records to a standardised therapeutics DataFrame.

    Args:
        records: Raw list of dicts from :class:`ChEMBLExtractor.extract`.

    Returns:
        A :class:`pd.DataFrame` with columns:
        ``chembl_id``, ``name``, ``max_phase``, ``mechanism``,
        ``target_name``, ``sequence_or_smiles``.
    """
    if not records:
        _log.warning("transform_chembl_records_empty_input")
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for rec in records:
        name_raw = rec.get("name") or rec.get("pref_name") or "Unknown"
        rows.append(
            {
                "chembl_id": rec.get("chembl_id") or rec.get("molecule_chembl_id"),
                "name": name_raw.strip() if name_raw else "Unknown",
                "max_phase": rec.get("max_phase"),
                "mechanism": rec.get("mechanism"),
                "target_name": rec.get("target_name"),
                "sequence_or_smiles": rec.get("sequence_or_smiles"),
            }
        )

    df = pd.DataFrame(rows)
    _log.info("transform_chembl_records_complete", n_rows=len(df))
    return df


def transform_uniprot_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Transform raw UniProt API records to a standardised B cell markers DataFrame.

    Args:
        records: Raw list of dicts from :class:`UniProtExtractor.extract`.

    Returns:
        A :class:`pd.DataFrame` with columns:
        ``uniprot_id``, ``gene_name``, ``protein_name``, ``organism``,
        ``function_text``.
    """
    if not records:
        _log.warning("transform_uniprot_records_empty_input")
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for rec in records:
        gene_name_raw = rec.get("gene_name") or rec.get("gene_names") or "Unknown"
        protein_name_raw = rec.get("protein_name") or rec.get("protein_names") or "Unknown"
        rows.append(
            {
                "uniprot_id": rec.get("uniprot_id") or rec.get("accession"),
                "gene_name": gene_name_raw.strip() if gene_name_raw else "Unknown",
                "protein_name": protein_name_raw.strip() if protein_name_raw else "Unknown",
                "organism": rec.get("organism"),
                "function_text": rec.get("function_text"),
            }
        )

    df = pd.DataFrame(rows)
    _log.info("transform_uniprot_records_complete", n_rows=len(df))
    return df
