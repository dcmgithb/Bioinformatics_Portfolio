"""Data validation and content-hash generation for the bioetl-pipeline.

Provides:
- Null/required-field checks
- Amino acid sequence validation
- SHA-256 content hash computation for deduplication
- Duplicate detection via content_hash
- ``validate_dataframe`` orchestration function

All functions are pure (no I/O) except for structlog calls.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from pipeline.logger import get_logger

_log = get_logger(__name__)

# Valid amino acid single-letter codes (no ambiguity codes)
VALID_AA: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")

# Required columns per source (used by validate_dataframe dispatch)
_REQUIRED_COLS_BY_SOURCE: dict[str, list[str]] = {
    "OAS": ["source", "full_vh_aa"],
    "ChEMBL": ["name"],
    "UniProt": ["gene_name", "protein_name"],
}

# Columns to use when computing content_hash per source
_HASH_COLS_BY_SOURCE: dict[str, list[str]] = {
    "OAS": ["full_vh_aa", "vh_gene"],
    "ChEMBL": ["chembl_id", "name"],
    "UniProt": ["uniprot_id", "gene_name"],
}


@dataclass
class ValidationReport:
    """Summary statistics from a single validation run.

    Attributes:
        total_records: Number of input records before validation.
        null_failures: Rows removed due to null in a required column.
        sequence_failures: Rows removed due to invalid amino acid sequence.
        duplicates_detected: Rows removed as duplicate content_hash.
        passed: Records that passed all validation checks.
        failed_indices: Original integer indices of rows that failed.
    """

    total_records: int
    null_failures: int
    sequence_failures: int
    duplicates_detected: int
    passed: int
    failed_indices: list[int] = field(default_factory=list)


def check_nulls(df: pd.DataFrame, required_cols: list[str]) -> list[int]:
    """Return integer indices of rows with null values in any required column.

    Only checks columns that actually exist in ``df``; missing columns are
    silently skipped so that callers do not need to guard against column
    absence.

    Args:
        df: Input DataFrame.
        required_cols: List of column names that must be non-null.

    Returns:
        Sorted list of row indices (``df.index`` positions) with nulls.
    """
    present_cols = [c for c in required_cols if c in df.columns]
    if not present_cols:
        return []
    null_mask = df[present_cols].isnull().any(axis=1)
    return sorted(df.index[null_mask].tolist())


def validate_aa_sequence(seq: str | None) -> bool:
    """Return ``True`` if ``seq`` is a non-empty, valid amino acid sequence.

    A sequence is valid if:
    - It is not ``None`` and not empty after stripping whitespace.
    - Every character is a member of the 20 standard amino acid codes.

    Args:
        seq: Amino acid sequence string to validate.

    Returns:
        ``True`` if valid, ``False`` otherwise.
    """
    if seq is None:
        return False
    cleaned = seq.strip().upper()
    if not cleaned:
        return False
    return all(ch in VALID_AA for ch in cleaned)


def add_content_hash(df: pd.DataFrame, hash_cols: list[str]) -> pd.DataFrame:
    """Add a ``content_hash`` column to ``df``.

    The hash is SHA-256 of the concatenation of the values in ``hash_cols``
    (converted to strings, separated by ``|``), taking the first 32 hex
    characters.  Null values are represented as the string ``"null"``
    to ensure a stable hash even when a column is missing.

    Args:
        df: Input DataFrame.
        hash_cols: Ordered list of column names whose values form the hash input.

    Returns:
        A copy of ``df`` with a new ``content_hash`` column appended.

    Raises:
        ValueError: If ``hash_cols`` is empty.
    """
    if not hash_cols:
        raise ValueError("hash_cols must not be empty")

    result = df.copy()

    def _row_hash(row: pd.Series) -> str:  # type: ignore[type-arg]
        parts: list[str] = []
        for col in hash_cols:
            val = row.get(col) if col in row.index else None
            parts.append("null" if val is None or (isinstance(val, float) and pd.isna(val)) else str(val))
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    result["content_hash"] = result.apply(_row_hash, axis=1)
    return result


def detect_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows with duplicate ``content_hash`` values.

    Keeps the first occurrence of each content_hash.  If ``df`` does not
    contain a ``content_hash`` column, returns the original DataFrame
    unchanged with 0 duplicates reported.

    Args:
        df: Input DataFrame, expected to have a ``content_hash`` column.

    Returns:
        A 2-tuple of ``(deduplicated_df, n_dropped)`` where ``n_dropped`` is
        the number of rows removed as duplicates.
    """
    if "content_hash" not in df.columns:
        _log.warning("detect_duplicates_no_hash_column")
        return df, 0

    before = len(df)
    deduped = df.drop_duplicates(subset=["content_hash"], keep="first").reset_index(drop=True)
    n_dropped = before - len(deduped)
    return deduped, n_dropped


def validate_dataframe(
    df: pd.DataFrame, source: str
) -> tuple[pd.DataFrame, ValidationReport]:
    """Run all validation steps on a transformed DataFrame.

    Steps:
    1. Check for nulls in required columns for the given source.
    2. If a ``full_vh_aa`` or ``cdr3_aa`` column exists, validate amino acid
       sequences in those columns.
    3. Add ``content_hash`` column using source-specific hash columns.
    4. Detect and remove duplicate rows based on ``content_hash``.
    5. Return the clean DataFrame and a :class:`ValidationReport`.

    Args:
        df: Input DataFrame from a transformer function.
        source: Source name (``"OAS"``, ``"ChEMBL"``, or ``"UniProt"``).

    Returns:
        A 2-tuple of ``(clean_df, report)`` where ``clean_df`` is the
        validated, deduplicated DataFrame ready for loading.
    """
    if df.empty:
        report = ValidationReport(
            total_records=0,
            null_failures=0,
            sequence_failures=0,
            duplicates_detected=0,
            passed=0,
            failed_indices=[],
        )
        return df, report

    total = len(df)
    failed_indices: list[int] = []

    # ----------------------------------------------------------------
    # Step 1: Null check on required columns
    # ----------------------------------------------------------------
    required_cols = _REQUIRED_COLS_BY_SOURCE.get(source, [])
    null_idx = check_nulls(df, required_cols)
    null_failures = len(null_idx)
    if null_idx:
        _log.warning(
            "validation_null_failures",
            source=source,
            n_failed=null_failures,
            example_indices=null_idx[:5],
        )
        failed_indices.extend(null_idx)

    # Remove null-failing rows
    df_clean = df.drop(index=null_idx).reset_index(drop=True)

    # ----------------------------------------------------------------
    # Step 2: Amino acid sequence validation
    # ----------------------------------------------------------------
    seq_failed_indices: list[int] = []
    for seq_col in ("full_vh_aa", "cdr3_aa"):
        if seq_col in df_clean.columns:
            # Only validate non-null values in this column
            non_null_mask = df_clean[seq_col].notna()
            invalid_mask = non_null_mask & df_clean[seq_col].apply(
                lambda s: not validate_aa_sequence(s) if isinstance(s, str) else False
            )
            bad_indices = df_clean.index[invalid_mask].tolist()
            seq_failed_indices.extend(bad_indices)

    # Deduplicate seq_failed_indices across columns
    seq_failed_indices = sorted(set(seq_failed_indices))
    sequence_failures = len(seq_failed_indices)
    if seq_failed_indices:
        _log.warning(
            "validation_sequence_failures",
            source=source,
            n_failed=sequence_failures,
            example_indices=seq_failed_indices[:5],
        )
        failed_indices.extend(seq_failed_indices)

    df_clean = df_clean.drop(index=seq_failed_indices).reset_index(drop=True)

    # ----------------------------------------------------------------
    # Step 3: Add content_hash
    # ----------------------------------------------------------------
    hash_cols = _HASH_COLS_BY_SOURCE.get(source, list(df_clean.columns[:2]))
    # Only use columns that exist in the cleaned df
    hash_cols_present = [c for c in hash_cols if c in df_clean.columns]
    if not hash_cols_present:
        hash_cols_present = [df_clean.columns[0]]
    df_clean = add_content_hash(df_clean, hash_cols_present)

    # ----------------------------------------------------------------
    # Step 4: Deduplicate by content_hash
    # ----------------------------------------------------------------
    df_clean, duplicates_detected = detect_duplicates(df_clean)

    passed = len(df_clean)
    report = ValidationReport(
        total_records=total,
        null_failures=null_failures,
        sequence_failures=sequence_failures,
        duplicates_detected=duplicates_detected,
        passed=passed,
        failed_indices=sorted(set(failed_indices)),
    )

    _log.info(
        "validation_complete",
        source=source,
        total=total,
        null_failures=null_failures,
        sequence_failures=sequence_failures,
        duplicates_detected=duplicates_detected,
        passed=passed,
    )

    return df_clean, report
