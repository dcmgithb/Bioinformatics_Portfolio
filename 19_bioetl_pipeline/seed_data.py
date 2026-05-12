"""Synthetic seed data generator for the bioetl-pipeline ingestion tables.

Generates 200 realistic antibody VH sequence records and inserts them
directly into ``ingested_antibody_sequences``, bypassing the external API
extractors. Useful for development and testing without internet access.

Usage::

    python seed_data.py --db-url postgresql://etl_user:changeme@localhost:5432/bioetl_db
    python seed_data.py   # reads DATABASE_URL from environment
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import uuid
from datetime import datetime
from typing import Any

# Ensure the pipeline package is importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from pipeline.database import get_engine, get_session_factory, session_scope  # noqa: E402
from pipeline.models import Base, IngestedAntibodySequence  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RANDOM_SEED: int = 42

_VH_GENES: list[str] = [
    "IGHV1-2", "IGHV1-3", "IGHV1-8", "IGHV1-18", "IGHV1-46", "IGHV1-69",
    "IGHV2-5", "IGHV2-26", "IGHV2-70",
    "IGHV3-7", "IGHV3-9", "IGHV3-11", "IGHV3-20", "IGHV3-21",
    "IGHV3-23", "IGHV3-30", "IGHV3-33", "IGHV3-48", "IGHV3-53",
    "IGHV3-64", "IGHV3-72", "IGHV3-74",
    "IGHV4-4", "IGHV4-28", "IGHV4-30", "IGHV4-34", "IGHV4-39",
    "IGHV4-59", "IGHV4-61",
    "IGHV5-51",
    "IGHV6-1",
]

_DH_GENES: list[str] = [
    "IGHD1-1", "IGHD2-2", "IGHD2-15", "IGHD3-3", "IGHD3-10",
    "IGHD3-16", "IGHD4-17", "IGHD5-5", "IGHD5-18", "IGHD6-13",
    "IGHD6-19", "IGHD7-27",
]

_JH_GENES: list[str] = ["IGHJ1", "IGHJ2", "IGHJ3", "IGHJ4", "IGHJ5", "IGHJ6"]

_ISOTYPES: list[str] = [
    "IgG1", "IgG2", "IgG3", "IgG4", "IgM", "IgA1", "IgA2",
]
_ISOTYPE_WEIGHTS: list[float] = [0.35, 0.15, 0.05, 0.08, 0.18, 0.12, 0.07]

_SPECIES: list[str] = ["human", "human", "human", "mouse", "human"]

_STUDY_IDS: list[str] = [
    "OAS_SRR1234567", "OAS_ERR8901234", "OAS_SRR5678901",
    "OAS_ERR2345678", "OAS_SRR9012345", "OAS_ERR6789012",
    "OAS_SRR3456789", "OAS_ERR0123456",
]

_VALID_AA: str = "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _random_aa(length: int, rng: random.Random) -> str:
    """Generate a random amino acid string of given length using the provided RNG.

    Args:
        length: Number of amino acid characters to generate.
        rng: Random number generator for reproducibility.

    Returns:
        Random amino acid string.
    """
    return "".join(rng.choices(_VALID_AA, k=length))


def _build_full_vh(cdr3_aa: str, rng: random.Random) -> str:
    """Build a plausible full VH amino acid sequence incorporating the given CDR3.

    Uses random framework and CDR1/CDR2 regions of realistic lengths.

    Args:
        cdr3_aa: CDR3 amino acid sequence to embed.
        rng: Random number generator.

    Returns:
        Full VH amino acid string (FW1 + CDR1 + FW2 + CDR2 + FW3 + CDR3 + FW4).
    """
    fw1 = _random_aa(25, rng)
    cdr1 = _random_aa(rng.randint(5, 12), rng)
    fw2 = _random_aa(17, rng)
    cdr2 = _random_aa(rng.randint(6, 10), rng)
    fw3 = _random_aa(38, rng)
    fw4 = _random_aa(11, rng)
    return fw1 + cdr1 + fw2 + cdr2 + fw3 + cdr3_aa + fw4


def _make_content_hash(full_vh_aa: str, vh_gene: str | None) -> str:
    """Compute a 32-character hex SHA-256 hash for deduplication.

    Args:
        full_vh_aa: Full VH amino acid sequence.
        vh_gene: VH gene name (or empty string).

    Returns:
        First 32 hex characters of SHA-256(full_vh_aa|vh_gene).
    """
    raw = f"{full_vh_aa}|{vh_gene or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def generate_synthetic_sequences(n: int = 200) -> list[dict[str, Any]]:
    """Generate ``n`` synthetic antibody sequence records.

    Args:
        n: Number of records to generate. Defaults to 200.

    Returns:
        List of record dicts ready for insertion into
        ``ingested_antibody_sequences``.
    """
    rng = random.Random(_RANDOM_SEED)
    records: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    attempts = 0
    while len(records) < n and attempts < n * 3:
        attempts += 1

        vh_gene = rng.choice(_VH_GENES)
        dh_gene = rng.choice(_DH_GENES)
        jh_gene = rng.choice(_JH_GENES)
        isotype = rng.choices(_ISOTYPES, weights=_ISOTYPE_WEIGHTS)[0]
        species = rng.choice(_SPECIES)
        study_id = rng.choice(_STUDY_IDS)

        cdr3_len = rng.randint(8, 20)
        cdr3_aa = _random_aa(cdr3_len, rng)
        full_vh_aa = _build_full_vh(cdr3_aa, rng)

        content_hash = _make_content_hash(full_vh_aa, vh_gene)
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        records.append(
            {
                "seq_id": str(uuid.uuid4()),
                "source": "OAS",
                "vh_gene": vh_gene,
                "dh_gene": dh_gene,
                "jh_gene": jh_gene,
                "cdr3_aa": cdr3_aa,
                "cdr3_length": len(cdr3_aa),
                "full_vh_aa": full_vh_aa,
                "isotype": isotype,
                "species": species,
                "study_id": study_id,
                "content_hash": content_hash,
                "ingested_at": datetime.utcnow(),
            }
        )

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def seed(db_url: str, n: int = 200) -> None:
    """Generate and insert synthetic antibody sequences into the database.

    Args:
        db_url: SQLAlchemy database URL.
        n: Number of records to generate and insert.
    """
    print(f"[seed] Connecting to database ...")
    engine = get_engine(db_url)

    print("[seed] Ensuring ingestion tables exist ...")
    Base.metadata.create_all(engine)

    print(f"[seed] Generating {n} synthetic antibody sequence records ...")
    records = generate_synthetic_sequences(n=n)

    session_factory = get_session_factory(engine)

    inserted = 0
    skipped = 0

    with session_scope(session_factory) as session:
        from pipeline.loaders.db import load_dataframe
        import pandas as pd

        df = pd.DataFrame(records)
        n_ins, n_skip = load_dataframe(
            session, IngestedAntibodySequence, df, batch_size=100
        )
        inserted = n_ins
        skipped = n_skip

    print(f"[seed] Done. Inserted: {inserted}, Skipped (duplicates): {skipped}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the seed script."""
    parser = argparse.ArgumentParser(
        description="Generate 200 synthetic antibody sequences into ingested_antibody_sequences."
    )
    parser.add_argument(
        "--db-url",
        default=os.environ.get("DATABASE_URL", ""),
        help="SQLAlchemy database URL (default: DATABASE_URL env var)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of records to generate (default: 200)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.db_url:
        print(
            "ERROR: No database URL. Set DATABASE_URL or pass --db-url.",
            file=sys.stderr,
        )
        sys.exit(1)
    seed(db_url=args.db_url, n=args.n)
