"""Extractor for the Observed Antibody Space (OAS) REST API.

OAS provides curated, paired and unpaired antibody sequences from published
studies. This extractor targets the unpaired heavy-chain sequence endpoint.

Real API endpoint::

    GET http://opig.stats.ox.ac.uk/webapps/oas/api/v1/sequence/?format=json&limit=100

If the API is unavailable (network error, timeout, unexpected response format),
the extractor falls back to generating 50 synthetic but biologically realistic
antibody sequence records so the pipeline can complete without external access.
"""

from __future__ import annotations

import random
from typing import Any

from pipeline.extractors.base import BaseExtractor
from pipeline.logger import get_logger

_log = get_logger(__name__)

# Seed for reproducible synthetic data
_SYNTHETIC_SEED: int = 42

# Biologically realistic IGHV gene pool (most commonly observed in humans)
_VH_GENES: list[str] = [
    "IGHV1-2", "IGHV1-3", "IGHV1-8", "IGHV1-18", "IGHV1-46", "IGHV1-69",
    "IGHV2-5", "IGHV2-70",
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
    "IgG", "IgM", "IgA", "IgG1", "IgG2", "IgG3", "IgG4", "IgA1", "IgA2",
]
_ISOTYPE_WEIGHTS: list[float] = [0.25, 0.15, 0.10, 0.15, 0.10, 0.05, 0.05, 0.08, 0.07]

_SPECIES: list[str] = ["human", "human", "human", "mouse", "human"]  # human-biased

_STUDY_IDS: list[str] = [
    "OAS_SRR1234567", "OAS_ERR8901234", "OAS_SRR5678901",
    "OAS_ERR2345678", "OAS_SRR9012345", "OAS_ERR6789012",
]

_VALID_AA: str = "ACDEFGHIKLMNPQRSTVWY"


def _random_aa(length: int, rng: random.Random) -> str:
    """Generate a random amino acid string of the given length."""
    return "".join(rng.choices(_VALID_AA, k=length))


def _generate_synthetic_records(n: int = 50) -> list[dict[str, Any]]:
    """Generate ``n`` synthetic OAS-like antibody sequence records.

    Used as a fallback when the real OAS API is unavailable.

    Args:
        n: Number of synthetic records to generate.

    Returns:
        A list of raw record dicts mimicking OAS API output.
    """
    rng = random.Random(_SYNTHETIC_SEED)
    records: list[dict[str, Any]] = []

    for i in range(n):
        vh_gene = rng.choice(_VH_GENES)
        cdr3_len = rng.randint(8, 20)
        cdr3_aa = _random_aa(cdr3_len, rng)
        # Build a plausible full VH amino acid sequence
        fw1 = _random_aa(25, rng)
        cdr1 = _random_aa(rng.randint(5, 12), rng)
        fw2 = _random_aa(17, rng)
        cdr2 = _random_aa(rng.randint(6, 10), rng)
        fw3 = _random_aa(38, rng)
        fw4 = _random_aa(11, rng)
        full_vh = fw1 + cdr1 + fw2 + cdr2 + fw3 + cdr3_aa + fw4

        records.append(
            {
                "sequence_id_heavy": f"synthetic_{i:04d}",
                "VH": vh_gene,
                "DH": rng.choice(_DH_GENES),
                "JH": rng.choice(_JH_GENES),
                "CDRH3_aa": cdr3_aa,
                "full_vh_aa": full_vh,
                "isotype": rng.choices(_ISOTYPES, weights=_ISOTYPE_WEIGHTS)[0],
                "species": rng.choice(_SPECIES),
                "study_id": rng.choice(_STUDY_IDS),
                "_synthetic": True,
            }
        )

    return records


class OASExtractor(BaseExtractor):
    """Extract antibody sequences from the Observed Antibody Space (OAS) API.

    Falls back to ``_generate_synthetic_records()`` if the API is unavailable.
    """

    source_name: str = "OAS"

    def __init__(self, base_url: str, timeout: int = 30) -> None:
        """Initialise the OAS extractor.

        Args:
            base_url: Base URL for the OAS API
                (e.g. ``http://opig.stats.ox.ac.uk/webapps/oas/api``).
            timeout: HTTP request timeout in seconds.
        """
        super().__init__(base_url=base_url, timeout=timeout)

    def extract(self) -> list[dict[str, Any]]:
        """Extract antibody sequences from the OAS API.

        Attempts to fetch from the live API; falls back to synthetic data
        if the API is unreachable or returns an unexpected payload.

        Returns:
            A list of raw record dicts, each with keys:
            ``VH``, ``DH``, ``JH``, ``CDRH3_aa``, ``full_vh_aa``,
            ``isotype``, ``species``, ``study_id``.
        """
        url = f"{self.base_url}/v1/sequence/"
        params: dict[str, Any] = {"format": "json", "limit": 100}

        _log.info("oas_extraction_started", url=url)
        raw = self._get(url, params=params)

        records: list[dict[str, Any]] = []

        if isinstance(raw, dict):
            results = raw.get("results", [])
            if isinstance(results, list) and len(results) > 0:
                records = results
                _log.info("oas_api_success", n_records=len(records))
            else:
                _log.warning(
                    "oas_api_empty_results",
                    response_keys=list(raw.keys()) if raw else [],
                )
        elif isinstance(raw, list) and len(raw) > 0:
            records = raw
            _log.info("oas_api_success_list", n_records=len(records))
        else:
            _log.warning("oas_api_failed_or_empty", falling_back="synthetic")

        if not records:
            _log.info("oas_using_synthetic_fallback", n_synthetic=50)
            records = _generate_synthetic_records(n=50)

        return records
