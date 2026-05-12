"""Extractor for antibody therapeutics from the ChEMBL database.

ChEMBL is a large-scale bioactivity database covering approved and
clinical-stage drugs. This extractor fetches antibody-type molecules
and their mechanism-of-action annotations.

Real API endpoint::

    GET https://www.ebi.ac.uk/chembl/api/data/molecule.json
        ?molecule_type=Antibody&limit=50

Falls back to 30 synthetic antibody drug records (modelled on real approved
antibodies) if the ChEMBL API is unavailable.
"""

from __future__ import annotations

import random
from typing import Any

from pipeline.extractors.base import BaseExtractor
from pipeline.logger import get_logger

_log = get_logger(__name__)

_SYNTHETIC_SEED: int = 43

# Reference dataset of approved antibody therapeutics (publicly known)
_SYNTHETIC_DRUGS: list[dict[str, Any]] = [
    {
        "chembl_id": "CHEMBL1201576",
        "name": "Rituximab",
        "max_phase": 4,
        "mechanism": "CD20 antagonist",
        "target_name": "CD20 / MS4A1",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL1201585",
        "name": "Trastuzumab",
        "max_phase": 4,
        "mechanism": "HER2 antagonist",
        "target_name": "ERBB2",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL3707320",
        "name": "Pembrolizumab",
        "max_phase": 4,
        "mechanism": "PD-1 inhibitor",
        "target_name": "PDCD1",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL1743081",
        "name": "Nivolumab",
        "max_phase": 4,
        "mechanism": "PD-1 inhibitor",
        "target_name": "PDCD1",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL2007641",
        "name": "Atezolizumab",
        "max_phase": 4,
        "mechanism": "PD-L1 inhibitor",
        "target_name": "CD274",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL3833335",
        "name": "Durvalumab",
        "max_phase": 4,
        "mechanism": "PD-L1 inhibitor",
        "target_name": "CD274",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL1201580",
        "name": "Infliximab",
        "max_phase": 4,
        "mechanism": "TNF-alpha inhibitor",
        "target_name": "TNF",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL1201579",
        "name": "Adalimumab",
        "max_phase": 4,
        "mechanism": "TNF-alpha inhibitor",
        "target_name": "TNF",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL1201577",
        "name": "Bevacizumab",
        "max_phase": 4,
        "mechanism": "VEGF-A inhibitor",
        "target_name": "VEGFA",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL3137343",
        "name": "Secukinumab",
        "max_phase": 4,
        "mechanism": "IL-17A inhibitor",
        "target_name": "IL17A",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL2146425",
        "name": "Dupilumab",
        "max_phase": 4,
        "mechanism": "IL-4 receptor alpha antagonist",
        "target_name": "IL4R",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL2107865",
        "name": "Ixekizumab",
        "max_phase": 4,
        "mechanism": "IL-17A inhibitor",
        "target_name": "IL17A",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL3544988",
        "name": "Sarilumab",
        "max_phase": 4,
        "mechanism": "IL-6 receptor antagonist",
        "target_name": "IL6R",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL1201583",
        "name": "Tocilizumab",
        "max_phase": 4,
        "mechanism": "IL-6 receptor antagonist",
        "target_name": "IL6R",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL1201584",
        "name": "Cetuximab",
        "max_phase": 4,
        "mechanism": "EGFR antagonist",
        "target_name": "EGFR",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL3833336",
        "name": "Daratumumab",
        "max_phase": 4,
        "mechanism": "CD38 antagonist",
        "target_name": "CD38",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL2107853",
        "name": "Elotuzumab",
        "max_phase": 4,
        "mechanism": "SLAMF7 antagonist",
        "target_name": "SLAMF7",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL3833337",
        "name": "Isatuximab",
        "max_phase": 4,
        "mechanism": "CD38 antagonist",
        "target_name": "CD38",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL3707323",
        "name": "Obinutuzumab",
        "max_phase": 4,
        "mechanism": "CD20 antagonist",
        "target_name": "CD20 / MS4A1",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL1201588",
        "name": "Ofatumumab",
        "max_phase": 4,
        "mechanism": "CD20 antagonist",
        "target_name": "CD20 / MS4A1",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL3833338",
        "name": "Zanubrutinib",
        "max_phase": 3,
        "mechanism": "BTK inhibitor",
        "target_name": "BTK",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL2107859",
        "name": "Blinatumomab",
        "max_phase": 4,
        "mechanism": "CD19/CD3 bispecific T cell engager",
        "target_name": "CD19 / CD3E",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL3707326",
        "name": "Mogamulizumab",
        "max_phase": 4,
        "mechanism": "CCR4 antagonist",
        "target_name": "CCR4",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL3544990",
        "name": "Guselkumab",
        "max_phase": 4,
        "mechanism": "IL-23 inhibitor",
        "target_name": "IL23A",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL3544991",
        "name": "Risankizumab",
        "max_phase": 4,
        "mechanism": "IL-23 inhibitor",
        "target_name": "IL23A",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL3833340",
        "name": "Tezepelumab",
        "max_phase": 4,
        "mechanism": "TSLP inhibitor",
        "target_name": "TSLP",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL3833341",
        "name": "Tralokinumab",
        "max_phase": 4,
        "mechanism": "IL-13 inhibitor",
        "target_name": "IL13",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL3833342",
        "name": "Crizanlizumab",
        "max_phase": 4,
        "mechanism": "P-selectin antagonist",
        "target_name": "SELP",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL3833343",
        "name": "Sutimlimab",
        "max_phase": 4,
        "mechanism": "Complement C1s inhibitor",
        "target_name": "C1S",
        "sequence_or_smiles": None,
    },
    {
        "chembl_id": "CHEMBL3833344",
        "name": "Iplilimumab",
        "max_phase": 4,
        "mechanism": "CTLA-4 antagonist",
        "target_name": "CTLA4",
        "sequence_or_smiles": None,
    },
]


class ChEMBLExtractor(BaseExtractor):
    """Extract antibody therapeutics from the ChEMBL REST API.

    Fetches molecule records with ``molecule_type=Antibody``, then
    optionally queries mechanism-of-action data per molecule ChEMBL ID.

    Falls back to ``_SYNTHETIC_DRUGS`` if the API is unavailable.
    """

    source_name: str = "ChEMBL"

    def __init__(self, base_url: str, timeout: int = 30) -> None:
        """Initialise the ChEMBL extractor.

        Args:
            base_url: Base URL for the ChEMBL API
                (e.g. ``https://www.ebi.ac.uk/chembl/api/data``).
            timeout: HTTP request timeout in seconds.
        """
        super().__init__(base_url=base_url, timeout=timeout)

    def _fetch_mechanism(self, chembl_id: str) -> str | None:
        """Fetch mechanism-of-action text for a single molecule.

        Args:
            chembl_id: ChEMBL molecule identifier (e.g. ``CHEMBL1201576``).

        Returns:
            Mechanism of action string, or ``None`` if not found.
        """
        url = f"{self.base_url}/mechanism.json"
        params: dict[str, Any] = {"molecule_chembl_id": chembl_id}
        raw = self._get(url, params=params)
        if not isinstance(raw, dict):
            return None
        mechanisms = raw.get("mechanisms", [])
        if mechanisms and isinstance(mechanisms, list):
            moa = mechanisms[0]
            return moa.get("mechanism_of_action") or moa.get("action_type")
        return None

    def extract(self) -> list[dict[str, Any]]:
        """Extract antibody therapeutics from ChEMBL.

        Returns:
            A list of raw record dicts with keys: ``chembl_id``, ``name``,
            ``max_phase``, ``mechanism``, ``target_name``, ``sequence_or_smiles``.
        """
        url = f"{self.base_url}/molecule.json"
        params: dict[str, Any] = {"molecule_type": "Antibody", "limit": 50}

        _log.info("chembl_extraction_started", url=url)
        raw = self._get(url, params=params)

        records: list[dict[str, Any]] = []

        if isinstance(raw, dict):
            molecules = raw.get("molecules", [])
            if isinstance(molecules, list) and len(molecules) > 0:
                for mol in molecules:
                    chembl_id = mol.get("molecule_chembl_id")
                    pref_name = mol.get("pref_name") or mol.get("name") or "Unknown"
                    max_phase = mol.get("max_phase")
                    mol_props = mol.get("molecule_properties") or {}
                    sequence_or_smiles = (
                        mol_props.get("full_molformula")
                        or mol_props.get("canonical_smiles")
                    )

                    mechanism: str | None = None
                    target_name: str | None = None
                    if chembl_id:
                        mechanism = self._fetch_mechanism(chembl_id)

                    records.append(
                        {
                            "chembl_id": chembl_id,
                            "name": pref_name,
                            "max_phase": max_phase,
                            "mechanism": mechanism,
                            "target_name": target_name,
                            "sequence_or_smiles": sequence_or_smiles,
                        }
                    )

                _log.info("chembl_api_success", n_records=len(records))

        if not records:
            _log.warning(
                "chembl_api_failed_or_empty", falling_back="synthetic"
            )
            records = list(_SYNTHETIC_DRUGS)
            _log.info("chembl_using_synthetic_fallback", n_synthetic=len(records))

        return records
