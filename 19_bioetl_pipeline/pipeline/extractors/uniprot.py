"""Extractor for B cell surface marker proteins from UniProt.

Queries the UniProt REST API for key B cell markers:
CD19, CD20 (MS4A1), CD38, CD79A, CD22, MS4A1.

Real API endpoint::

    GET https://rest.uniprot.org/uniprotkb/search
        ?query=gene:CD19+OR+gene:CD20+OR+gene:CD38+OR+gene:CD79A+OR+gene:CD22+OR+gene:MS4A1
        &format=json
        &fields=accession,gene_names,protein_name,organism_name,cc_function
        &size=50

Falls back to 10 curated synthetic B cell marker records if the API is
unavailable.
"""

from __future__ import annotations

from typing import Any

from pipeline.extractors.base import BaseExtractor
from pipeline.logger import get_logger

_log = get_logger(__name__)

# Curated synthetic fallback records based on known UniProt annotations
_SYNTHETIC_MARKERS: list[dict[str, Any]] = [
    {
        "uniprot_id": "P15391",
        "gene_name": "CD19",
        "protein_name": "B-lymphocyte antigen CD19",
        "organism": "Homo sapiens",
        "function_text": (
            "Acts as a co-receptor for the B-cell antigen receptor complex (BCR) on mature "
            "B lymphocytes. Required for B lymphocyte development, differentiation and antibody "
            "production. Key therapeutic target for B cell malignancies."
        ),
    },
    {
        "uniprot_id": "P11836",
        "gene_name": "MS4A1",
        "protein_name": "B-lymphocyte antigen CD20",
        "organism": "Homo sapiens",
        "function_text": (
            "Involved in the regulation of B-cell activation and proliferation. May function as "
            "a calcium channel or regulator. Targeted by rituximab and other anti-CD20 therapies "
            "for treatment of B cell lymphomas and autoimmune diseases."
        ),
    },
    {
        "uniprot_id": "P28907",
        "gene_name": "CD38",
        "protein_name": "ADP-ribosyl cyclase/cyclic ADP-ribose hydrolase 1",
        "organism": "Homo sapiens",
        "function_text": (
            "Multifunctional ectoenzyme that synthesizes and hydrolyzes cyclic ADP-ribose "
            "(cADPR). Involved in cell adhesion, signal transduction and calcium signaling. "
            "Highly expressed on plasma cells and plasmablasts; target of daratumumab."
        ),
    },
    {
        "uniprot_id": "P11912",
        "gene_name": "CD79A",
        "protein_name": "B-cell antigen receptor complex-associated protein alpha chain",
        "organism": "Homo sapiens",
        "function_text": (
            "Required for surface expression of the BCR complex. Associates non-covalently "
            "with membrane immunoglobulin to form the BCR. Contains ITAM motifs that transduce "
            "B cell activation signals upon antigen binding."
        ),
    },
    {
        "uniprot_id": "P20273",
        "gene_name": "CD22",
        "protein_name": "B-cell receptor CD22",
        "organism": "Homo sapiens",
        "function_text": (
            "Mediates B cell adhesion and acts as an inhibitory co-receptor of the BCR. "
            "Binds sialic acid residues via lectin domain. Regulates B cell activation "
            "threshold and survival. Expressed on mature naïve and memory B cells."
        ),
    },
    {
        "uniprot_id": "P14770",
        "gene_name": "CD79B",
        "protein_name": "B-cell antigen receptor complex-associated protein beta chain",
        "organism": "Homo sapiens",
        "function_text": (
            "Required for BCR complex surface expression together with CD79A. Contains ITAM "
            "motif required for downstream signaling. Mutations in CD79B are associated with "
            "diffuse large B cell lymphoma (DLBCL)."
        ),
    },
    {
        "uniprot_id": "Q06830",
        "gene_name": "PTPRC",
        "protein_name": "Receptor-type tyrosine-protein phosphatase C",
        "organism": "Homo sapiens",
        "function_text": (
            "Also known as CD45. Regulates B and T cell activation by controlling "
            "Src-family kinase activity. Isoforms distinguish B cell subsets: naive B "
            "cells express CD45RA; class-switched memory B cells express CD45RO."
        ),
    },
    {
        "uniprot_id": "P21854",
        "gene_name": "CD27",
        "protein_name": "CD27 antigen",
        "organism": "Homo sapiens",
        "function_text": (
            "TNFRSF member. CD27 expression marks memory B cells; its ligand CD70 promotes "
            "B cell differentiation and antibody production. Important marker distinguishing "
            "naive (CD27-) from memory (CD27+) B cells."
        ),
    },
    {
        "uniprot_id": "P16581",
        "gene_name": "SELE",
        "protein_name": "E-selectin",
        "organism": "Homo sapiens",
        "function_text": (
            "Cell adhesion molecule expressed on endothelium. Mediates B cell homing to "
            "inflamed tissues. Upregulated by cytokines during inflammation and relevant to "
            "B cell trafficking in autoimmune diseases."
        ),
    },
    {
        "uniprot_id": "O14931",
        "gene_name": "NCR3",
        "protein_name": "Natural cytotoxicity triggering receptor 3",
        "organism": "Homo sapiens",
        "function_text": (
            "NKp30 receptor expressed on NK cells and some activated B cells. Involved in "
            "innate immune recognition. Used as a lineage marker in lymphocyte immunophenotyping "
            "panels alongside classical B cell markers."
        ),
    },
]


class UniProtExtractor(BaseExtractor):
    """Extract B cell surface marker proteins from the UniProt REST API.

    Queries for key B cell lineage markers by gene name and parses protein
    name, organism, and functional annotation.

    Falls back to ``_SYNTHETIC_MARKERS`` if the API is unavailable.
    """

    source_name: str = "UniProt"

    def __init__(self, base_url: str, timeout: int = 30) -> None:
        """Initialise the UniProt extractor.

        Args:
            base_url: Base URL for the UniProt REST API
                (e.g. ``https://rest.uniprot.org/uniprotkb``).
            timeout: HTTP request timeout in seconds.
        """
        super().__init__(base_url=base_url, timeout=timeout)

    def _parse_protein_name(self, protein_name_obj: Any) -> str:
        """Extract a flat protein name string from the UniProt JSON protein_name object.

        UniProt nests protein names as::

            {"recommendedName": {"fullName": {"value": "..."}}}

        Args:
            protein_name_obj: The ``proteinDescription`` JSON sub-object.

        Returns:
            A plain protein name string, or ``"Unknown"`` if not parseable.
        """
        if not isinstance(protein_name_obj, dict):
            return "Unknown"
        recommended = protein_name_obj.get("recommendedName", {})
        if isinstance(recommended, dict):
            full_name = recommended.get("fullName", {})
            if isinstance(full_name, dict):
                return full_name.get("value", "Unknown")
        # Fallback: try submittedName
        submitted = protein_name_obj.get("submittedNames", [])
        if submitted and isinstance(submitted, list):
            first = submitted[0]
            if isinstance(first, dict):
                full_name = first.get("fullName", {})
                if isinstance(full_name, dict):
                    return full_name.get("value", "Unknown")
        return "Unknown"

    def _parse_gene_name(self, genes_list: Any) -> str:
        """Extract the primary gene name from the UniProt genes list.

        Args:
            genes_list: The ``genes`` JSON array from UniProt.

        Returns:
            Primary gene name string, or ``"Unknown"`` if not parseable.
        """
        if not isinstance(genes_list, list) or not genes_list:
            return "Unknown"
        first_gene = genes_list[0]
        if isinstance(first_gene, dict):
            gene_names = first_gene.get("geneName", {})
            if isinstance(gene_names, dict):
                return gene_names.get("value", "Unknown")
        return "Unknown"

    def _parse_function(self, comments_list: Any) -> str | None:
        """Extract functional annotation text from UniProt comments.

        Args:
            comments_list: The ``comments`` JSON array from UniProt.

        Returns:
            Function text string, or ``None`` if not present.
        """
        if not isinstance(comments_list, list):
            return None
        for comment in comments_list:
            if isinstance(comment, dict) and comment.get("commentType") == "FUNCTION":
                texts = comment.get("texts", [])
                if texts and isinstance(texts, list):
                    first_text = texts[0]
                    if isinstance(first_text, dict):
                        return first_text.get("value")
        return None

    def extract(self) -> list[dict[str, Any]]:
        """Extract B cell surface marker proteins from UniProt.

        Returns:
            A list of raw record dicts with keys: ``uniprot_id``, ``gene_name``,
            ``protein_name``, ``organism``, ``function_text``.
        """
        url = f"{self.base_url}/search"
        query = (
            "gene:CD19 OR gene:MS4A1 OR gene:CD38 OR gene:CD79A "
            "OR gene:CD22 OR gene:CD79B OR gene:CD27"
        )
        params: dict[str, Any] = {
            "query": query,
            "format": "json",
            "fields": "accession,gene_names,protein_name,organism_name,cc_function",
            "size": 50,
        }

        _log.info("uniprot_extraction_started", url=url, query=query)
        raw = self._get(url, params=params)

        records: list[dict[str, Any]] = []

        if isinstance(raw, dict):
            results = raw.get("results", [])
            if isinstance(results, list) and len(results) > 0:
                for entry in results:
                    if not isinstance(entry, dict):
                        continue

                    uniprot_id = entry.get("primaryAccession")
                    gene_name = self._parse_gene_name(entry.get("genes", []))
                    protein_name = self._parse_protein_name(
                        entry.get("proteinDescription", {})
                    )
                    organism_obj = entry.get("organism", {})
                    organism = (
                        organism_obj.get("scientificName", "Unknown")
                        if isinstance(organism_obj, dict)
                        else "Unknown"
                    )
                    function_text = self._parse_function(entry.get("comments", []))

                    records.append(
                        {
                            "uniprot_id": uniprot_id,
                            "gene_name": gene_name,
                            "protein_name": protein_name,
                            "organism": organism,
                            "function_text": function_text,
                        }
                    )

                _log.info("uniprot_api_success", n_records=len(records))

        if not records:
            _log.warning(
                "uniprot_api_failed_or_empty", falling_back="synthetic"
            )
            records = list(_SYNTHETIC_MARKERS)
            _log.info(
                "uniprot_using_synthetic_fallback", n_synthetic=len(records)
            )

        return records
