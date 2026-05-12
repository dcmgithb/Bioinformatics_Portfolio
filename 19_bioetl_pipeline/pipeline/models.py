"""SQLAlchemy 2.0 ORM models for the bioetl-pipeline ingestion schema.

Three tables hold records ingested from external sources:
- ``ingested_antibody_sequences`` — BCR sequences from OAS
- ``ingested_therapeutics`` — antibody drugs from ChEMBL
- ``ingested_b_cell_markers`` — B cell surface markers from UniProt

All tables use a ``content_hash`` (SHA-256[:32]) for idempotent upserts.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base for all bioetl ingestion models."""

    pass


class IngestedAntibodySequence(Base):
    """A single antibody VH sequence ingested from the Observed Antibody Space (OAS).

    The ``content_hash`` (SHA-256 of ``full_vh_aa``, hex[:32]) enforces
    deduplication across ingestion runs via an ON CONFLICT DO NOTHING upsert.
    """

    __tablename__ = "ingested_antibody_sequences"
    __table_args__ = (
        UniqueConstraint("content_hash", name="uq_ias_content_hash"),
    )

    seq_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    source: Mapped[str] = mapped_column(String(32), nullable=False)  # "OAS"
    vh_gene: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    dh_gene: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    jh_gene: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    cdr3_aa: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    cdr3_length: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    full_vh_aa: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    isotype: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    species: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    study_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<IngestedAntibodySequence seq_id={self.seq_id!r} vh={self.vh_gene!r}>"


class IngestedTherapeutic(Base):
    """An antibody therapeutic ingested from ChEMBL.

    Covers approved and clinical-stage antibody drugs with mechanism,
    target name, max clinical phase, and sequence/SMILES where available.
    """

    __tablename__ = "ingested_therapeutics"
    __table_args__ = (
        UniqueConstraint("content_hash", name="uq_it_content_hash"),
    )

    drug_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    chembl_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    max_phase: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    mechanism: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    target_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    sequence_or_smiles: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<IngestedTherapeutic drug_id={self.drug_id!r} name={self.name!r}>"


class IngestedBCellMarker(Base):
    """A B cell surface marker protein ingested from UniProt.

    Captures key B cell markers (CD19, CD20/MS4A1, CD38, CD79A, CD22, etc.)
    with gene name, protein name, organism, and functional annotation.
    """

    __tablename__ = "ingested_b_cell_markers"
    __table_args__ = (
        UniqueConstraint("content_hash", name="uq_ibm_content_hash"),
    )

    marker_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    uniprot_id: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    gene_name: Mapped[str] = mapped_column(String(64), nullable=False)
    protein_name: Mapped[str] = mapped_column(String(256), nullable=False)
    organism: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    function_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<IngestedBCellMarker marker_id={self.marker_id!r} gene={self.gene_name!r}>"
