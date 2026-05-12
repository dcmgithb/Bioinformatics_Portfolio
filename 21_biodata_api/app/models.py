from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Optional

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class Donor(Base):
    """Represents a study donor/participant in the LIMS."""

    __tablename__ = "donors"

    donor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    donor_code: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    sex: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    disease_status: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cohort: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    enrolled_at: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    # Relationships
    samples: Mapped[list[BCellSample]] = relationship(
        "BCellSample", back_populates="donor", lazy="selectin"
    )


class BCellSample(Base):
    """Represents a B-cell sample collected from a donor."""

    __tablename__ = "b_cell_samples"

    sample_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    donor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("donors.donor_id"), nullable=False
    )
    collected_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=True
    )
    collection_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    cell_count_1e6: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    viability_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    storage_condition: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    donor: Mapped[Optional[Donor]] = relationship(
        "Donor", back_populates="samples", lazy="selectin"
    )
    flow_runs: Mapped[list[FlowCytometryRun]] = relationship(
        "FlowCytometryRun", back_populates="sample", lazy="selectin"
    )
    antibody_sequences: Mapped[list[AntibodySequence]] = relationship(
        "AntibodySequence", back_populates="sample", lazy="selectin"
    )


class FlowCytometryRun(Base):
    """Represents a flow cytometry run performed on a B-cell sample."""

    __tablename__ = "flow_cytometry_runs"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    sample_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("b_cell_samples.sample_id"), nullable=False
    )
    instrument_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("instruments.instrument_id"), nullable=True
    )
    operator_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=True
    )
    run_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    panel_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    b_cell_gate_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cd19_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cd20_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cd38_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cd138_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    naive_b_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    memory_b_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    plasmablast_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    qc_pass: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    sample: Mapped[Optional[BCellSample]] = relationship(
        "BCellSample", back_populates="flow_runs", lazy="selectin"
    )


class AntibodySequence(Base):
    """Represents an antibody sequence (VH/VL) derived from a B-cell sample."""

    __tablename__ = "antibody_sequences"

    seq_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    sample_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("b_cell_samples.sample_id"), nullable=False
    )
    chain_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    vh_gene: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    dh_gene: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    jh_gene: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    vl_gene: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    jl_gene: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cdr1_aa: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cdr2_aa: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cdr3_aa: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cdr3_length: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    full_vh_aa: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    full_vl_aa: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    isotype: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    clone_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    read_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, default=1)
    content_hash: Mapped[Optional[str]] = mapped_column(Text, unique=True, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    sample: Mapped[Optional[BCellSample]] = relationship(
        "BCellSample", back_populates="antibody_sequences", lazy="selectin"
    )
    expression_results: Mapped[list[ExpressionResult]] = relationship(
        "ExpressionResult", back_populates="sequence", lazy="selectin"
    )
    assay_results: Mapped[list[AssayResult]] = relationship(
        "AssayResult", back_populates="sequence", lazy="selectin"
    )


class ExpressionResult(Base):
    """Represents the result of expressing an antibody construct."""

    __tablename__ = "expression_results"

    result_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    seq_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("antibody_sequences.seq_id"), nullable=False
    )
    construct_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    expression_system: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    yield_mg_l: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    purity_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    aggregation_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    endotoxin_eu_ml: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    expression_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    expressed_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=True
    )
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    sequence: Mapped[Optional[AntibodySequence]] = relationship(
        "AntibodySequence", back_populates="expression_results", lazy="selectin"
    )


class AssayResult(Base):
    """Represents a functional assay result for an antibody sequence."""

    __tablename__ = "assay_results"

    assay_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    seq_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("antibody_sequences.seq_id"), nullable=False
    )
    assay_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    target_antigen: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    binding_kd_nm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ic50_nm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    neutralisation_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pass_fail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    assay_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    run_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=True
    )
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    sequence: Mapped[Optional[AntibodySequence]] = relationship(
        "AntibodySequence", back_populates="assay_results", lazy="selectin"
    )
