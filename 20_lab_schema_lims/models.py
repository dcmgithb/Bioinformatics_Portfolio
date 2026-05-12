"""SQLAlchemy 2.0 ORM models for the Antibody Discovery LIMS.

Tables:
    users, instruments, donors, b_cell_samples, flow_cytometry_runs,
    antibody_sequences, expression_results, assay_results, audit_log
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Shared declarative base for all LIMS models."""


# ---------------------------------------------------------------------------
# users
# ---------------------------------------------------------------------------

class User(Base):
    """Lab personnel who interact with the LIMS."""

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(256), unique=True, nullable=False)
    role: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # scientist | analyst | admin
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Reverse relationships
    samples_collected: Mapped[list[BCellSample]] = relationship(
        "BCellSample", back_populates="collector", foreign_keys="BCellSample.collected_by"
    )
    flow_runs_operated: Mapped[list[FlowCytometryRun]] = relationship(
        "FlowCytometryRun",
        back_populates="operator",
        foreign_keys="FlowCytometryRun.operator_id",
    )
    expression_results_expressed: Mapped[list[ExpressionResult]] = relationship(
        "ExpressionResult",
        back_populates="expresser",
        foreign_keys="ExpressionResult.expressed_by",
    )
    assay_results_run: Mapped[list[AssayResult]] = relationship(
        "AssayResult",
        back_populates="runner",
        foreign_keys="AssayResult.run_by",
    )
    audit_entries: Mapped[list[AuditLog]] = relationship(
        "AuditLog",
        back_populates="changer",
        foreign_keys="AuditLog.changed_by",
    )

    def __repr__(self) -> str:
        return f"<User {self.username!r} role={self.role!r}>"


# ---------------------------------------------------------------------------
# instruments
# ---------------------------------------------------------------------------

class Instrument(Base):
    """Lab instruments used in sample processing and assays."""

    __tablename__ = "instruments"

    instrument_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    instrument_type: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # flow_cytometer | plate_reader | biolayer_interferometry
    manufacturer: Mapped[str] = mapped_column(String(128), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    serial_number: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    calibration_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Reverse relationships
    flow_runs: Mapped[list[FlowCytometryRun]] = relationship(
        "FlowCytometryRun", back_populates="instrument"
    )

    def __repr__(self) -> str:
        return f"<Instrument {self.name!r} ({self.instrument_type})>"


# ---------------------------------------------------------------------------
# donors
# ---------------------------------------------------------------------------

class Donor(Base):
    """Study participants who provide B cell samples."""

    __tablename__ = "donors"

    donor_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    donor_code: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    sex: Mapped[str] = mapped_column(String(1), nullable=False)  # M | F | U
    disease_status: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # healthy | autoimmune | cancer
    cohort: Mapped[str] = mapped_column(String(128), nullable=False)
    enrolled_at: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    # Relationships
    samples: Mapped[list[BCellSample]] = relationship(
        "BCellSample", back_populates="donor"
    )

    def __repr__(self) -> str:
        return f"<Donor {self.donor_code!r} age={self.age} status={self.disease_status!r}>"


# ---------------------------------------------------------------------------
# b_cell_samples
# ---------------------------------------------------------------------------

class BCellSample(Base):
    """A B cell sample collected from a donor at a specific time."""

    __tablename__ = "b_cell_samples"
    __table_args__ = (
        Index("idx_bcell_samples_donor_id", "donor_id"),
        Index("idx_bcell_samples_collection_date", "collection_date"),
    )

    sample_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    donor_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("donors.donor_id"), nullable=False
    )
    collected_by: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.user_id"), nullable=True
    )
    collection_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    cell_count_1e6: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    viability_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    storage_condition: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )  # liquid_nitrogen | DMSO | -80C
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Relationships
    donor: Mapped[Donor] = relationship("Donor", back_populates="samples")
    collector: Mapped[Optional[User]] = relationship(
        "User",
        back_populates="samples_collected",
        foreign_keys=[collected_by],
    )
    flow_runs: Mapped[list[FlowCytometryRun]] = relationship(
        "FlowCytometryRun", back_populates="sample"
    )
    sequences: Mapped[list[AntibodySequence]] = relationship(
        "AntibodySequence", back_populates="sample"
    )

    def __repr__(self) -> str:
        return f"<BCellSample {self.sample_id!r} donor={self.donor_id!r}>"


# ---------------------------------------------------------------------------
# flow_cytometry_runs
# ---------------------------------------------------------------------------

class FlowCytometryRun(Base):
    """A single flow cytometry acquisition run on a B cell sample."""

    __tablename__ = "flow_cytometry_runs"
    __table_args__ = (
        Index("idx_flow_runs_sample_id", "sample_id"),
        Index("idx_flow_runs_run_date", "run_date"),
    )

    run_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    sample_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("b_cell_samples.sample_id"), nullable=False
    )
    instrument_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("instruments.instrument_id"), nullable=False
    )
    operator_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.user_id"), nullable=True
    )
    run_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    panel_name: Mapped[str] = mapped_column(String(128), nullable=False)
    b_cell_gate_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cd19_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cd20_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cd38_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cd138_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    naive_b_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    memory_b_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    plasmablast_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    qc_pass: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Relationships
    sample: Mapped[BCellSample] = relationship(
        "BCellSample", back_populates="flow_runs"
    )
    instrument: Mapped[Instrument] = relationship(
        "Instrument", back_populates="flow_runs"
    )
    operator: Mapped[Optional[User]] = relationship(
        "User",
        back_populates="flow_runs_operated",
        foreign_keys=[operator_id],
    )

    def __repr__(self) -> str:
        return (
            f"<FlowCytometryRun {self.run_id!r} sample={self.sample_id!r} "
            f"qc_pass={self.qc_pass}>"
        )


# ---------------------------------------------------------------------------
# antibody_sequences
# ---------------------------------------------------------------------------

class AntibodySequence(Base):
    """A single antibody chain sequence derived from B cell sequencing."""

    __tablename__ = "antibody_sequences"
    __table_args__ = (
        Index("idx_ab_seqs_sample_id", "sample_id"),
        Index("idx_ab_seqs_clone_id", "clone_id"),
        Index("idx_ab_seqs_vh_gene", "vh_gene"),
        Index("idx_ab_seqs_cdr3_length", "cdr3_length"),
    )

    seq_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    sample_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("b_cell_samples.sample_id"), nullable=False
    )
    chain_type: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # heavy | light
    vh_gene: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    dh_gene: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    jh_gene: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    vl_gene: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    jl_gene: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    cdr1_aa: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    cdr2_aa: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    cdr3_aa: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    cdr3_length: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    full_vh_aa: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    full_vl_aa: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    isotype: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    clone_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    read_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Relationships
    sample: Mapped[BCellSample] = relationship(
        "BCellSample", back_populates="sequences"
    )
    expression_results: Mapped[list[ExpressionResult]] = relationship(
        "ExpressionResult", back_populates="sequence"
    )
    assay_results: Mapped[list[AssayResult]] = relationship(
        "AssayResult", back_populates="sequence"
    )

    def __repr__(self) -> str:
        return (
            f"<AntibodySequence {self.seq_id!r} vh={self.vh_gene!r} "
            f"cdr3={self.cdr3_aa!r}>"
        )


# ---------------------------------------------------------------------------
# expression_results
# ---------------------------------------------------------------------------

class ExpressionResult(Base):
    """Recombinant expression result for an antibody construct."""

    __tablename__ = "expression_results"
    __table_args__ = (
        Index("idx_expr_results_seq_id", "seq_id"),
        Index("idx_expr_results_construct_name", "construct_name"),
    )

    result_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    seq_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("antibody_sequences.seq_id"), nullable=False
    )
    construct_name: Mapped[str] = mapped_column(String(128), nullable=False)
    expression_system: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # CHO | HEK293 | Expi293
    yield_mg_l: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    purity_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    aggregation_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    endotoxin_eu_ml: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    expression_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    expressed_by: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.user_id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Relationships
    sequence: Mapped[AntibodySequence] = relationship(
        "AntibodySequence", back_populates="expression_results"
    )
    expresser: Mapped[Optional[User]] = relationship(
        "User",
        back_populates="expression_results_expressed",
        foreign_keys=[expressed_by],
    )

    def __repr__(self) -> str:
        return (
            f"<ExpressionResult {self.result_id!r} construct={self.construct_name!r} "
            f"yield={self.yield_mg_l}>"
        )


# ---------------------------------------------------------------------------
# assay_results
# ---------------------------------------------------------------------------

class AssayResult(Base):
    """Binding, neutralisation, or other functional assay result for a sequence."""

    __tablename__ = "assay_results"
    __table_args__ = (
        Index("idx_assay_results_seq_id", "seq_id"),
        Index("idx_assay_results_assay_type", "assay_type"),
        Index("idx_assay_results_pass_fail", "pass_fail"),
    )

    assay_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    seq_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("antibody_sequences.seq_id"), nullable=False
    )
    assay_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # SPR | ELISA | neutralisation | flow_binding
    target_antigen: Mapped[str] = mapped_column(String(128), nullable=False)
    binding_kd_nm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ic50_nm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    neutralisation_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pass_fail: Mapped[Optional[str]] = mapped_column(
        String(16), nullable=True
    )  # pass | fail | inconclusive
    assay_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    run_by: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.user_id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Relationships
    sequence: Mapped[AntibodySequence] = relationship(
        "AntibodySequence", back_populates="assay_results"
    )
    runner: Mapped[Optional[User]] = relationship(
        "User",
        back_populates="assay_results_run",
        foreign_keys=[run_by],
    )

    def __repr__(self) -> str:
        return (
            f"<AssayResult {self.assay_id!r} type={self.assay_type!r} "
            f"kd={self.binding_kd_nm} pass_fail={self.pass_fail!r}>"
        )


# ---------------------------------------------------------------------------
# audit_log
# ---------------------------------------------------------------------------

class AuditLog(Base):
    """Immutable audit trail for all INSERT/UPDATE/DELETE events."""

    __tablename__ = "audit_log"
    __table_args__ = (
        Index("idx_audit_log_table_record", "table_name", "record_id"),
        Index("idx_audit_log_changed_at", "changed_at"),
    )

    log_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    table_name: Mapped[str] = mapped_column(String(64), nullable=False)
    record_id: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # INSERT | UPDATE | DELETE
    changed_by: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.user_id"), nullable=True
    )
    changed_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    old_values: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    new_values: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)

    # Relationships
    changer: Mapped[Optional[User]] = relationship(
        "User",
        back_populates="audit_entries",
        foreign_keys=[changed_by],
    )

    def __repr__(self) -> str:
        return (
            f"<AuditLog {self.log_id!r} {self.action} on {self.table_name!r} "
            f"record={self.record_id!r}>"
        )
