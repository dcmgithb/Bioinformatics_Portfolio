from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Flow cytometry nested schema
# ---------------------------------------------------------------------------


class FlowRunRead(BaseModel):
    """Schema for reading a flow cytometry run record."""

    model_config = ConfigDict(from_attributes=True)

    run_id: str
    sample_id: str
    run_date: Optional[date]
    panel_name: Optional[str]
    b_cell_gate_pct: Optional[float]
    cd19_pct: Optional[float]
    cd20_pct: Optional[float]
    cd38_pct: Optional[float]
    cd138_pct: Optional[float]
    naive_b_pct: Optional[float]
    memory_b_pct: Optional[float]
    plasmablast_pct: Optional[float]
    qc_pass: Optional[bool]

    @classmethod
    def model_validate(cls, obj: object, **kwargs: object) -> FlowRunRead:  # type: ignore[override]
        """Override to convert UUID fields to strings."""
        if hasattr(obj, "__dict__") or hasattr(obj, "__table__"):
            data = {c: getattr(obj, c) for c in cls.model_fields}
            data["run_id"] = str(data["run_id"]) if data.get("run_id") else data.get("run_id")
            data["sample_id"] = str(data["sample_id"]) if data.get("sample_id") else data.get("sample_id")
            return cls(**data)
        return super().model_validate(obj, **kwargs)


# ---------------------------------------------------------------------------
# Sample schemas
# ---------------------------------------------------------------------------


class SampleCreate(BaseModel):
    """Schema for creating a new B-cell sample."""

    donor_id: str
    collection_date: Optional[date] = None
    cell_count_1e6: Optional[float] = None
    viability_pct: Optional[float] = None
    storage_condition: Optional[str] = None
    notes: Optional[str] = None


class SampleUpdate(BaseModel):
    """Schema for partial update of a B-cell sample. All fields are optional."""

    donor_id: Optional[str] = None
    collection_date: Optional[date] = None
    cell_count_1e6: Optional[float] = None
    viability_pct: Optional[float] = None
    storage_condition: Optional[str] = None
    notes: Optional[str] = None


class SampleRead(BaseModel):
    """Schema for reading a B-cell sample with nested flow cytometry runs."""

    model_config = ConfigDict(from_attributes=True)

    sample_id: str
    donor_id: str
    collection_date: Optional[date]
    cell_count_1e6: Optional[float]
    viability_pct: Optional[float]
    storage_condition: Optional[str]
    notes: Optional[str]
    created_at: Optional[datetime]
    flow_runs: list[FlowRunRead] = []


# ---------------------------------------------------------------------------
# Expression / assay nested schemas
# ---------------------------------------------------------------------------


class ExpressionRead(BaseModel):
    """Schema for reading an expression result."""

    model_config = ConfigDict(from_attributes=True)

    result_id: str
    construct_name: Optional[str]
    expression_system: Optional[str]
    yield_mg_l: Optional[float]
    purity_pct: Optional[float]
    aggregation_pct: Optional[float]
    expression_date: Optional[date]


class AssayRead(BaseModel):
    """Schema for reading an assay result."""

    model_config = ConfigDict(from_attributes=True)

    assay_id: str
    assay_type: Optional[str]
    target_antigen: Optional[str]
    binding_kd_nm: Optional[float]
    ic50_nm: Optional[float]
    neutralisation_pct: Optional[float]
    pass_fail: Optional[str]
    assay_date: Optional[date]


# ---------------------------------------------------------------------------
# Sequence schemas
# ---------------------------------------------------------------------------


class SequenceCreate(BaseModel):
    """Schema for creating a new antibody sequence."""

    sample_id: str
    chain_type: Optional[str] = None
    vh_gene: Optional[str] = None
    dh_gene: Optional[str] = None
    jh_gene: Optional[str] = None
    vl_gene: Optional[str] = None
    jl_gene: Optional[str] = None
    cdr1_aa: Optional[str] = None
    cdr2_aa: Optional[str] = None
    cdr3_aa: Optional[str] = None
    full_vh_aa: Optional[str] = None
    full_vl_aa: Optional[str] = None
    isotype: Optional[str] = None
    clone_id: Optional[str] = None
    read_count: int = 1


class SequenceUpdate(BaseModel):
    """Schema for partial update of an antibody sequence. All fields are optional."""

    chain_type: Optional[str] = None
    vh_gene: Optional[str] = None
    dh_gene: Optional[str] = None
    jh_gene: Optional[str] = None
    vl_gene: Optional[str] = None
    jl_gene: Optional[str] = None
    cdr1_aa: Optional[str] = None
    cdr2_aa: Optional[str] = None
    cdr3_aa: Optional[str] = None
    full_vh_aa: Optional[str] = None
    full_vl_aa: Optional[str] = None
    isotype: Optional[str] = None
    clone_id: Optional[str] = None
    read_count: Optional[int] = None


class SequenceRead(BaseModel):
    """Schema for reading an antibody sequence with nested expression and assay results."""

    model_config = ConfigDict(from_attributes=True)

    seq_id: str
    sample_id: str
    chain_type: Optional[str]
    vh_gene: Optional[str]
    dh_gene: Optional[str]
    jh_gene: Optional[str]
    vl_gene: Optional[str]
    jl_gene: Optional[str]
    cdr1_aa: Optional[str]
    cdr2_aa: Optional[str]
    cdr3_aa: Optional[str]
    cdr3_length: Optional[int]
    full_vh_aa: Optional[str]
    full_vl_aa: Optional[str]
    isotype: Optional[str]
    clone_id: Optional[str]
    read_count: Optional[int]
    content_hash: Optional[str]
    created_at: Optional[datetime]
    expression_results: list[ExpressionRead] = []
    assay_results: list[AssayRead] = []


# ---------------------------------------------------------------------------
# Report schemas
# ---------------------------------------------------------------------------


class CloneDiversityStats(BaseModel):
    """Statistics describing clonal diversity in the antibody sequence repertoire."""

    total_sequences: int
    total_clones: int
    shannon_diversity_index: float
    d50: int  # number of clones covering 50% of reads
    top_clone_frequency_pct: float


class ConstructStat(BaseModel):
    """Aggregated statistics for a single construct/expression-system combination."""

    construct_name: str
    expression_system: str
    mean_yield_mg_l: float
    n_runs: int


class AssayStat(BaseModel):
    """Pass-rate and KD statistics for a single assay type."""

    assay_type: str
    total: int
    passed: int
    pass_rate_pct: float
    median_kd_nm: Optional[float]


class ReportResponse(BaseModel):
    """Full report response containing diversity, expression, and assay summaries."""

    generated_at: datetime
    clone_diversity: CloneDiversityStats
    top_constructs: list[ConstructStat]
    assay_summary: list[AssayStat]


# ---------------------------------------------------------------------------
# Search schema
# ---------------------------------------------------------------------------


class SequenceSearchResult(BaseModel):
    """Lightweight schema returned by the sequence search endpoint."""

    model_config = ConfigDict(from_attributes=True)

    seq_id: str
    vh_gene: Optional[str]
    cdr3_aa: Optional[str]
    cdr3_length: Optional[int]
    isotype: Optional[str]
    clone_id: Optional[str]
    read_count: Optional[int]


# ---------------------------------------------------------------------------
# Pagination schemas
# ---------------------------------------------------------------------------


class PaginatedSamples(BaseModel):
    """Paginated list of B-cell sample records."""

    total: int
    page: int
    size: int
    items: list[SampleRead]


class PaginatedSequences(BaseModel):
    """Paginated list of antibody sequence records."""

    total: int
    page: int
    size: int
    items: list[SequenceRead]
