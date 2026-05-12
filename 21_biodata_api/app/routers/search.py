from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import verify_api_key
from ..database import get_db
from ..models import AntibodySequence
from ..schemas import SequenceSearchResult

router = APIRouter(tags=["search"])


@router.get("/sequences", response_model=list[SequenceSearchResult])
async def search_sequences(
    vh_gene_family: Optional[str] = Query(
        None,
        description="VH gene family prefix, e.g. 'IGHV3'. Uses case-insensitive prefix match.",
    ),
    cdr3_min_length: Optional[int] = Query(None, ge=1, description="Minimum CDR3 length (inclusive)"),
    cdr3_max_length: Optional[int] = Query(None, le=50, description="Maximum CDR3 length (inclusive)"),
    isotype: Optional[str] = Query(None, description="Exact isotype match, e.g. 'IgG1'"),
    clone_id: Optional[str] = Query(None, description="Exact clone_id match"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of results to return"),
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
) -> list[SequenceSearchResult]:
    """Search antibody sequences by one or more filter criteria.

    All filters are combined with AND logic. Filters not provided are ignored.

    - ``vh_gene_family``: case-insensitive prefix match on the ``vh_gene`` column
      (e.g. ``IGHV3`` matches ``IGHV3-23*01``).
    - ``cdr3_min_length`` / ``cdr3_max_length``: inclusive range filter on the
      ``cdr3_length`` column.
    - ``isotype``: exact (case-sensitive) match on the ``isotype`` column.
    - ``clone_id``: exact match on the ``clone_id`` column.
    - ``limit``: maximum rows returned (default 100, max 500).

    Args:
        vh_gene_family: Optional VH gene family prefix for ILIKE filtering.
        cdr3_min_length: Optional minimum CDR3 amino-acid length.
        cdr3_max_length: Optional maximum CDR3 amino-acid length.
        isotype: Optional exact isotype string.
        clone_id: Optional exact clone identifier string.
        limit: Maximum number of results (capped at 500).
        db: Injected async database session.
        _: Validated API key (unused directly).

    Returns:
        A list of up to ``limit`` ``SequenceSearchResult`` objects matching all
        supplied filters.
    """
    stmt = select(
        AntibodySequence.seq_id,
        AntibodySequence.vh_gene,
        AntibodySequence.cdr3_aa,
        AntibodySequence.cdr3_length,
        AntibodySequence.isotype,
        AntibodySequence.clone_id,
        AntibodySequence.read_count,
    )

    if vh_gene_family is not None:
        stmt = stmt.where(AntibodySequence.vh_gene.ilike(f"{vh_gene_family}%"))

    if cdr3_min_length is not None:
        stmt = stmt.where(AntibodySequence.cdr3_length >= cdr3_min_length)

    if cdr3_max_length is not None:
        stmt = stmt.where(AntibodySequence.cdr3_length <= cdr3_max_length)

    if isotype is not None:
        stmt = stmt.where(AntibodySequence.isotype == isotype)

    if clone_id is not None:
        stmt = stmt.where(AntibodySequence.clone_id == clone_id)

    stmt = stmt.limit(limit)

    result = await db.execute(stmt)
    rows = result.all()

    return [
        SequenceSearchResult(
            seq_id=str(row.seq_id),
            vh_gene=row.vh_gene,
            cdr3_aa=row.cdr3_aa,
            cdr3_length=row.cdr3_length,
            isotype=row.isotype,
            clone_id=row.clone_id,
            read_count=row.read_count,
        )
        for row in rows
    ]
