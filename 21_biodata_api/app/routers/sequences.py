from __future__ import annotations

import hashlib
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import verify_api_key
from ..database import get_db
from ..models import AntibodySequence
from ..schemas import (
    AssayRead,
    ExpressionRead,
    PaginatedSequences,
    SequenceCreate,
    SequenceRead,
    SequenceUpdate,
)

router = APIRouter(prefix="/sequences", tags=["sequences"])


def _compute_content_hash(body: SequenceCreate) -> str:
    """Compute a deterministic SHA-256 hash over the immutable sequence fields.

    The hash is derived from the concatenation of ``sample_id``, ``chain_type``,
    ``full_vh_aa``, ``full_vl_aa``, and ``cdr3_aa`` so that identical sequences
    from the same sample always produce the same hash.

    Args:
        body: The ``SequenceCreate`` payload.

    Returns:
        A 64-character hexadecimal SHA-256 digest string.
    """
    raw = "|".join(
        [
            str(body.sample_id),
            str(body.chain_type or ""),
            str(body.full_vh_aa or ""),
            str(body.full_vl_aa or ""),
            str(body.cdr3_aa or ""),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _orm_to_sequence_read(obj: AntibodySequence) -> SequenceRead:
    """Convert an ``AntibodySequence`` ORM instance to a ``SequenceRead`` schema.

    Handles UUID-to-string conversion and builds nested ``ExpressionRead`` and
    ``AssayRead`` lists from eagerly-loaded relationships.

    Args:
        obj: A fully-loaded ``AntibodySequence`` ORM object.

    Returns:
        A populated ``SequenceRead`` instance.
    """
    expression_results = [
        ExpressionRead(
            result_id=str(er.result_id),
            construct_name=er.construct_name,
            expression_system=er.expression_system,
            yield_mg_l=er.yield_mg_l,
            purity_pct=er.purity_pct,
            aggregation_pct=er.aggregation_pct,
            expression_date=er.expression_date,
        )
        for er in obj.expression_results
    ]

    assay_results = [
        AssayRead(
            assay_id=str(ar.assay_id),
            assay_type=ar.assay_type,
            target_antigen=ar.target_antigen,
            binding_kd_nm=ar.binding_kd_nm,
            ic50_nm=ar.ic50_nm,
            neutralisation_pct=ar.neutralisation_pct,
            pass_fail=ar.pass_fail,
            assay_date=ar.assay_date,
        )
        for ar in obj.assay_results
    ]

    return SequenceRead(
        seq_id=str(obj.seq_id),
        sample_id=str(obj.sample_id),
        chain_type=obj.chain_type,
        vh_gene=obj.vh_gene,
        dh_gene=obj.dh_gene,
        jh_gene=obj.jh_gene,
        vl_gene=obj.vl_gene,
        jl_gene=obj.jl_gene,
        cdr1_aa=obj.cdr1_aa,
        cdr2_aa=obj.cdr2_aa,
        cdr3_aa=obj.cdr3_aa,
        cdr3_length=obj.cdr3_length,
        full_vh_aa=obj.full_vh_aa,
        full_vl_aa=obj.full_vl_aa,
        isotype=obj.isotype,
        clone_id=obj.clone_id,
        read_count=obj.read_count,
        content_hash=obj.content_hash,
        created_at=obj.created_at,
        expression_results=expression_results,
        assay_results=assay_results,
    )


@router.get("", response_model=PaginatedSequences)
async def list_sequences(
    page: int = Query(1, ge=1, description="1-based page number"),
    size: int = Query(20, ge=1, le=200, description="Items per page"),
    sample_id: Optional[str] = Query(None, description="Filter by sample UUID"),
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
) -> PaginatedSequences:
    """List antibody sequences with optional sample_id filter.

    Returns a paginated response with the total count, current page, page size,
    and sequence records (each including nested expression and assay results).

    Args:
        page: 1-based page number.
        size: Items per page (max 200).
        sample_id: Optional UUID string to filter sequences belonging to a specific sample.
        db: Injected async database session.
        _: Validated API key (unused directly).

    Returns:
        ``PaginatedSequences`` containing ``total``, ``page``, ``size``, and ``items``.

    Raises:
        HTTPException 400: If ``sample_id`` is not a valid UUID.
    """
    base_stmt = select(AntibodySequence)

    if sample_id is not None:
        try:
            sample_uuid = uuid.UUID(sample_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid sample_id UUID: {sample_id}",
            ) from exc
        base_stmt = base_stmt.where(AntibodySequence.sample_id == sample_uuid)

    count_stmt = select(func.count()).select_from(base_stmt.subquery())
    total_result = await db.execute(count_stmt)
    total: int = total_result.scalar_one()

    offset = (page - 1) * size
    page_stmt = base_stmt.offset(offset).limit(size)
    result = await db.execute(page_stmt)
    sequences = result.scalars().all()

    items = [_orm_to_sequence_read(s) for s in sequences]
    return PaginatedSequences(total=total, page=page, size=size, items=items)


@router.post("", response_model=SequenceRead, status_code=status.HTTP_201_CREATED)
async def create_sequence(
    body: SequenceCreate,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
) -> SequenceRead:
    """Create a new antibody sequence record.

    Automatically computes ``cdr3_length`` from ``cdr3_aa`` (if provided) and
    generates a deterministic ``content_hash`` to support deduplication.  If a
    sequence with the same hash already exists the endpoint returns HTTP 409.

    Args:
        body: Request body containing sequence fields.
        db: Injected async database session.
        _: Validated API key (unused directly).

    Returns:
        The newly created ``SequenceRead`` with a generated ``seq_id``.

    Raises:
        HTTPException 400: If ``sample_id`` is not a valid UUID.
        HTTPException 409: If a sequence with the same content_hash already exists.
    """
    try:
        sample_uuid = uuid.UUID(body.sample_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid sample_id UUID: {body.sample_id}",
        ) from exc

    content_hash = _compute_content_hash(body)

    # Deduplication check
    existing_result = await db.execute(
        select(AntibodySequence).where(AntibodySequence.content_hash == content_hash)
    )
    if existing_result.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Sequence with content_hash {content_hash} already exists",
        )

    cdr3_length: Optional[int] = len(body.cdr3_aa) if body.cdr3_aa else None

    seq = AntibodySequence(
        seq_id=uuid.uuid4(),
        sample_id=sample_uuid,
        chain_type=body.chain_type,
        vh_gene=body.vh_gene,
        dh_gene=body.dh_gene,
        jh_gene=body.jh_gene,
        vl_gene=body.vl_gene,
        jl_gene=body.jl_gene,
        cdr1_aa=body.cdr1_aa,
        cdr2_aa=body.cdr2_aa,
        cdr3_aa=body.cdr3_aa,
        cdr3_length=cdr3_length,
        full_vh_aa=body.full_vh_aa,
        full_vl_aa=body.full_vl_aa,
        isotype=body.isotype,
        clone_id=body.clone_id,
        read_count=body.read_count,
        content_hash=content_hash,
    )
    db.add(seq)
    await db.flush()
    await db.refresh(seq)
    return _orm_to_sequence_read(seq)


@router.get("/{seq_id}", response_model=SequenceRead)
async def get_sequence(
    seq_id: str,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
) -> SequenceRead:
    """Retrieve a single antibody sequence by its UUID.

    The response includes nested ``expression_results`` and ``assay_results``.

    Args:
        seq_id: UUID string of the antibody sequence to fetch.
        db: Injected async database session.
        _: Validated API key (unused directly).

    Returns:
        ``SequenceRead`` with fully populated nested lists.

    Raises:
        HTTPException 400: If ``seq_id`` is not a valid UUID.
        HTTPException 404: If no sequence with that ID exists.
    """
    try:
        seq_uuid = uuid.UUID(seq_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid seq_id UUID: {seq_id}",
        ) from exc

    result = await db.execute(
        select(AntibodySequence).where(AntibodySequence.seq_id == seq_uuid)
    )
    seq = result.scalar_one_or_none()
    if seq is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sequence {seq_id} not found",
        )
    return _orm_to_sequence_read(seq)


@router.patch("/{seq_id}", response_model=SequenceRead)
async def update_sequence(
    seq_id: str,
    body: SequenceUpdate,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
) -> SequenceRead:
    """Partially update mutable fields of an antibody sequence.

    Only the non-None fields in the request body are applied.  If ``cdr3_aa``
    is updated, ``cdr3_length`` is automatically recalculated.

    Args:
        seq_id: UUID string of the sequence to update.
        body: Partial update payload.
        db: Injected async database session.
        _: Validated API key (unused directly).

    Returns:
        The updated ``SequenceRead``.

    Raises:
        HTTPException 400: If ``seq_id`` is not a valid UUID.
        HTTPException 404: If the sequence does not exist.
    """
    try:
        seq_uuid = uuid.UUID(seq_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid seq_id UUID: {seq_id}",
        ) from exc

    result = await db.execute(
        select(AntibodySequence).where(AntibodySequence.seq_id == seq_uuid)
    )
    seq = result.scalar_one_or_none()
    if seq is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sequence {seq_id} not found",
        )

    update_data = body.model_dump(exclude_none=True)

    # Recompute cdr3_length if cdr3_aa is being updated
    if "cdr3_aa" in update_data:
        cdr3 = update_data["cdr3_aa"]
        update_data["cdr3_length"] = len(cdr3) if cdr3 else None

    for field, value in update_data.items():
        setattr(seq, field, value)

    await db.flush()
    await db.refresh(seq)
    return _orm_to_sequence_read(seq)


@router.delete("/{seq_id}", status_code=status.HTTP_405_METHOD_NOT_ALLOWED)
async def delete_sequence(
    seq_id: str,
    response: Response,
) -> dict[str, str]:
    """Sequences are immutable once created — deletion is not permitted.

    Args:
        seq_id: UUID string of the sequence (ignored).
        response: FastAPI response object used to set the Allow header.

    Returns:
        A JSON body explaining why deletion is forbidden.
    """
    response.headers["Allow"] = "GET, POST, PATCH"
    return {
        "detail": (
            "Antibody sequences are immutable records and cannot be deleted. "
            "Allowed methods: GET, POST, PATCH."
        )
    }
