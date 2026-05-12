from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import verify_api_key
from ..database import get_db
from ..models import BCellSample
from ..schemas import PaginatedSamples, SampleCreate, SampleRead, SampleUpdate

router = APIRouter(prefix="/samples", tags=["samples"])


def _orm_to_sample_read(obj: BCellSample) -> SampleRead:
    """Convert a BCellSample ORM instance to a SampleRead schema.

    Handles UUID-to-string conversion for all ID fields and nested flow runs.

    Args:
        obj: A BCellSample ORM object with ``flow_runs`` already loaded.

    Returns:
        A fully populated ``SampleRead`` instance.
    """
    from ..schemas import FlowRunRead

    flow_runs = []
    for run in obj.flow_runs:
        flow_runs.append(
            FlowRunRead(
                run_id=str(run.run_id),
                sample_id=str(run.sample_id),
                run_date=run.run_date,
                panel_name=run.panel_name,
                b_cell_gate_pct=run.b_cell_gate_pct,
                cd19_pct=run.cd19_pct,
                cd20_pct=run.cd20_pct,
                cd38_pct=run.cd38_pct,
                cd138_pct=run.cd138_pct,
                naive_b_pct=run.naive_b_pct,
                memory_b_pct=run.memory_b_pct,
                plasmablast_pct=run.plasmablast_pct,
                qc_pass=run.qc_pass,
            )
        )

    return SampleRead(
        sample_id=str(obj.sample_id),
        donor_id=str(obj.donor_id),
        collection_date=obj.collection_date,
        cell_count_1e6=obj.cell_count_1e6,
        viability_pct=obj.viability_pct,
        storage_condition=obj.storage_condition,
        notes=obj.notes,
        created_at=obj.created_at,
        flow_runs=flow_runs,
    )


@router.get("", response_model=PaginatedSamples)
async def list_samples(
    page: int = Query(1, ge=1, description="1-based page number"),
    size: int = Query(20, ge=1, le=200, description="Items per page"),
    donor_id: Optional[str] = Query(None, description="Filter by donor UUID"),
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
) -> PaginatedSamples:
    """List B-cell samples with optional donor_id filter.

    Returns a paginated response containing the total row count, the current page,
    the page size, and the list of sample records (each with nested flow runs).

    Args:
        page: 1-based page index.
        size: Number of items per page (max 200).
        donor_id: Optional UUID string to filter by a specific donor.
        db: Injected async database session.
        _: Validated API key (unused directly).

    Returns:
        ``PaginatedSamples`` with ``total``, ``page``, ``size``, and ``items``.

    Raises:
        HTTPException 400: If ``donor_id`` is not a valid UUID.
    """
    base_stmt = select(BCellSample)

    if donor_id is not None:
        try:
            donor_uuid = uuid.UUID(donor_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid donor_id UUID: {donor_id}",
            ) from exc
        base_stmt = base_stmt.where(BCellSample.donor_id == donor_uuid)

    # Count total matching rows
    count_stmt = select(func.count()).select_from(base_stmt.subquery())
    total_result = await db.execute(count_stmt)
    total: int = total_result.scalar_one()

    # Fetch the page
    offset = (page - 1) * size
    page_stmt = base_stmt.offset(offset).limit(size)
    result = await db.execute(page_stmt)
    samples = result.scalars().all()

    items = [_orm_to_sample_read(s) for s in samples]
    return PaginatedSamples(total=total, page=page, size=size, items=items)


@router.post("", response_model=SampleRead, status_code=status.HTTP_201_CREATED)
async def create_sample(
    body: SampleCreate,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
) -> SampleRead:
    """Create a new B-cell sample record.

    Validates that the ``donor_id`` in the request body is a well-formed UUID,
    then persists the record and returns the full ``SampleRead`` representation.

    Args:
        body: Request body containing sample fields.
        db: Injected async database session.
        _: Validated API key (unused directly).

    Returns:
        The newly created ``SampleRead`` with a generated ``sample_id``.

    Raises:
        HTTPException 400: If ``donor_id`` is not a valid UUID.
    """
    try:
        donor_uuid = uuid.UUID(body.donor_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid donor_id UUID: {body.donor_id}",
        ) from exc

    sample = BCellSample(
        sample_id=uuid.uuid4(),
        donor_id=donor_uuid,
        collection_date=body.collection_date,
        cell_count_1e6=body.cell_count_1e6,
        viability_pct=body.viability_pct,
        storage_condition=body.storage_condition,
        notes=body.notes,
    )
    db.add(sample)
    await db.flush()  # assigns PK without committing; commit happens in get_db on yield exit

    # Re-fetch to load relationships (flow_runs will be empty for a new record)
    await db.refresh(sample)
    return _orm_to_sample_read(sample)


@router.get("/{sample_id}", response_model=SampleRead)
async def get_sample(
    sample_id: str,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
) -> SampleRead:
    """Retrieve a single B-cell sample by its UUID, including nested flow cytometry runs.

    Args:
        sample_id: UUID string of the sample to fetch.
        db: Injected async database session.
        _: Validated API key (unused directly).

    Returns:
        ``SampleRead`` with ``flow_runs`` populated.

    Raises:
        HTTPException 400: If ``sample_id`` is not a valid UUID.
        HTTPException 404: If no sample with that ID exists.
    """
    try:
        sample_uuid = uuid.UUID(sample_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid sample_id UUID: {sample_id}",
        ) from exc

    result = await db.execute(
        select(BCellSample).where(BCellSample.sample_id == sample_uuid)
    )
    sample = result.scalar_one_or_none()
    if sample is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample {sample_id} not found",
        )
    return _orm_to_sample_read(sample)


@router.patch("/{sample_id}", response_model=SampleRead)
async def update_sample(
    sample_id: str,
    body: SampleUpdate,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
) -> SampleRead:
    """Partially update a B-cell sample.

    Only fields present (non-None) in the request body are updated. Fields not
    supplied retain their current values.

    Args:
        sample_id: UUID string of the sample to update.
        body: Partial update payload; only non-None fields are applied.
        db: Injected async database session.
        _: Validated API key (unused directly).

    Returns:
        The updated ``SampleRead``.

    Raises:
        HTTPException 400: If ``sample_id`` or ``donor_id`` is not a valid UUID.
        HTTPException 404: If the sample does not exist.
    """
    try:
        sample_uuid = uuid.UUID(sample_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid sample_id UUID: {sample_id}",
        ) from exc

    result = await db.execute(
        select(BCellSample).where(BCellSample.sample_id == sample_uuid)
    )
    sample = result.scalar_one_or_none()
    if sample is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample {sample_id} not found",
        )

    update_data = body.model_dump(exclude_none=True)
    if "donor_id" in update_data:
        try:
            update_data["donor_id"] = uuid.UUID(update_data["donor_id"])
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid donor_id UUID: {update_data['donor_id']}",
            ) from exc

    for field, value in update_data.items():
        setattr(sample, field, value)

    await db.flush()
    await db.refresh(sample)
    return _orm_to_sample_read(sample)


@router.delete("/{sample_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_sample(
    sample_id: str,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
) -> None:
    """Hard-delete a B-cell sample record.

    Args:
        sample_id: UUID string of the sample to delete.
        db: Injected async database session.
        _: Validated API key (unused directly).

    Raises:
        HTTPException 400: If ``sample_id`` is not a valid UUID.
        HTTPException 404: If no sample with that ID exists.
    """
    try:
        sample_uuid = uuid.UUID(sample_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid sample_id UUID: {sample_id}",
        ) from exc

    result = await db.execute(
        select(BCellSample).where(BCellSample.sample_id == sample_uuid)
    )
    sample = result.scalar_one_or_none()
    if sample is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample {sample_id} not found",
        )

    await db.delete(sample)
    await db.flush()
