"""Database loader for the bioetl-pipeline.

Provides bulk upsert operations using PostgreSQL's
``INSERT ... ON CONFLICT (content_hash) DO NOTHING`` strategy for idempotent
ingestion: re-running the pipeline with the same records is safe and
deterministic.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Type

import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from pipeline.logger import get_logger

_log = get_logger(__name__)


def _get_model_columns(model_class: Any) -> list[str]:
    """Return a list of column names for the given SQLAlchemy ORM model class.

    Args:
        model_class: A SQLAlchemy declarative model class.

    Returns:
        List of column name strings.
    """
    mapper = inspect(model_class)
    return [col.key for col in mapper.columns]


def _prepare_record(
    record: dict[str, Any],
    model_class: Any,
    pk_col: str = "seq_id",
) -> dict[str, Any]:
    """Prepare a raw record dict for insertion into the given model table.

    - Ensures a primary key is set (generates a UUID if absent).
    - Adds ``ingested_at`` if the column exists and is not already set.
    - Removes keys that are not valid columns in the model.

    Args:
        record: Raw record dictionary from a transformer/validator.
        model_class: SQLAlchemy ORM model class for the target table.
        pk_col: Name of the primary key column.

    Returns:
        A cleaned record dict containing only valid model columns.
    """
    valid_cols = set(_get_model_columns(model_class))
    cleaned: dict[str, Any] = {}

    for col in valid_cols:
        if col in record:
            val = record[col]
            # Convert pandas NA/NaT/float NaN to Python None
            if val is not None and isinstance(val, float) and pd.isna(val):
                val = None
            elif hasattr(val, "__class__") and val.__class__.__name__ in ("NAType", "NaTType"):
                val = None
            cleaned[col] = val

    # Ensure PK is set
    if pk_col in valid_cols and not cleaned.get(pk_col):
        cleaned[pk_col] = str(uuid.uuid4())

    # Ensure ingested_at is set
    if "ingested_at" in valid_cols and not cleaned.get("ingested_at"):
        cleaned["ingested_at"] = datetime.utcnow()

    return cleaned


def upsert_records(
    session: Session,
    model_class: Any,
    records: list[dict[str, Any]],
) -> tuple[int, int]:
    """Bulk upsert records using ``INSERT ... ON CONFLICT (content_hash) DO NOTHING``.

    This is a PostgreSQL-specific upsert. Each record is inserted; if the
    ``content_hash`` already exists in the table, the conflicting row is
    silently skipped.

    Implementation note: Since SQLAlchemy Core's ``insert().on_conflict_do_nothing``
    requires ``sqlalchemy.dialects.postgresql``, and to keep the loader database-agnostic
    for unit testing, we use a try/except approach: attempt the PostgreSQL dialect upsert
    first, falling back to individual INSERT with exception handling for non-PostgreSQL
    databases.

    Args:
        session: An open SQLAlchemy :class:`Session`.
        model_class: Target ORM model class.
        records: List of record dicts to insert.

    Returns:
        A 2-tuple ``(n_inserted, n_skipped)`` counting outcomes.
    """
    if not records:
        return 0, 0

    n_inserted = 0
    n_skipped = 0
    table = model_class.__table__

    try:
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        stmt = pg_insert(table).values(records)
        stmt = stmt.on_conflict_do_nothing(index_elements=["content_hash"])
        result = session.execute(stmt)
        # rowcount reflects actually inserted rows in PostgreSQL with ON CONFLICT DO NOTHING
        rowcount = result.rowcount if result.rowcount >= 0 else len(records)
        n_inserted = rowcount
        n_skipped = len(records) - n_inserted

    except ImportError:
        # Fallback for non-PostgreSQL databases (SQLite for testing)
        from sqlalchemy.exc import IntegrityError

        for rec in records:
            try:
                session.execute(table.insert(), rec)
                n_inserted += 1
            except IntegrityError:
                session.rollback()
                n_skipped += 1

    _log.debug(
        "upsert_batch_complete",
        table=table.name,
        n_inserted=n_inserted,
        n_skipped=n_skipped,
    )
    return n_inserted, n_skipped


def load_dataframe(
    session: Session,
    model_class: Any,
    df: pd.DataFrame,
    batch_size: int = 100,
) -> tuple[int, int]:
    """Convert a DataFrame to record dicts and upsert in batches.

    Determines the correct primary key column name from the model, prepares
    each record, and calls :func:`upsert_records` for each batch.

    Args:
        session: An open SQLAlchemy :class:`Session`.
        model_class: Target ORM model class.
        df: Validated and transformed :class:`pd.DataFrame`.
        batch_size: Number of records per upsert batch.

    Returns:
        Total ``(n_inserted, n_skipped)`` across all batches.
    """
    if df.empty:
        _log.info("load_dataframe_empty_input", table=model_class.__tablename__)
        return 0, 0

    # Determine primary key column name from model mapper
    mapper = inspect(model_class)
    pk_cols = [col.key for col in mapper.primary_key]
    pk_col = pk_cols[0] if pk_cols else "id"

    records = df.to_dict(orient="records")
    prepared = [_prepare_record(rec, model_class, pk_col=pk_col) for rec in records]

    total_inserted = 0
    total_skipped = 0

    for batch_start in range(0, len(prepared), batch_size):
        batch = prepared[batch_start : batch_start + batch_size]
        n_ins, n_skip = upsert_records(session, model_class, batch)
        total_inserted += n_ins
        total_skipped += n_skip
        _log.info(
            "load_batch_progress",
            table=model_class.__tablename__,
            batch_start=batch_start,
            batch_end=batch_start + len(batch),
            n_inserted=n_ins,
            n_skipped=n_skip,
        )

    _log.info(
        "load_dataframe_complete",
        table=model_class.__tablename__,
        total_records=len(prepared),
        total_inserted=total_inserted,
        total_skipped=total_skipped,
    )
    return total_inserted, total_skipped
