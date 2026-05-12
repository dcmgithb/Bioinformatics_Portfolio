from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import verify_api_key
from ..database import get_db
from ..models import AntibodySequence, AssayResult, ExpressionResult
from ..schemas import (
    AssayStat,
    CloneDiversityStats,
    ConstructStat,
    ReportResponse,
)

router = APIRouter(tags=["reports"])


def _compute_shannon_diversity(clone_reads: list[int]) -> float:
    """Compute the Shannon diversity index H = -Σ p_i * ln(p_i).

    Args:
        clone_reads: A list of per-clone read counts (all positive integers).

    Returns:
        The Shannon entropy H as a float. Returns 0.0 for empty or single-element input.
    """
    total = sum(clone_reads)
    if total == 0 or len(clone_reads) == 0:
        return 0.0
    h = 0.0
    for reads in clone_reads:
        p = reads / total
        if p > 0:
            h -= p * math.log(p)
    return round(h, 6)


def _compute_d50(clone_reads: list[int]) -> int:
    """Compute D50: the minimum number of clones that together cover >= 50% of total reads.

    Clones are sorted in descending order of read count before cumulative summing.

    Args:
        clone_reads: A list of per-clone read counts.

    Returns:
        The D50 integer value.  Returns 0 for empty input.
    """
    if not clone_reads:
        return 0
    total = sum(clone_reads)
    threshold = total * 0.5
    sorted_reads = sorted(clone_reads, reverse=True)
    cumsum = 0
    for i, reads in enumerate(sorted_reads, start=1):
        cumsum += reads
        if cumsum >= threshold:
            return i
    return len(sorted_reads)


@router.get("", response_model=ReportResponse)
async def get_report(
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
) -> ReportResponse:
    """Generate a summary report covering clone diversity, top constructs, and assay statistics.

    **Clone Diversity** is computed from ``antibody_sequences`` rows that have a
    non-null ``clone_id``.  Per-clone read counts are aggregated, then Shannon H
    and D50 are derived from those aggregates.

    **Top Constructs** returns up to 10 construct/expression-system combinations
    ranked by mean yield (mg/L).

    **Assay Summary** groups ``assay_results`` by ``assay_type`` and reports the
    total count, number of passing runs, pass-rate percentage, and median KD (nM).

    Args:
        db: Injected async database session.
        _: Validated API key (unused directly).

    Returns:
        A ``ReportResponse`` with ``generated_at``, ``clone_diversity``,
        ``top_constructs``, and ``assay_summary``.
    """
    # ------------------------------------------------------------------
    # 1. Clone diversity stats
    # ------------------------------------------------------------------
    clone_stmt = (
        select(
            AntibodySequence.clone_id,
            func.sum(AntibodySequence.read_count).label("total_reads"),
        )
        .where(AntibodySequence.clone_id.isnot(None))
        .group_by(AntibodySequence.clone_id)
    )
    clone_result = await db.execute(clone_stmt)
    clone_rows = clone_result.all()

    clone_reads_list: list[int] = [int(row.total_reads) for row in clone_rows if row.total_reads]
    total_clones = len(clone_reads_list)
    total_clone_reads = sum(clone_reads_list)

    # Total sequences (all rows, not just cloned ones)
    total_seq_result = await db.execute(select(func.count()).select_from(AntibodySequence))
    total_sequences: int = total_seq_result.scalar_one()

    shannon_h = _compute_shannon_diversity(clone_reads_list)
    d50 = _compute_d50(clone_reads_list)
    top_clone_freq = (
        (max(clone_reads_list) / total_clone_reads * 100.0) if total_clone_reads > 0 else 0.0
    )

    clone_diversity = CloneDiversityStats(
        total_sequences=total_sequences,
        total_clones=total_clones,
        shannon_diversity_index=round(shannon_h, 4),
        d50=d50,
        top_clone_frequency_pct=round(top_clone_freq, 2),
    )

    # ------------------------------------------------------------------
    # 2. Top 10 constructs by mean yield
    # ------------------------------------------------------------------
    construct_stmt = (
        select(
            ExpressionResult.construct_name,
            ExpressionResult.expression_system,
            func.avg(ExpressionResult.yield_mg_l).label("mean_yield"),
            func.count(ExpressionResult.result_id).label("n_runs"),
        )
        .where(ExpressionResult.yield_mg_l.isnot(None))
        .group_by(ExpressionResult.construct_name, ExpressionResult.expression_system)
        .order_by(func.avg(ExpressionResult.yield_mg_l).desc())
        .limit(10)
    )
    construct_result = await db.execute(construct_stmt)
    construct_rows = construct_result.all()

    top_constructs: list[ConstructStat] = [
        ConstructStat(
            construct_name=row.construct_name or "unknown",
            expression_system=row.expression_system or "unknown",
            mean_yield_mg_l=round(float(row.mean_yield), 2),
            n_runs=int(row.n_runs),
        )
        for row in construct_rows
    ]

    # ------------------------------------------------------------------
    # 3. Assay pass rates per assay_type
    # ------------------------------------------------------------------
    assay_stmt = select(AssayResult).where(AssayResult.assay_type.isnot(None))
    assay_result = await db.execute(assay_stmt)
    assay_rows = assay_result.scalars().all()

    # Group in Python to allow median computation without window functions
    assay_groups: dict[str, list[AssayResult]] = {}
    for row in assay_rows:
        key = row.assay_type or "unknown"
        assay_groups.setdefault(key, []).append(row)

    assay_summary: list[AssayStat] = []
    for assay_type, rows in assay_groups.items():
        total_n = len(rows)
        passed_n = sum(1 for r in rows if (r.pass_fail or "").upper() == "PASS")
        pass_rate = (passed_n / total_n * 100.0) if total_n > 0 else 0.0

        kd_values: list[float] = [r.binding_kd_nm for r in rows if r.binding_kd_nm is not None]  # type: ignore[misc]
        median_kd: Optional[float] = None
        if kd_values:
            sorted_kd = sorted(kd_values)
            mid = len(sorted_kd) // 2
            if len(sorted_kd) % 2 == 0:
                median_kd = round((sorted_kd[mid - 1] + sorted_kd[mid]) / 2.0, 4)
            else:
                median_kd = round(sorted_kd[mid], 4)

        assay_summary.append(
            AssayStat(
                assay_type=assay_type,
                total=total_n,
                passed=passed_n,
                pass_rate_pct=round(pass_rate, 2),
                median_kd_nm=median_kd,
            )
        )

    # Sort assay summary by assay_type for deterministic output
    assay_summary.sort(key=lambda x: x.assay_type)

    return ReportResponse(
        generated_at=datetime.utcnow(),
        clone_diversity=clone_diversity,
        top_constructs=top_constructs,
        assay_summary=assay_summary,
    )
