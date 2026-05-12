from __future__ import annotations

from typing import Any

from ..database import get_cursor
from .base import BaseCheck, CheckResult


class BCellGateOutOfRangeCheck(BaseCheck):
    """Check for flow cytometry runs with B-cell gate percentages outside expected bounds.

    Typical B-cell gates in PBMCs from healthy donors range from ~2% to ~15%.
    Very low values may indicate poor sample preparation or gating errors; very
    high values may indicate a B-cell-enriched sample not reflected in metadata.

    Threshold defaults:
        - Minimum: ``config.b_cell_gate_min_pct`` (default 0.5%)
        - Maximum: ``config.b_cell_gate_max_pct`` (default 60.0%)

    Status levels:
        - ``pass``: All runs are within the configured range.
        - ``warn``: One or more runs fall outside the configured range.
    """

    name: str = "b_cell_gate_out_of_range"

    def run(self) -> CheckResult:
        """Query for flow runs with ``b_cell_gate_pct`` outside the configured range.

        Returns:
            ``CheckResult`` with ``status="warn"`` if out-of-range runs are found,
            otherwise ``status="pass"``.
        """
        query = f"""
            SELECT run_id::text, sample_id::text, b_cell_gate_pct
            FROM flow_cytometry_runs
            WHERE
                b_cell_gate_pct IS NOT NULL
                AND (
                    b_cell_gate_pct < {self.config.b_cell_gate_min_pct}
                    OR b_cell_gate_pct > {self.config.b_cell_gate_max_pct}
                )
        """
        with get_cursor(self.database_url) as cur:
            cur.execute(query)
            rows = cur.fetchall()

        if not rows:
            return self._pass(
                f"All B-cell gate percentages are within "
                f"{self.config.b_cell_gate_min_pct}–"
                f"{self.config.b_cell_gate_max_pct}%."
            )

        details: list[dict[str, Any]] = [
            {
                "run_id": row["run_id"],
                "sample_id": row["sample_id"],
                "b_cell_gate_pct": row["b_cell_gate_pct"],
            }
            for row in rows
        ]
        n = len(rows)
        return self._warn(
            affected_rows=n,
            details=details,
            message=(
                f"{n} flow cytometry run(s) have B-cell gate percentages outside "
                f"the expected range "
                f"({self.config.b_cell_gate_min_pct}–{self.config.b_cell_gate_max_pct}%)."
            ),
        )


class GateSumExceeds100Check(BaseCheck):
    """Check for runs where the naive + memory + plasmablast subset percentages sum to > 100%.

    Gate percentages within a population must sum to ≤ 100%.  Values exceeding
    100% (with a 0.5% tolerance) are biologically impossible and indicate a
    gating or data entry error.

    The tolerance threshold is hardcoded at 100.5% to allow for floating-point
    rounding in acquisition software.

    Status levels:
        - ``pass``: All runs have valid gate sums.
        - ``fail``: One or more runs have subset percentages summing above 100.5%.
    """

    name: str = "gate_sum_exceeds_100"

    def run(self) -> CheckResult:
        """Query for runs where naive_b_pct + memory_b_pct + plasmablast_pct > 100.5.

        Returns:
            ``CheckResult`` with ``status="fail"`` if any gate sums exceed 100.5%,
            otherwise ``status="pass"``.
        """
        query = """
            SELECT
                run_id::text,
                sample_id::text,
                naive_b_pct,
                memory_b_pct,
                plasmablast_pct,
                (naive_b_pct + memory_b_pct + plasmablast_pct) AS gate_sum
            FROM flow_cytometry_runs
            WHERE
                naive_b_pct IS NOT NULL
                AND memory_b_pct IS NOT NULL
                AND plasmablast_pct IS NOT NULL
                AND (naive_b_pct + memory_b_pct + plasmablast_pct) > 100.5
        """
        with get_cursor(self.database_url) as cur:
            cur.execute(query)
            rows = cur.fetchall()

        if not rows:
            return self._pass(
                "No flow runs have subset gate percentages summing above 100.5%."
            )

        details: list[dict[str, Any]] = [
            {
                "run_id": row["run_id"],
                "sample_id": row["sample_id"],
                "naive_b_pct": row["naive_b_pct"],
                "memory_b_pct": row["memory_b_pct"],
                "plasmablast_pct": row["plasmablast_pct"],
                "gate_sum": round(float(row["gate_sum"]), 2),
            }
            for row in rows
        ]
        n = len(rows)
        return self._fail(
            affected_rows=n,
            details=details,
            message=(
                f"{n} flow cytometry run(s) have naive + memory + plasmablast "
                f"gate percentages summing above 100.5% (biologically impossible)."
            ),
        )


class QCFlagMismatchCheck(BaseCheck):
    """Identify runs marked as QC-pass but with suspicious B-cell gate percentages.

    A run flagged ``qc_pass = TRUE`` is expected to have a B-cell gate in the
    physiologically plausible range (1.0–50.0%).  Runs outside this range that
    are nonetheless marked as passing may indicate an incorrect QC call.

    Status levels:
        - ``pass``: No QC-pass runs have suspicious gate percentages.
        - ``warn``: One or more QC-pass runs have B-cell gates < 1.0% or > 50.0%.
    """

    name: str = "qc_flag_mismatch"

    def run(self) -> CheckResult:
        """Query for QC-pass runs with B-cell gates outside 1.0–50.0%.

        Returns:
            ``CheckResult`` with ``status="warn"`` if mismatched runs are found,
            otherwise ``status="pass"``.
        """
        query = """
            SELECT run_id::text, sample_id::text, b_cell_gate_pct, qc_pass
            FROM flow_cytometry_runs
            WHERE
                qc_pass = TRUE
                AND b_cell_gate_pct IS NOT NULL
                AND (b_cell_gate_pct < 1.0 OR b_cell_gate_pct > 50.0)
        """
        with get_cursor(self.database_url) as cur:
            cur.execute(query)
            rows = cur.fetchall()

        if not rows:
            return self._pass(
                "No QC-pass runs have suspicious B-cell gate percentages."
            )

        details: list[dict[str, Any]] = [
            {
                "run_id": row["run_id"],
                "sample_id": row["sample_id"],
                "b_cell_gate_pct": row["b_cell_gate_pct"],
                "qc_pass": row["qc_pass"],
            }
            for row in rows
        ]
        n = len(rows)
        return self._warn(
            affected_rows=n,
            details=details,
            message=(
                f"{n} flow run(s) are flagged qc_pass=TRUE but have a B-cell gate "
                f"< 1.0% or > 50.0% (possible QC call error)."
            ),
        )
