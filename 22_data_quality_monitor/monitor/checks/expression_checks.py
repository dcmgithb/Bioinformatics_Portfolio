from __future__ import annotations

from typing import Any

from ..database import get_cursor
from .base import BaseCheck, CheckResult


def _quartiles(values: list[float]) -> tuple[float, float]:
    """Compute Q1 and Q3 of a sorted list of floats using linear interpolation.

    Args:
        values: A non-empty list of numeric values (need not be pre-sorted).

    Returns:
        A ``(Q1, Q3)`` tuple.
    """
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def _percentile(p: float) -> float:
        """Compute the p-th percentile (0 ≤ p ≤ 100) via linear interpolation."""
        idx = (p / 100.0) * (n - 1)
        lower = int(idx)
        upper = lower + 1
        frac = idx - lower
        if upper >= n:
            return float(sorted_vals[lower])
        return float(sorted_vals[lower]) * (1 - frac) + float(sorted_vals[upper]) * frac

    return _percentile(25), _percentile(75)


class OutlierYieldCheck(BaseCheck):
    """Detect expression yield values that fall outside the IQR-based outlier fence.

    For each expression system, Q1, Q3, and IQR are computed from yields.  Any
    yield below ``Q1 - multiplier * IQR`` or above ``Q3 + multiplier * IQR`` is
    flagged.  The multiplier defaults to 1.5 (standard Tukey fences).

    Status levels:
        - ``pass``: No outlier yields detected.
        - ``warn``: One or more yields fall outside the Tukey fence for their system.
    """

    name: str = "outlier_expression_yield"

    def run(self) -> CheckResult:
        """Fetch all yields, group by expression system, and flag Tukey outliers.

        Returns:
            ``CheckResult`` with ``status="warn"`` if outliers are detected,
            otherwise ``status="pass"``.
        """
        query = """
            SELECT
                result_id::text,
                seq_id::text,
                construct_name,
                expression_system,
                yield_mg_l
            FROM expression_results
            WHERE yield_mg_l IS NOT NULL
        """
        with get_cursor(self.database_url) as cur:
            cur.execute(query)
            rows = cur.fetchall()

        if not rows:
            return self._pass("No expression yield data found.")

        # Group yields by expression system
        by_system: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            system = row["expression_system"] or "unknown"
            by_system.setdefault(system, []).append(dict(row))

        multiplier = self.config.yield_iqr_multiplier
        details: list[dict[str, Any]] = []

        for system, group_rows in by_system.items():
            yields = [r["yield_mg_l"] for r in group_rows]
            if len(yields) < 4:
                # Not enough data to compute meaningful quartiles — skip
                continue
            q1, q3 = _quartiles(yields)
            iqr = q3 - q1
            lower_fence = q1 - multiplier * iqr
            upper_fence = q3 + multiplier * iqr

            for r in group_rows:
                y = r["yield_mg_l"]
                if y < lower_fence or y > upper_fence:
                    details.append(
                        {
                            "result_id": r["result_id"],
                            "expression_system": system,
                            "yield_mg_l": y,
                            "q1": round(q1, 2),
                            "q3": round(q3, 2),
                            "lower_fence": round(lower_fence, 2),
                            "upper_fence": round(upper_fence, 2),
                        }
                    )

        if not details:
            return self._pass(
                f"No outlier yields detected (Tukey multiplier = {multiplier})."
            )

        n = len(details)
        return self._warn(
            affected_rows=n,
            details=details,
            message=(
                f"{n} expression yield value(s) fall outside the Tukey "
                f"IQR fence (multiplier = {multiplier})."
            ),
        )


class NegativeYieldCheck(BaseCheck):
    """Check for expression results with physically impossible negative yields.

    A yield below zero indicates a data entry error or instrument calibration
    problem and is an unambiguous data integrity failure.

    Status levels:
        - ``pass``: All yields are non-negative (or NULL).
        - ``fail``: One or more yields are strictly negative.
    """

    name: str = "negative_yield"

    def run(self) -> CheckResult:
        """Query for expression results where ``yield_mg_l < 0``.

        Returns:
            ``CheckResult`` with ``status="fail"`` if negative yields exist,
            otherwise ``status="pass"``.
        """
        query = """
            SELECT result_id::text, seq_id::text, yield_mg_l
            FROM expression_results
            WHERE yield_mg_l < 0
        """
        with get_cursor(self.database_url) as cur:
            cur.execute(query)
            rows = cur.fetchall()

        if not rows:
            return self._pass("No negative yield values detected.")

        details: list[dict[str, Any]] = [
            {"result_id": row["result_id"], "seq_id": row["seq_id"], "yield_mg_l": row["yield_mg_l"]}
            for row in rows
        ]
        n = len(rows)
        return self._fail(
            affected_rows=n,
            details=details,
            message=f"{n} expression result(s) have a negative yield_mg_l value (data entry error).",
        )


class MissingPurityCheck(BaseCheck):
    """Warn when a large fraction of expression results lack a purity measurement.

    A high proportion of NULL purity values may indicate that QC measurements
    are not being recorded consistently.  Warns when the missing fraction exceeds
    20% of all expression results.

    Status levels:
        - ``pass``: ≤ 20% of expression results have NULL purity.
        - ``warn``: > 20% of expression results have NULL purity.
    """

    name: str = "missing_purity"

    def run(self) -> CheckResult:
        """Query for expression results with NULL ``purity_pct`` and assess the fraction.

        Returns:
            ``CheckResult`` with ``status="warn"`` if > 20% of rows have NULL
            purity, otherwise ``status="pass"``.
        """
        total_query = "SELECT COUNT(*) AS total FROM expression_results"
        missing_query = """
            SELECT result_id::text, seq_id::text, construct_name
            FROM expression_results
            WHERE purity_pct IS NULL
        """
        with get_cursor(self.database_url) as cur:
            cur.execute(total_query)
            total_row = cur.fetchone()
            total: int = int(total_row["total"]) if total_row else 0

            cur.execute(missing_query)
            missing_rows = cur.fetchall()

        missing_n = len(missing_rows)

        if total == 0:
            return self._pass("No expression results found — purity check skipped.")

        missing_pct = (missing_n / total) * 100.0

        details: list[dict[str, Any]] = [
            {
                "result_id": row["result_id"],
                "seq_id": row["seq_id"],
                "construct_name": row["construct_name"],
            }
            for row in missing_rows
        ]

        if missing_pct <= 20.0:
            return self._pass(
                f"{missing_n}/{total} expression results missing purity_pct "
                f"({missing_pct:.1f}%) — within the 20% threshold."
            )

        return self._warn(
            affected_rows=missing_n,
            details=details,
            message=(
                f"{missing_n}/{total} expression results are missing purity_pct "
                f"({missing_pct:.1f}%) — exceeds the 20% threshold."
            ),
        )
