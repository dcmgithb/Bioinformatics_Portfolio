from __future__ import annotations

from typing import Any

from ..database import get_cursor
from .base import BaseCheck, CheckResult


class OrphanSamplesCheck(BaseCheck):
    """Identify B-cell samples that have no associated flow cytometry run.

    Every sample collected should have at least one flow cytometry characterisation
    run.  Samples without any run may indicate incomplete data entry or samples
    that were processed but not yet analysed.

    Status levels:
        - ``pass``: All samples have at least one flow cytometry run.
        - ``warn``: One or more samples have no associated flow cytometry runs.
    """

    name: str = "orphan_samples"

    def run(self) -> CheckResult:
        """Query for samples with no matching rows in ``flow_cytometry_runs``.

        Uses a LEFT JOIN to find samples where the corresponding ``run_id`` is NULL.

        Returns:
            ``CheckResult`` with ``status="warn"`` if orphan samples are found,
            otherwise ``status="pass"``.
        """
        query = """
            SELECT
                s.sample_id::text AS sample_id,
                s.donor_id::text AS donor_id,
                s.collection_date::text AS collection_date
            FROM b_cell_samples s
            LEFT JOIN flow_cytometry_runs f ON s.sample_id = f.sample_id
            WHERE f.run_id IS NULL
        """
        with get_cursor(self.database_url) as cur:
            cur.execute(query)
            rows = cur.fetchall()

        if not rows:
            return self._pass("All B-cell samples have at least one flow cytometry run.")

        details: list[dict[str, Any]] = [
            {
                "sample_id": row["sample_id"],
                "donor_id": row["donor_id"],
                "collection_date": row["collection_date"],
            }
            for row in rows
        ]
        n = len(rows)
        return self._warn(
            affected_rows=n,
            details=details,
            message=f"{n} sample(s) have no associated flow cytometry runs.",
        )


class OrphanSequencesCheck(BaseCheck):
    """Identify antibody sequences that have no associated assay results.

    Sequences that have been imported but never taken forward into functional
    assays may represent incomplete workflows.  This is a soft warning rather
    than a hard failure, as sequences may legitimately be awaiting assay.

    Status levels:
        - ``pass``: All sequences have at least one assay result.
        - ``warn``: One or more sequences have no associated assay results.
    """

    name: str = "orphan_sequences"

    def run(self) -> CheckResult:
        """Query for antibody sequences with no matching rows in ``assay_results``.

        Returns:
            ``CheckResult`` with ``status="warn"`` if orphan sequences are found,
            otherwise ``status="pass"``.
        """
        query = """
            SELECT
                ab.seq_id::text AS seq_id,
                ab.sample_id::text AS sample_id,
                ab.clone_id,
                ab.isotype
            FROM antibody_sequences ab
            LEFT JOIN assay_results ar ON ab.seq_id = ar.seq_id
            WHERE ar.assay_id IS NULL
        """
        with get_cursor(self.database_url) as cur:
            cur.execute(query)
            rows = cur.fetchall()

        if not rows:
            return self._pass("All antibody sequences have at least one assay result.")

        details: list[dict[str, Any]] = [
            {
                "seq_id": row["seq_id"],
                "sample_id": row["sample_id"],
                "clone_id": row["clone_id"],
                "isotype": row["isotype"],
            }
            for row in rows
        ]
        n = len(rows)
        return self._warn(
            affected_rows=n,
            details=details,
            message=f"{n} antibody sequence(s) have no associated assay results.",
        )


class AssayWithoutSequenceCheck(BaseCheck):
    """Detect assay results that reference a non-existent antibody sequence.

    This represents a referential integrity violation: an ``assay_result`` row
    points to a ``seq_id`` that does not exist in ``antibody_sequences``.  This
    should not occur in a properly constrained database but may arise from manual
    data migrations or constraint violations.

    Status levels:
        - ``pass``: All assay results reference valid sequences.
        - ``fail``: One or more assay results reference missing sequences.
    """

    name: str = "assay_without_sequence"

    def run(self) -> CheckResult:
        """Query for assay results whose ``seq_id`` foreign key has no matching sequence.

        Returns:
            ``CheckResult`` with ``status="fail"`` if orphan assay results exist,
            otherwise ``status="pass"``.
        """
        query = """
            SELECT
                ar.assay_id::text AS assay_id,
                ar.seq_id::text AS seq_id,
                ar.assay_type,
                ar.assay_date::text AS assay_date
            FROM assay_results ar
            LEFT JOIN antibody_sequences ab ON ar.seq_id = ab.seq_id
            WHERE ab.seq_id IS NULL
        """
        with get_cursor(self.database_url) as cur:
            cur.execute(query)
            rows = cur.fetchall()

        if not rows:
            return self._pass(
                "All assay results reference a valid antibody sequence (referential integrity OK)."
            )

        details: list[dict[str, Any]] = [
            {
                "assay_id": row["assay_id"],
                "seq_id": row["seq_id"],
                "assay_type": row["assay_type"],
                "assay_date": row["assay_date"],
            }
            for row in rows
        ]
        n = len(rows)
        return self._fail(
            affected_rows=n,
            details=details,
            message=(
                f"{n} assay result(s) reference a seq_id that does not exist in "
                f"antibody_sequences (referential integrity violation)."
            ),
        )
