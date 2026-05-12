from __future__ import annotations

import re
from typing import Any

from ..database import get_cursor
from .base import BaseCheck, CheckResult

# Valid amino-acid alphabet for CDR3 sequences
_CDR3_PATTERN: re.Pattern[str] = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


class MissingCDR3Check(BaseCheck):
    """Check for antibody sequences that have a NULL or empty CDR3 amino-acid sequence.

    CDR3 is the most variable and functionally critical complementarity-determining
    region.  Sequences missing CDR3 cannot be characterised for specificity.

    Status levels:
        - ``pass``: All sequences have a non-empty CDR3.
        - ``fail``: One or more sequences are missing CDR3.
    """

    name: str = "missing_cdr3"

    def run(self) -> CheckResult:
        """Query for sequences with NULL or empty ``cdr3_aa`` and return a ``CheckResult``.

        Returns:
            ``CheckResult`` with ``status="fail"`` if any sequences lack CDR3,
            otherwise ``status="pass"``.
        """
        query = """
            SELECT seq_id::text, sample_id::text
            FROM antibody_sequences
            WHERE cdr3_aa IS NULL OR cdr3_aa = ''
        """
        with get_cursor(self.database_url) as cur:
            cur.execute(query)
            rows = cur.fetchall()

        if not rows:
            return self._pass("All sequences have a CDR3 amino-acid sequence.")

        details: list[dict[str, Any]] = [
            {"seq_id": row["seq_id"], "sample_id": row["sample_id"]}
            for row in rows
        ]
        n = len(rows)
        return self._fail(
            affected_rows=n,
            details=details,
            message=f"{n} sequence(s) are missing CDR3 (cdr3_aa IS NULL or empty).",
        )


class MalformedCDR3Check(BaseCheck):
    """Check for CDR3 sequences containing invalid characters or outside length thresholds.

    Valid CDR3 amino-acid sequences must:
    - Consist entirely of the 20 standard amino-acid single-letter codes
      (``ACDEFGHIKLMNPQRSTVWY``).
    - Have a length between ``config.cdr3_min_length`` and ``config.cdr3_max_length``
      (inclusive).

    Status levels:
        - ``pass``: All non-empty CDR3 sequences are well-formed.
        - ``fail``: One or more CDR3 sequences are malformed or outside length bounds.
    """

    name: str = "malformed_cdr3"

    def run(self) -> CheckResult:
        """Fetch all non-empty CDR3 sequences, validate characters and length in Python.

        Returns:
            ``CheckResult`` with ``status="fail"`` if any malformed CDR3 is found,
            otherwise ``status="pass"``.
        """
        query = """
            SELECT seq_id::text, cdr3_aa
            FROM antibody_sequences
            WHERE cdr3_aa IS NOT NULL AND cdr3_aa != ''
        """
        with get_cursor(self.database_url) as cur:
            cur.execute(query)
            rows = cur.fetchall()

        details: list[dict[str, Any]] = []
        for row in rows:
            seq_id: str = row["seq_id"]
            cdr3: str = row["cdr3_aa"]
            reason: str | None = None

            if not _CDR3_PATTERN.match(cdr3):
                reason = f"Contains invalid characters (not in ACDEFGHIKLMNPQRSTVWY): '{cdr3}'"
            elif len(cdr3) < self.config.cdr3_min_length:
                reason = (
                    f"Length {len(cdr3)} is below minimum "
                    f"{self.config.cdr3_min_length}"
                )
            elif len(cdr3) > self.config.cdr3_max_length:
                reason = (
                    f"Length {len(cdr3)} exceeds maximum "
                    f"{self.config.cdr3_max_length}"
                )

            if reason is not None:
                details.append({"seq_id": seq_id, "cdr3_aa": cdr3, "reason": reason})

        if not details:
            return self._pass(
                f"All CDR3 sequences pass character and length validation "
                f"({self.config.cdr3_min_length}–{self.config.cdr3_max_length} AA)."
            )

        n = len(details)
        return self._fail(
            affected_rows=n,
            details=details,
            message=(
                f"{n} CDR3 sequence(s) are malformed (invalid characters or "
                f"length outside {self.config.cdr3_min_length}–"
                f"{self.config.cdr3_max_length} AA)."
            ),
        )


class DuplicateSequenceCheck(BaseCheck):
    """Check for antibody sequences with duplicate ``content_hash`` values.

    A ``content_hash`` is a deterministic SHA-256 digest of the sequence fields.
    Duplicate hashes indicate that the same sequence was inserted more than once,
    which may reflect a data ingestion error or a missing deduplication step.

    Status levels:
        - ``pass``: All ``content_hash`` values are unique.
        - ``warn``: One or more ``content_hash`` values appear multiple times.
    """

    name: str = "duplicate_sequences"

    def run(self) -> CheckResult:
        """Query for ``content_hash`` values appearing in more than one row.

        Returns:
            ``CheckResult`` with ``status="warn"`` if duplicates are found,
            otherwise ``status="pass"``.
        """
        query = """
            SELECT
                content_hash,
                COUNT(*) AS cnt,
                array_agg(seq_id::text) AS seq_ids
            FROM antibody_sequences
            WHERE content_hash IS NOT NULL
            GROUP BY content_hash
            HAVING COUNT(*) > 1
        """
        with get_cursor(self.database_url) as cur:
            cur.execute(query)
            rows = cur.fetchall()

        if not rows:
            return self._pass("No duplicate content_hash values found.")

        details: list[dict[str, Any]] = [
            {
                "content_hash": row["content_hash"],
                "count": row["cnt"],
                "seq_ids": list(row["seq_ids"]),
            }
            for row in rows
        ]
        n = len(rows)
        return self._warn(
            affected_rows=n,
            details=details,
            message=f"{n} content_hash value(s) appear in more than one sequence row (possible duplicate ingestion).",
        )
