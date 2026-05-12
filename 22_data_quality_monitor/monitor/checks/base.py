from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class CheckResult:
    """The outcome of a single data quality check.

    Attributes:
        check_name: Machine-readable identifier for the check.
        status: One of ``"pass"``, ``"warn"``, or ``"fail"``.
        affected_rows: Number of rows that triggered the check condition.
        details: List of per-row evidence dicts (offending IDs and values).
        message: Human-readable summary of what was found.
        run_at: UTC timestamp at which the check was executed.
    """

    check_name: str
    status: Literal["pass", "warn", "fail"]
    affected_rows: int
    details: list[dict[str, Any]]
    message: str
    run_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a JSON-compatible dictionary.

        The ``details`` list is capped at 10 entries to keep reports concise.

        Returns:
            A dictionary suitable for JSON serialisation.
        """
        return {
            "check_name": self.check_name,
            "status": self.status,
            "affected_rows": self.affected_rows,
            "message": self.message,
            "run_at": self.run_at.isoformat(),
            "details": self.details[:10],
        }


class BaseCheck(ABC):
    """Abstract base class for all LIMS data quality checks.

    Subclasses must declare a ``name`` class attribute and implement ``run()``.
    Helper methods ``_pass``, ``_warn``, and ``_fail`` create correctly typed
    ``CheckResult`` instances without boilerplate.

    Attributes:
        name: Machine-readable check identifier. Override in each subclass.
        database_url: PostgreSQL connection string passed at construction.
        config: ``MonitorConfig`` instance with threshold values.
    """

    name: str = "unnamed_check"

    def __init__(self, database_url: str, config: Any) -> None:
        """Initialise the check with a database connection string and configuration.

        Args:
            database_url: PostgreSQL ``postgresql://...`` connection string.
            config: A ``MonitorConfig`` (or compatible) object exposing threshold
                attributes.
        """
        self.database_url = database_url
        self.config = config

    @abstractmethod
    def run(self) -> CheckResult:
        """Execute the check and return a ``CheckResult``.

        Subclasses query the database, evaluate results against thresholds, and
        call one of ``_pass``, ``_warn``, or ``_fail`` to construct the outcome.

        Returns:
            A ``CheckResult`` describing the outcome of the check.
        """
        ...

    def _pass(self, message: str = "Check passed.") -> CheckResult:
        """Build a passing ``CheckResult`` with zero affected rows.

        Args:
            message: Optional human-readable message. Defaults to ``"Check passed."``.

        Returns:
            A ``CheckResult`` with ``status="pass"``.
        """
        return CheckResult(
            check_name=self.name,
            status="pass",
            affected_rows=0,
            details=[],
            message=message,
        )

    def _warn(
        self,
        affected_rows: int,
        details: list[dict[str, Any]],
        message: str,
    ) -> CheckResult:
        """Build a warning ``CheckResult``.

        Args:
            affected_rows: Count of rows that triggered the warning condition.
            details: Per-row evidence dicts (capped to 10 in serialisation).
            message: Human-readable summary.

        Returns:
            A ``CheckResult`` with ``status="warn"``.
        """
        return CheckResult(
            check_name=self.name,
            status="warn",
            affected_rows=affected_rows,
            details=details,
            message=message,
        )

    def _fail(
        self,
        affected_rows: int,
        details: list[dict[str, Any]],
        message: str,
    ) -> CheckResult:
        """Build a failing ``CheckResult``.

        Args:
            affected_rows: Count of rows that triggered the failure condition.
            details: Per-row evidence dicts (capped to 10 in serialisation).
            message: Human-readable summary.

        Returns:
            A ``CheckResult`` with ``status="fail"``.
        """
        return CheckResult(
            check_name=self.name,
            status="fail",
            affected_rows=affected_rows,
            details=details,
            message=message,
        )
