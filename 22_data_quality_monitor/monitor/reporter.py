from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .checks.base import CheckResult

# Status display symbols
_STATUS_SYMBOLS: dict[str, str] = {
    "pass": "✓",  # ✓
    "warn": "⚠",  # ⚠
    "fail": "✗",  # ✗
}

_STATUS_LABELS: dict[str, str] = {
    "pass": "PASS",
    "warn": "WARN",
    "fail": "FAIL",
}


def build_report(results: list[CheckResult], database_url: str) -> dict[str, Any]:
    """Build a structured JSON-compatible report from a list of ``CheckResult`` objects.

    The report contains metadata (run timestamp, database name), a summary of
    pass/warn/fail counts, and the full serialised list of check results.

    Args:
        results: List of ``CheckResult`` instances returned by each check's
            ``run()`` method.
        database_url: The database connection string (used to extract the database
            name for the report header).

    Returns:
        A JSON-serialisable dictionary with keys ``run_at``, ``database``,
        ``summary``, and ``checks``.
    """
    db_name = database_url.rsplit("/", 1)[-1] if "/" in database_url else "unknown"

    summary: dict[str, int] = {
        "total": len(results),
        "pass": sum(1 for r in results if r.status == "pass"),
        "warn": sum(1 for r in results if r.status == "warn"),
        "fail": sum(1 for r in results if r.status == "fail"),
    }

    return {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "database": db_name,
        "summary": summary,
        "checks": [r.to_dict() for r in results],
    }


def format_human_summary(report: dict[str, Any]) -> str:
    """Format the report as a human-readable multi-line text string.

    Uses Unicode symbols to indicate each check's status:
    - ``✓ PASS`` for passing checks
    - ``⚠ WARN`` for warnings
    - ``✗ FAIL`` for failures

    Args:
        report: A report dictionary as returned by ``build_report()``.

    Returns:
        A formatted string suitable for printing to a terminal or log file.
    """
    lines: list[str] = []
    sep = "=" * 70

    lines.append(sep)
    lines.append("  DATA QUALITY MONITOR — REPORT SUMMARY")
    lines.append(sep)
    lines.append(f"  Run at  : {report.get('run_at', 'unknown')}")
    lines.append(f"  Database: {report.get('database', 'unknown')}")
    lines.append("")

    summary = report.get("summary", {})
    total = summary.get("total", 0)
    passed = summary.get("pass", 0)
    warned = summary.get("warn", 0)
    failed = summary.get("fail", 0)

    lines.append(f"  Checks run  : {total}")
    lines.append(f"  {_STATUS_SYMBOLS['pass']} PASS        : {passed}")
    lines.append(f"  {_STATUS_SYMBOLS['warn']} WARN        : {warned}")
    lines.append(f"  {_STATUS_SYMBOLS['fail']} FAIL        : {failed}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("  Check Results")
    lines.append("-" * 70)

    for check in report.get("checks", []):
        status = check.get("status", "pass")
        symbol = _STATUS_SYMBOLS.get(status, "?")
        label = _STATUS_LABELS.get(status, status.upper())
        name = check.get("check_name", "unknown")
        message = check.get("message", "")
        affected = check.get("affected_rows", 0)

        lines.append(f"  {symbol} [{label:<4}] {name}")
        lines.append(f"          {message}")
        if affected > 0:
            lines.append(f"          Affected rows: {affected}")
        lines.append("")

    lines.append(sep)

    if failed > 0:
        lines.append(
            f"  RESULT: {_STATUS_SYMBOLS['fail']} FAILED — {failed} critical issue(s) require attention."
        )
    elif warned > 0:
        lines.append(
            f"  RESULT: {_STATUS_SYMBOLS['warn']} WARNING — {warned} issue(s) require review."
        )
    else:
        lines.append(
            f"  RESULT: {_STATUS_SYMBOLS['pass']} ALL CHECKS PASSED"
        )

    lines.append(sep)

    return "\n".join(lines)


def save_report(report: dict[str, Any], output_path: Path) -> None:
    """Write the JSON report to a file, creating parent directories if needed.

    Args:
        report: The report dictionary as returned by ``build_report()``.
        output_path: Absolute or relative path to the output JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def print_summary(report: dict[str, Any]) -> None:
    """Print the human-readable report summary to stdout.

    Args:
        report: The report dictionary as returned by ``build_report()``.
    """
    print(format_human_summary(report))
