#!/usr/bin/env python3
"""Data Quality Monitor — CLI entrypoint.

Runs one or more groups of data quality checks against the LIMS PostgreSQL
database and produces a JSON report plus a human-readable summary.

Usage::

    python run_monitor.py [--checks all|sequence|expression|flow|integrity]
                          [--output-dir DIR] [--no-save]

Exit codes:
    0 — all checks passed or warned only
    1 — at least one check failed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from monitor.checks import CHECK_GROUPS
from monitor.config import MonitorConfig
from monitor.reporter import build_report, print_summary, save_report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list for testing. Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed ``argparse.Namespace`` with ``checks``, ``output_dir``, and
        ``no_save`` attributes.
    """
    parser = argparse.ArgumentParser(
        description="Run data quality checks on the LIMS PostgreSQL database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_monitor.py
  python run_monitor.py --checks sequence
  python run_monitor.py --checks flow --output-dir /reports
  python run_monitor.py --checks all --no-save
        """,
    )
    parser.add_argument(
        "--checks",
        choices=["all", "sequence", "expression", "flow", "integrity"],
        default="all",
        help="Which check group to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory in which to write the JSON report (default: current directory)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print summary to stdout only; do not write a JSON report file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the selected data quality checks and report results.

    Loads configuration from environment variables / ``.env``, instantiates
    each check class in the selected group, runs them sequentially, and builds
    a report.  Prints a human-readable summary to stdout.  Optionally saves a
    JSON report to disk.

    Args:
        argv: Optional argument list for testing. Defaults to ``sys.argv[1:]``.

    Returns:
        Exit code: ``0`` if all checks passed or warned, ``1`` if any check failed.
    """
    args = _parse_args(argv)

    config = MonitorConfig()  # type: ignore[call-arg]

    check_classes = CHECK_GROUPS[args.checks]

    # Mask password in URL for display
    display_url = config.database_url
    if "@" in display_url:
        # Replace the user:password portion with user:***
        at_index = display_url.rindex("@")
        scheme_end = display_url.index("://") + 3
        credentials = display_url[scheme_end:at_index]
        if ":" in credentials:
            user = credentials.split(":")[0]
            display_url = display_url[:scheme_end] + f"{user}:***" + display_url[at_index:]

    print(
        f"Running {len(check_classes)} check(s) against "
        f"{display_url.rsplit('@', 1)[-1] if '@' in display_url else display_url} …"
    )
    print()

    results = []
    for check_cls in check_classes:
        check = check_cls(config.database_url, config)
        try:
            result = check.run()
        except Exception as exc:  # noqa: BLE001
            # Treat unexpected errors as failures rather than crashing the monitor
            from monitor.checks.base import CheckResult
            from datetime import datetime

            result = CheckResult(
                check_name=check.name,
                status="fail",
                affected_rows=0,
                details=[{"error": str(exc)}],
                message=f"Check raised an unexpected exception: {exc}",
                run_at=datetime.utcnow(),
            )

        results.append(result)
        status_icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}[result.status]
        print(f"  {status_icon} {result.check_name}: {result.message}")

    print()

    report = build_report(results, config.database_url)
    print_summary(report)

    if not args.no_save:
        out_path = Path(args.output_dir) / config.report_filename
        save_report(report, out_path)
        print(f"\nReport saved → {out_path.resolve()}")

    has_failures = any(r.status == "fail" for r in results)
    return 1 if has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
