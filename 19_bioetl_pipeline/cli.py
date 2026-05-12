"""Command-line interface for the bioetl-pipeline.

Commands:
    run      Extract, transform, validate, and load data from one or all sources.
    init-db  Create ingestion tables in the target database.
    status   Print record counts for all ingestion tables.

Usage::

    python cli.py run --source oas --db-url postgresql://...
    python cli.py run --source all --log-format console
    python cli.py init-db --db-url postgresql://...
    python cli.py status
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import click

# Ensure the repo root is on sys.path so ``pipeline`` is importable when
# running as ``python cli.py`` from the project directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_settings(db_url: str | None, log_format: str = "console") -> Any:
    """Construct a Settings object, injecting db_url and log_format overrides.

    Args:
        db_url: Optional database URL override; falls back to DATABASE_URL env var.
        log_format: Logging format override (``json`` or ``console``).

    Returns:
        Populated :class:`pipeline.config.Settings` instance.

    Raises:
        SystemExit: If no database URL can be determined.
    """
    from pipeline.config import Settings

    url = db_url or os.environ.get("DATABASE_URL", "")
    if not url:
        click.echo(
            "ERROR: No database URL. Pass --db-url or set DATABASE_URL.", err=True
        )
        sys.exit(1)

    # Temporarily inject into environment so pydantic-settings reads it
    os.environ["DATABASE_URL"] = url
    os.environ["LOG_FORMAT"] = log_format
    return Settings()


def _run_source(
    source_name: str,
    settings: Any,
) -> dict[str, int]:
    """Run the full ETL pipeline for a single source.

    Performs: extract → transform → validate → load.

    Args:
        source_name: One of ``"oas"``, ``"chembl"``, ``"uniprot"``.
        settings: Application settings.

    Returns:
        Dict with keys: ``extracted``, ``passed``, ``inserted``, ``skipped``.
    """
    from pipeline.database import get_engine, get_session_factory, session_scope
    from pipeline.extractors.chembl import ChEMBLExtractor
    from pipeline.extractors.oas import OASExtractor
    from pipeline.extractors.uniprot import UniProtExtractor
    from pipeline.loaders.db import load_dataframe
    from pipeline.models import (
        IngestedAntibodySequence,
        IngestedBCellMarker,
        IngestedTherapeutic,
    )
    from pipeline.transformers.antibody import (
        transform_chembl_records,
        transform_oas_records,
        transform_uniprot_records,
    )
    from pipeline.transformers.validator import validate_dataframe

    engine = get_engine(settings.database_url)
    session_factory = get_session_factory(engine)

    source_upper = source_name.upper()

    if source_name == "oas":
        extractor = OASExtractor(
            base_url=settings.oas_base_url, timeout=settings.request_timeout
        )
        raw_records = extractor.extract()
        extractor.close()
        df_raw = transform_oas_records(raw_records)
        df_clean, report = validate_dataframe(df_raw, source="OAS")
        model = IngestedAntibodySequence

    elif source_name == "chembl":
        extractor = ChEMBLExtractor(
            base_url=settings.chembl_base_url, timeout=settings.request_timeout
        )
        raw_records = extractor.extract()
        extractor.close()
        df_raw = transform_chembl_records(raw_records)
        df_clean, report = validate_dataframe(df_raw, source="ChEMBL")
        model = IngestedTherapeutic

    elif source_name == "uniprot":
        extractor = UniProtExtractor(
            base_url=settings.uniprot_base_url, timeout=settings.request_timeout
        )
        raw_records = extractor.extract()
        extractor.close()
        df_raw = transform_uniprot_records(raw_records)
        df_clean, report = validate_dataframe(df_raw, source="UniProt")
        model = IngestedBCellMarker

    else:
        raise ValueError(f"Unknown source: {source_name!r}")

    if df_clean.empty:
        return {
            "extracted": len(raw_records),
            "passed": 0,
            "inserted": 0,
            "skipped": 0,
        }

    with session_scope(session_factory) as session:
        n_inserted, n_skipped = load_dataframe(
            session, model, df_clean, batch_size=settings.batch_size
        )

    return {
        "extracted": report.total_records,
        "passed": report.passed,
        "inserted": n_inserted,
        "skipped": n_skipped,
    }


def _print_summary_table(results: dict[str, dict[str, int]]) -> None:
    """Print a formatted summary table of ETL results to stdout.

    Args:
        results: Dict mapping source name → result dict with keys
            ``extracted``, ``passed``, ``inserted``, ``skipped``.
    """
    header = f"{'Source':<12} | {'Extracted':>9} | {'Passed':>6} | {'Inserted':>8} | {'Skipped':>7}"
    separator = "-" * len(header)
    click.echo(separator)
    click.echo(header)
    click.echo(separator)
    for source, stats in results.items():
        row = (
            f"{source:<12} | "
            f"{stats.get('extracted', 0):>9} | "
            f"{stats.get('passed', 0):>6} | "
            f"{stats.get('inserted', 0):>8} | "
            f"{stats.get('skipped', 0):>7}"
        )
        click.echo(row)
    click.echo(separator)


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """bioetl-pipeline: Extract, transform, and load bioinformatics data."""
    pass


@cli.command("run")
@click.option(
    "--source",
    type=click.Choice(["oas", "chembl", "uniprot", "all"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Data source to ingest. 'all' runs all sources concurrently.",
)
@click.option(
    "--db-url",
    default=None,
    envvar="DATABASE_URL",
    help="PostgreSQL connection URL. Defaults to DATABASE_URL env var.",
)
@click.option(
    "--log-format",
    type=click.Choice(["json", "console"], case_sensitive=False),
    default="console",
    show_default=True,
    help="Logging output format.",
)
def run_command(source: str, db_url: Optional[str], log_format: str) -> None:
    """Run the ETL pipeline for one or all data sources.

    Uses ThreadPoolExecutor for concurrent extraction when --source=all.
    Prints a summary table on completion.
    """
    settings = _build_settings(db_url, log_format)

    from pipeline.logger import configure_logging

    configure_logging(settings)

    sources = ["oas", "chembl", "uniprot"] if source == "all" else [source.lower()]
    results: dict[str, dict[str, int]] = {}

    if len(sources) == 1:
        # Single source — run directly
        src = sources[0]
        click.echo(f"Running ETL for source: {src.upper()} ...")
        try:
            results[src.upper()] = _run_source(src, settings)
        except Exception as exc:
            click.echo(f"ERROR running {src.upper()}: {exc}", err=True)
            results[src.upper()] = {"extracted": 0, "passed": 0, "inserted": 0, "skipped": 0}
    else:
        # Multiple sources — run concurrently
        click.echo("Running ETL for all sources concurrently ...")
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_source = {
                executor.submit(_run_source, src, settings): src for src in sources
            }
            for future in as_completed(future_to_source):
                src = future_to_source[future]
                try:
                    stats = future.result()
                    results[src.upper()] = stats
                    click.echo(f"  {src.upper()} complete: {stats}")
                except Exception as exc:
                    click.echo(f"  ERROR in {src.upper()}: {exc}", err=True)
                    results[src.upper()] = {
                        "extracted": 0,
                        "passed": 0,
                        "inserted": 0,
                        "skipped": 0,
                    }

    click.echo("")
    _print_summary_table(results)


@cli.command("init-db")
@click.option(
    "--db-url",
    default=None,
    envvar="DATABASE_URL",
    help="PostgreSQL connection URL. Defaults to DATABASE_URL env var.",
)
def init_db_command(db_url: Optional[str]) -> None:
    """Create all ingestion tables in the target database.

    Equivalent to ``CREATE TABLE IF NOT EXISTS`` for all three ingestion
    tables. Safe to run multiple times (idempotent).
    """
    settings = _build_settings(db_url)
    from pipeline.database import get_engine
    from pipeline.models import Base

    engine = get_engine(settings.database_url)
    Base.metadata.create_all(engine)
    click.echo("Ingestion tables created (or already exist):")
    for table_name in Base.metadata.tables:
        click.echo(f"  - {table_name}")


@cli.command("status")
@click.option(
    "--db-url",
    default=None,
    envvar="DATABASE_URL",
    help="PostgreSQL connection URL. Defaults to DATABASE_URL env var.",
)
def status_command(db_url: Optional[str]) -> None:
    """Print record counts for all ingestion tables.

    Queries each table and reports how many rows have been ingested.
    """
    settings = _build_settings(db_url)
    from sqlalchemy import text

    from pipeline.database import get_engine, get_session_factory, session_scope
    from pipeline.models import (
        Base,
        IngestedAntibodySequence,
        IngestedBCellMarker,
        IngestedTherapeutic,
    )

    engine = get_engine(settings.database_url)
    session_factory = get_session_factory(engine)

    table_models = [
        ("ingested_antibody_sequences", IngestedAntibodySequence),
        ("ingested_therapeutics", IngestedTherapeutic),
        ("ingested_b_cell_markers", IngestedBCellMarker),
    ]

    click.echo(f"{'Table':<35} | {'Row Count':>10}")
    click.echo("-" * 50)

    with session_scope(session_factory) as session:
        for table_name, model in table_models:
            try:
                count = session.query(model).count()
                click.echo(f"{table_name:<35} | {count:>10}")
            except Exception as exc:
                click.echo(f"{table_name:<35} | ERROR: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
