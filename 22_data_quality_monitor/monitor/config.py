from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class MonitorConfig(BaseSettings):
    """Configuration for the data quality monitor.

    All fields can be set via environment variables (or a ``.env`` file).  The
    field names map directly to upper-cased environment variable names, e.g.
    ``database_url`` → ``DATABASE_URL``.

    Attributes:
        database_url: PostgreSQL connection string for the LIMS database.
        cdr3_min_length: Minimum acceptable CDR3 amino-acid sequence length.
        cdr3_max_length: Maximum acceptable CDR3 amino-acid sequence length.
        b_cell_gate_min_pct: Minimum acceptable B-cell gate percentage in flow data.
        b_cell_gate_max_pct: Maximum acceptable B-cell gate percentage in flow data.
        gate_sum_max_pct: Maximum allowed sum of naive + memory + plasmablast gate percentages.
        yield_iqr_multiplier: IQR multiplier used for outlier detection in expression yield.
        output_dir: Directory where JSON reports are written.
        report_filename: Filename for the JSON report output.
    """

    database_url: str
    cdr3_min_length: int = 6
    cdr3_max_length: int = 30
    b_cell_gate_min_pct: float = 0.5
    b_cell_gate_max_pct: float = 60.0
    gate_sum_max_pct: float = 100.0
    yield_iqr_multiplier: float = 1.5
    output_dir: str = "."
    report_filename: str = "dqm_report.json"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
