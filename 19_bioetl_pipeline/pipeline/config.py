"""Pipeline configuration loaded from environment variables / .env file.

Uses pydantic-settings BaseSettings so every field can be overridden via
environment variable.  No credential defaults are hardcoded here.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration for the bioetl-pipeline.

    Values are read (in priority order) from:
    1. Environment variables
    2. .env file in the current working directory
    3. The defaults defined here (non-sensitive fields only)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    database_url: str = Field(
        ...,
        description="PostgreSQL connection URL, e.g. postgresql://user:pass@host/db",
    )

    # ------------------------------------------------------------------
    # External API base URLs
    # ------------------------------------------------------------------
    oas_base_url: str = Field(
        default="http://opig.stats.ox.ac.uk/webapps/oas/api",
        description="Base URL for the Observed Antibody Space (OAS) REST API",
    )
    chembl_base_url: str = Field(
        default="https://www.ebi.ac.uk/chembl/api/data",
        description="Base URL for the ChEMBL REST API",
    )
    uniprot_base_url: str = Field(
        default="https://rest.uniprot.org/uniprotkb",
        description="Base URL for the UniProt REST API",
    )

    # ------------------------------------------------------------------
    # HTTP client settings
    # ------------------------------------------------------------------
    request_timeout: int = Field(
        default=30,
        description="HTTP request timeout in seconds",
    )

    # ------------------------------------------------------------------
    # ETL settings
    # ------------------------------------------------------------------
    batch_size: int = Field(
        default=100,
        description="Number of records per database upsert batch",
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    log_format: str = Field(
        default="json",
        description="Log format: 'json' for structured JSON output, 'console' for human-readable",
    )
