"""Alembic migration environment configuration.

Reads DATABASE_URL from the environment (or .env file) and supports both
online (live database) and offline (SQL script generation) migration modes.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# ---------------------------------------------------------------------------
# Path manipulation — ensure the project root (lab-schema-lims/) is importable
# ---------------------------------------------------------------------------

# env.py lives in lab-schema-lims/alembic/env.py, so the project root is one
# level up from this file's parent directory.
_THIS_DIR = Path(__file__).resolve().parent          # …/lab-schema-lims/alembic/
_PROJECT_ROOT = _THIS_DIR.parent                      # …/lab-schema-lims/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models import Base  # noqa: E402  (imported after sys.path manipulation)

# ---------------------------------------------------------------------------
# Alembic Config object (gives access to values in alembic.ini)
# ---------------------------------------------------------------------------

config = context.config

# If a DATABASE_URL environment variable is set, inject it into the config
# so it overrides whatever is in alembic.ini.
database_url = os.environ.get("DATABASE_URL")
if database_url:
    config.set_main_option("sqlalchemy.url", database_url)

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The MetaData object used for "autogenerate" support.
target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Offline migration mode
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    In offline mode Alembic emits SQL statements to stdout (or a file) instead
    of executing them against a live database.  Useful for review and for
    environments where direct DB access is not available at migration time.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online migration mode
# ---------------------------------------------------------------------------

def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Creates an engine, obtains a connection, and runs the migrations against
    the live database inside a transaction.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


# ---------------------------------------------------------------------------
# Entry point — choose mode based on Alembic's runtime context
# ---------------------------------------------------------------------------

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
