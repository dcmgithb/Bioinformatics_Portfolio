from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import psycopg2
import psycopg2.extensions
from psycopg2.extras import RealDictCursor


def get_connection(database_url: str) -> psycopg2.extensions.connection:
    """Open and return a psycopg2 connection to the given PostgreSQL database.

    The caller is responsible for closing the connection when finished.

    Args:
        database_url: A ``postgresql://user:password@host:port/dbname`` connection
            string.

    Returns:
        An open ``psycopg2`` connection object.

    Raises:
        psycopg2.OperationalError: If the database cannot be reached or the
            credentials are invalid.
    """
    return psycopg2.connect(database_url, cursor_factory=RealDictCursor)


@contextmanager
def get_cursor(database_url: str) -> Generator[RealDictCursor, None, None]:
    """Context manager that yields a ``RealDictCursor`` for the given database.

    Commits the transaction on clean exit; rolls back and re-raises on any
    exception.  Ensures that both the cursor and the connection are always
    closed regardless of outcome.

    Args:
        database_url: A ``postgresql://user:password@host:port/dbname`` connection
            string.

    Yields:
        A ``RealDictCursor`` ready for use.

    Raises:
        psycopg2.Error: Any database error encountered during execution, after
            rolling back the transaction.
    """
    conn: psycopg2.extensions.connection = get_connection(database_url)
    cursor: RealDictCursor = conn.cursor(cursor_factory=RealDictCursor)  # type: ignore[call-overload]
    try:
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()
