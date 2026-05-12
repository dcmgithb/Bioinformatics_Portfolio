"""SQLAlchemy engine and session management for the bioetl-pipeline.

Provides:
- ``get_engine`` — create a synchronous SQLAlchemy engine
- ``get_session_factory`` — create a bound sessionmaker
- ``session_scope`` — context manager with automatic commit/rollback/close
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker


def get_engine(database_url: str) -> Engine:
    """Create and return a SQLAlchemy synchronous engine.

    Connection pooling is left at SQLAlchemy defaults (QueuePool with pool_size=5).
    For short-lived CLI runs, ``NullPool`` can be passed as ``poolclass`` instead.

    Args:
        database_url: Full SQLAlchemy connection URL, e.g.
            ``postgresql://user:pass@localhost:5432/bioetl_db``.

    Returns:
        A configured SQLAlchemy :class:`Engine`.
    """
    engine = create_engine(
        database_url,
        echo=False,          # set to True to log SQL for debugging
        future=True,         # use SQLAlchemy 2.0 style
    )
    return engine


def get_session_factory(engine: Engine) -> sessionmaker:
    """Create a sessionmaker bound to the given engine.

    Args:
        engine: A SQLAlchemy :class:`Engine` returned by :func:`get_engine`.

    Returns:
        A :class:`sessionmaker` that produces :class:`Session` instances.
    """
    factory: sessionmaker = sessionmaker(
        bind=engine,
        autoflush=True,
        autocommit=False,
        expire_on_commit=False,
    )
    return factory


@contextmanager
def session_scope(session_factory: sessionmaker) -> Generator[Session, None, None]:
    """Provide a transactional database session as a context manager.

    Commits the transaction on clean exit; rolls back on any exception;
    always closes the session when the ``with`` block exits.

    Args:
        session_factory: A :class:`sessionmaker` returned by
            :func:`get_session_factory`.

    Yields:
        An open :class:`Session` ready for use.

    Raises:
        Exception: Re-raises any exception that occurs inside the ``with`` block
            after rolling back the transaction.

    Example::

        factory = get_session_factory(engine)
        with session_scope(factory) as session:
            session.add(some_orm_object)
        # committed automatically
    """
    session: Session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
