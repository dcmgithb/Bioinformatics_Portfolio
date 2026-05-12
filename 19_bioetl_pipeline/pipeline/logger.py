"""Structured logging configuration using structlog.

Supports two output modes:
- ``json``: machine-readable JSON (production / log aggregation)
- ``console``: human-readable coloured output (development)

Usage::

    from pipeline.config import Settings
    from pipeline.logger import configure_logging, get_logger

    settings = Settings()
    configure_logging(settings)
    log = get_logger(__name__)
    log.info("pipeline_started", source="OAS")
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from pipeline.config import Settings


def _add_datetime(
    logger: Any, method: str, event_dict: EventDict
) -> EventDict:
    """Inject an ISO-8601 timestamp into every log event."""
    import datetime

    event_dict["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    return event_dict


def configure_logging(settings: Settings) -> None:
    """Configure structlog based on the provided Settings object.

    Must be called once at application startup before any loggers are used.

    Args:
        settings: Application settings, used to read ``log_level`` and ``log_format``.
    """
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configure the standard library root logger so that third-party libraries
    # (SQLAlchemy, requests, etc.) also emit structured logs.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    shared_processors: list[Processor] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _add_datetime,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.log_format == "json":
        final_processor: Processor = structlog.processors.JSONRenderer()
    else:
        final_processor = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [final_processor],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a bound structlog logger namespaced to ``name``.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A structlog BoundLogger instance with ``name`` bound as context.
    """
    return structlog.get_logger(name)
