"""Abstract base class for all bioetl data-source extractors.

Every concrete extractor must:
1. Set the ``source_name`` class attribute.
2. Implement ``extract()`` returning a list of raw dicts.
3. Use ``_get()`` for all HTTP requests (handles retries and logging).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Union

import requests
import requests.exceptions

from pipeline.logger import get_logger

_log = get_logger(__name__)

# Maximum number of HTTP retry attempts for transient errors
_MAX_RETRIES: int = 3
# Base backoff delay in seconds (exponential: 1s, 2s, 4s)
_BACKOFF_BASE: float = 1.0


class BaseExtractor(ABC):
    """Abstract base class for all data-source extractors.

    Subclasses must:
    - Set ``source_name`` class attribute (used in logging).
    - Implement the :meth:`extract` method.

    Provides a ``_get`` helper for resilient HTTP GET requests with
    exponential backoff and structured error logging.
    """

    source_name: str = "unknown"

    def __init__(self, base_url: str, timeout: int = 30) -> None:
        """Initialise the extractor with a base URL and HTTP timeout.

        Args:
            base_url: Root URL of the data source API (no trailing slash needed).
            timeout: HTTP request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "bioetl-pipeline/1.0 (bioinformatics-portfolio)",
            }
        )

    @abstractmethod
    def extract(self) -> list[dict[str, Any]]:
        """Extract raw records from the data source.

        Returns:
            A list of raw dictionaries, one per record. Keys are source-specific.
            On partial failure, returns whatever records were successfully retrieved.
        """
        ...

    def _get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
    ) -> Union[dict[str, Any], list[Any]]:
        """Perform an HTTP GET request with exponential-backoff retry.

        Retries up to ``_MAX_RETRIES`` times on connection errors, timeouts,
        and 5xx server errors. Returns an empty dict on final failure so that
        callers can gracefully fall back to synthetic data.

        Args:
            url: Full URL to request (may include query parameters in the string).
            params: Optional dict of query parameters to append.

        Returns:
            Parsed JSON response as a dict or list, or ``{}`` on failure.
        """
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._session.get(url, params=params, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json()
                if response.status_code in (429, 500, 502, 503, 504):
                    # Retriable server-side errors
                    _log.warning(
                        "http_retriable_error",
                        source=self.source_name,
                        url=url,
                        status_code=response.status_code,
                        attempt=attempt,
                    )
                else:
                    # Non-retriable client error (4xx etc.)
                    _log.error(
                        "http_client_error",
                        source=self.source_name,
                        url=url,
                        status_code=response.status_code,
                    )
                    return {}
            except requests.exceptions.Timeout:
                _log.warning(
                    "http_timeout",
                    source=self.source_name,
                    url=url,
                    attempt=attempt,
                )
            except requests.exceptions.ConnectionError as exc:
                _log.warning(
                    "http_connection_error",
                    source=self.source_name,
                    url=url,
                    error=str(exc),
                    attempt=attempt,
                )
            except requests.exceptions.RequestException as exc:
                _log.error(
                    "http_unexpected_error",
                    source=self.source_name,
                    url=url,
                    error=str(exc),
                )
                return {}

            if attempt < _MAX_RETRIES:
                backoff = _BACKOFF_BASE * (2 ** (attempt - 1))
                _log.info(
                    "http_retry_backoff",
                    source=self.source_name,
                    backoff_seconds=backoff,
                    attempt=attempt,
                )
                time.sleep(backoff)

        _log.error(
            "http_max_retries_exceeded",
            source=self.source_name,
            url=url,
            max_retries=_MAX_RETRIES,
        )
        return {}

    def close(self) -> None:
        """Close the underlying requests Session and release connections."""
        self._session.close()
