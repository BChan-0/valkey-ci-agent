"""Shared GitHub API retry helpers."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

from github.GithubException import GithubException

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_BASE_DELAY_SECONDS = 1.0
_MAX_DELAY_SECONDS = 8.0

# Substrings in 403 response messages that indicate rate limiting
# (as opposed to permanent permission errors).
_RATE_LIMIT_403_INDICATORS = (
    "rate limit",
    "abuse detection",
    "retry after",
    "secondary rate",
    "exceeded a secondary",
)


def _is_retryable_error(exc: Exception) -> bool:
    if not isinstance(exc, GithubException):
        return False
    if exc.status in _RETRYABLE_STATUS_CODES:
        return True
    # Only retry 403 when it's clearly rate-limiting, not a permission error.
    if exc.status == 403:
        msg = str(exc).lower()
        return any(indicator in msg for indicator in _RATE_LIMIT_403_INDICATORS)
    return False


def _delay(attempt: int) -> float:
    return random.uniform(0, min(_MAX_DELAY_SECONDS, _BASE_DELAY_SECONDS * (2 ** attempt)))


def retry_github_call(
    operation: Callable[[], _T],
    *,
    retries: int,
    description: str,
) -> _T:
    """Retry transient GitHub API failures with exponential backoff."""
    if type(retries) is not int or retries < 0:
        raise ValueError("retries must be a non-negative integer")
    attempts = retries + 1
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            last_exc = exc
            if not _is_retryable_error(exc) or attempt == attempts - 1:
                raise
            wait_seconds = _delay(attempt)
            logger.warning(
                "Retrying GitHub API call for %s after %.2fs: %s",
                description,
                wait_seconds,
                exc,
            )
            time.sleep(wait_seconds)

    # Unreachable in practice: either the loop returns (success) or the
    # last attempt re-raises. Keep the fallback so mypy is happy about
    # the function always returning _T or raising.
    assert last_exc is not None
    raise last_exc
