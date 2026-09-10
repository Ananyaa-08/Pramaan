"""Shared retry/backoff for connectors (tenacity)."""

from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec, TypeVar

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

P = ParamSpec("P")
T = TypeVar("T")

# Permanent / logical failures must fail immediately (design: typed failure,
# never fabricate). FileNotFoundError is an OSError subclass — exclude it.
_PERMANENT_EXCEPTIONS = (
    FileNotFoundError,
    NotADirectoryError,
    IsADirectoryError,
    PermissionError,
    ValueError,
    TypeError,
)


def is_transient_exception(exc: BaseException) -> bool:
    """Return True only for failures that may succeed on retry."""
    if isinstance(exc, _PERMANENT_EXCEPTIONS):
        return False
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))


def _before_sleep_noop(retry_state: RetryCallState) -> None:
    return None


def with_retry(
    fn: Callable[P, T] | None = None,
    *,
    max_attempts: int = 3,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator: exponential backoff, max_attempts tries, transient errors only.

    Usage::

        @with_retry
        def open_remote(...): ...

        @with_retry(max_attempts=5)
        def fetch(...): ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        return retry(
            reraise=True,
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=0.05, min=0.05, max=1.0),
            retry=retry_if_exception(is_transient_exception),
            before_sleep=_before_sleep_noop,
        )(func)

    if fn is not None:
        return decorator(fn)
    return decorator
