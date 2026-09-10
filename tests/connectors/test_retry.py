"""Tests for shared connector retry/backoff (tenacity)."""

from __future__ import annotations

import pytest

from auditor.connectors.retry import is_transient_exception, with_retry


def test_transient_exceptions_are_retryable() -> None:
    assert is_transient_exception(ConnectionError("net"))
    assert is_transient_exception(TimeoutError("timeout"))
    assert is_transient_exception(OSError("io"))


def test_permanent_exceptions_are_not_retryable() -> None:
    assert not is_transient_exception(FileNotFoundError("missing"))
    assert not is_transient_exception(ValueError("bad"))
    assert not is_transient_exception(PermissionError("denied"))
    assert not is_transient_exception(NotADirectoryError("nodir"))


def test_with_retry_retries_transient_then_succeeds() -> None:
    calls = {"n": 0}

    @with_retry
    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3


def test_with_retry_does_not_retry_file_not_found() -> None:
    calls = {"n": 0}

    @with_retry
    def missing() -> None:
        calls["n"] += 1
        raise FileNotFoundError("gone")

    with pytest.raises(FileNotFoundError):
        missing()
    assert calls["n"] == 1


def test_with_retry_does_not_retry_value_error() -> None:
    calls = {"n": 0}

    @with_retry
    def bad() -> None:
        calls["n"] += 1
        raise ValueError("logical")

    with pytest.raises(ValueError):
        bad()
    assert calls["n"] == 1


def test_with_retry_exhausts_attempts_on_persistent_transient() -> None:
    calls = {"n": 0}

    @with_retry
    def always_transient() -> None:
        calls["n"] += 1
        raise TimeoutError("still down")

    with pytest.raises(TimeoutError):
        always_transient()
    assert calls["n"] == 3
