"""Tests for structured logging scaffolding."""

from __future__ import annotations

import json
import logging
from io import StringIO

import structlog

from auditor.logging_config import configure_logging, get_logger


def _capture_json_logs() -> StringIO:
    """Reconfigure structlog to write JSON into an in-memory stream."""
    configure_logging(log_level="DEBUG")

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))

    # Replace handlers after configure_logging (which binds stderr via basicConfig).
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    return stream


def test_get_logger_returns_bound_logger() -> None:
    configure_logging()
    logger = get_logger("auditor.test")
    assert logger is not None
    assert hasattr(logger, "info")
    assert hasattr(logger, "bind")


def test_logger_emits_structured_json() -> None:
    stream = _capture_json_logs()
    logger = get_logger("auditor.test.json")

    logger.info("hello_audit", component="scaffolding", count=3)

    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert lines, "expected at least one log line"
    payload = json.loads(lines[-1])

    assert payload["event"] == "hello_audit"
    assert payload["component"] == "scaffolding"
    assert payload["count"] == 3
    assert "level" in payload


def test_logger_supports_run_id_binding() -> None:
    stream = _capture_json_logs()
    logger = get_logger("auditor.test.run_id").bind(run_id="run-abc-123")

    logger.info("with_run_id", step="config")

    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert lines, "expected at least one log line"
    payload = json.loads(lines[-1])

    assert payload["run_id"] == "run-abc-123"
    assert payload["event"] == "with_run_id"
    assert payload["step"] == "config"


def test_contextvars_run_id_appears_in_output() -> None:
    stream = _capture_json_logs()
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(run_id="ctx-run-456")

    try:
        logger = get_logger("auditor.test.context")
        logger.warning("from_context")
        lines = [line for line in stream.getvalue().splitlines() if line.strip()]
        payload = json.loads(lines[-1])
        assert payload["run_id"] == "ctx-run-456"
    finally:
        structlog.contextvars.clear_contextvars()
