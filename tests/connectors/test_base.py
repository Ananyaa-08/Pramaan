"""Tests for the connector interface and InMemoryConnector (design §4.1)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from auditor.connectors.base import CapabilityDescriptor, Connector
from auditor.connectors.fake import InMemoryConnector

FIXTURE_RECORDS: list[dict[str, Any]] = [
    {"id": "r1", "label": "hello", "duration_s": 1.2},
    {"id": "r2", "label": "world", "duration_s": 3.4},
    {"id": "r3", "label": "sign", "duration_s": 2.0},
]

FIXTURE_SCHEMA: dict[str, Any] = {
    "id": "str",
    "label": "str",
    "duration_s": "float",
}

FIXTURE_SOURCE_METADATA: dict[str, Any] = {
    "source_uri": "memory://fixture",
    "revision": "test-rev-1",
    "provider": "in_memory",
}


def _default_capabilities(**overrides: Any) -> CapabilityDescriptor:
    fields = {
        "can_stream": True,
        "can_random_access": True,
        "has_index_metadata": True,
        "media_is_referenced_not_present": False,
        "requires_auth": False,
        "estimated_record_count": len(FIXTURE_RECORDS),
    }
    fields.update(overrides)
    return CapabilityDescriptor(**fields)


# ---------------------------------------------------------------------------
# CapabilityDescriptor
# ---------------------------------------------------------------------------


def test_capability_descriptor_round_trip() -> None:
    caps = CapabilityDescriptor(
        can_stream=True,
        can_random_access=False,
        has_index_metadata=True,
        media_is_referenced_not_present=True,
        requires_auth=True,
        estimated_record_count=118_000,
    )
    assert caps.can_stream is True
    assert caps.can_random_access is False
    assert caps.has_index_metadata is True
    assert caps.media_is_referenced_not_present is True
    assert caps.requires_auth is True
    assert caps.estimated_record_count == 118_000


def test_capability_descriptor_is_frozen() -> None:
    caps = _default_capabilities()
    with pytest.raises(Exception):
        caps.can_stream = False  # type: ignore[misc]


def test_capability_descriptor_allows_none_record_count() -> None:
    caps = _default_capabilities(estimated_record_count=None)
    assert caps.estimated_record_count is None


# ---------------------------------------------------------------------------
# InMemoryConnector satisfies Connector ABC
# ---------------------------------------------------------------------------


def test_in_memory_connector_is_connector_subclass() -> None:
    connector = InMemoryConnector(
        records=FIXTURE_RECORDS,
        schema=FIXTURE_SCHEMA,
        source_metadata=FIXTURE_SOURCE_METADATA,
        capabilities=_default_capabilities(),
    )
    assert isinstance(connector, Connector)


# ---------------------------------------------------------------------------
# Schema and source metadata
# ---------------------------------------------------------------------------


def test_get_schema_and_source_metadata_shape() -> None:
    connector = InMemoryConnector(
        records=FIXTURE_RECORDS,
        schema=FIXTURE_SCHEMA,
        source_metadata=FIXTURE_SOURCE_METADATA,
        capabilities=_default_capabilities(),
    )
    assert connector.get_schema() == FIXTURE_SCHEMA
    meta = connector.get_source_metadata()
    assert meta["source_uri"] == "memory://fixture"
    assert meta["revision"] == "test-rev-1"
    assert meta["provider"] == "in_memory"
    assert connector.get_capabilities().estimated_record_count == 3


# ---------------------------------------------------------------------------
# list_records laziness
# ---------------------------------------------------------------------------


def test_list_records_returns_iterator_not_list() -> None:
    connector = InMemoryConnector(
        records=FIXTURE_RECORDS,
        schema=FIXTURE_SCHEMA,
        source_metadata=FIXTURE_SOURCE_METADATA,
        capabilities=_default_capabilities(),
    )
    result = connector.list_records()
    assert isinstance(result, Iterator)
    assert not isinstance(result, list)


def test_list_records_is_lazy_with_production_counter() -> None:
    """Consuming one item must not force materialization of the rest."""
    produced = {"count": 0}

    def counting_source() -> Iterator[dict[str, Any]]:
        for record in FIXTURE_RECORDS:
            produced["count"] += 1
            yield record

    connector = InMemoryConnector(
        records=counting_source(),
        schema=FIXTURE_SCHEMA,
        source_metadata=FIXTURE_SOURCE_METADATA,
        capabilities=_default_capabilities(estimated_record_count=None),
    )

    assert produced["count"] == 0
    iterator = connector.list_records()
    assert produced["count"] == 0

    first = next(iterator)
    assert first["id"] == "r1"
    assert produced["count"] == 1

    second = next(iterator)
    assert second["id"] == "r2"
    assert produced["count"] == 2


def test_list_records_respects_limit() -> None:
    connector = InMemoryConnector(
        records=FIXTURE_RECORDS,
        schema=FIXTURE_SCHEMA,
        source_metadata=FIXTURE_SOURCE_METADATA,
        capabilities=_default_capabilities(),
    )
    records = list(connector.list_records(limit=2))
    assert len(records) == 2
    assert [r["id"] for r in records] == ["r1", "r2"]


# ---------------------------------------------------------------------------
# get_record / can_random_access
# ---------------------------------------------------------------------------


def test_get_record_works_when_random_access_enabled() -> None:
    connector = InMemoryConnector(
        records=FIXTURE_RECORDS,
        schema=FIXTURE_SCHEMA,
        source_metadata=FIXTURE_SOURCE_METADATA,
        capabilities=_default_capabilities(can_random_access=True),
        id_field="id",
    )
    record = connector.get_record("r2")
    assert record["label"] == "world"
    assert record["duration_s"] == 3.4


def test_get_record_raises_when_random_access_disabled() -> None:
    connector = InMemoryConnector(
        records=FIXTURE_RECORDS,
        schema=FIXTURE_SCHEMA,
        source_metadata=FIXTURE_SOURCE_METADATA,
        capabilities=_default_capabilities(can_random_access=False),
        id_field="id",
    )
    with pytest.raises(NotImplementedError):
        connector.get_record("r1")


def test_get_record_missing_id_raises_key_error() -> None:
    connector = InMemoryConnector(
        records=FIXTURE_RECORDS,
        schema=FIXTURE_SCHEMA,
        source_metadata=FIXTURE_SOURCE_METADATA,
        capabilities=_default_capabilities(can_random_access=True),
        id_field="id",
    )
    with pytest.raises(KeyError):
        connector.get_record("missing")
