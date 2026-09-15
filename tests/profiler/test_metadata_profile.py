"""C0/C1 DatasetProfiler contract tests."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from auditor.connectors.base import CapabilityDescriptor, Connector
from auditor.evidence.models import EvidenceStatus
from auditor.evidence.store import EvidenceStore
from auditor.evidence.writers import TrustedWriter
from auditor.profiler import DatasetProfiler

SOURCE_METADATA: dict[str, Any] = {
    "source_uri": "memory://phase2-c0",
    "revision": "rev-c0",
    "provider": "in_memory",
}

CAPABILITIES = CapabilityDescriptor(
    can_stream=True,
    can_random_access=False,
    has_index_metadata=True,
    media_is_referenced_not_present=False,
    requires_auth=False,
    estimated_record_count=3,
)

FLAT_SCHEMA: dict[str, Any] = {
    "id": "str",
    "label": "str",
    "duration_s": "float",
}

WRAPPED_SCHEMA: dict[str, Any] = {
    "columns": {
        "id": "str",
        "label": "str",
    },
    "_inference": "sample_head",
    "_sample_rows": 2,
}

_METHOD_KEYS = {
    "analyzer_name",
    "analyzer_version",
    "model_name",
    "model_version",
    "parameters",
    "sampling_strategy",
    "sampling_seed",
}


def _build_capabilities(*, estimated_record_count: int | None) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        can_stream=True,
        can_random_access=False,
        has_index_metadata=True,
        media_is_referenced_not_present=False,
        requires_auth=False,
        estimated_record_count=estimated_record_count,
    )


class CountingMetadataConnector(Connector):
    """Minimal Connector ABC double that forbids record access."""

    def __init__(
        self,
        *,
        source_metadata: dict[str, Any],
        capabilities: CapabilityDescriptor,
        schema: dict[str, Any],
    ) -> None:
        self._source_metadata = deepcopy(source_metadata)
        self._capabilities = capabilities
        self._schema = deepcopy(schema)
        self.source_metadata_calls = 0
        self.capabilities_calls = 0
        self.schema_calls = 0

    def get_capabilities(self) -> CapabilityDescriptor:
        self.capabilities_calls += 1
        return self._capabilities

    def get_schema(self) -> dict[str, Any]:
        self.schema_calls += 1
        return self._schema

    def list_records(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
        raise AssertionError(
            "list_records must not be called during C0 metadata profiling"
        )

    def get_record(self, record_id: str) -> dict[str, Any]:
        raise AssertionError(
            "get_record must not be called during C0 metadata profiling"
        )

    def get_source_metadata(self) -> dict[str, Any]:
        self.source_metadata_calls += 1
        return self._source_metadata


@pytest.fixture
def store(tmp_path: Path) -> EvidenceStore:
    return EvidenceStore(db_path=tmp_path / "evidence.db")


@pytest.fixture
def writer(store: EvidenceStore) -> TrustedWriter:
    return TrustedWriter(store)


def _build_connector(
    schema: dict[str, Any],
    *,
    capabilities: CapabilityDescriptor | None = None,
) -> CountingMetadataConnector:
    return CountingMetadataConnector(
        source_metadata=deepcopy(SOURCE_METADATA),
        capabilities=capabilities if capabilities is not None else CAPABILITIES,
        schema=deepcopy(schema),
    )


def _assert_method_provenance(method: dict[str, Any]) -> None:
    assert set(method) == _METHOD_KEYS
    assert method["analyzer_name"] == "dataset_profiler"
    assert isinstance(method["analyzer_version"], str)
    assert method["analyzer_version"].strip()
    assert method["model_name"] is None
    assert method["model_version"] is None
    assert method["parameters"] == {"operation": "metadata_profiling"}
    assert method["sampling_strategy"] is None
    assert method["sampling_seed"] is None


def _assert_shared_provenance(
    fact: Any, *, source_metadata: dict[str, Any], run_id: Any
) -> None:
    assert fact.run_id == run_id
    assert fact.confidence is None
    assert fact.parent_fact_ids == []
    assert fact.source == {
        "source_uri": source_metadata["source_uri"],
        "revision": source_metadata["revision"],
        "file_path": None,
        "field": None,
        "record_id": None,
    }
    _assert_method_provenance(fact.method)


def test_profile_writes_source_capabilities_and_schema_facts(
    store: EvidenceStore, writer: TrustedWriter
) -> None:
    connector = _build_connector(FLAT_SCHEMA)
    original_metadata = deepcopy(connector._source_metadata)
    original_schema = deepcopy(connector._schema)
    original_capabilities = CAPABILITIES.model_dump(mode="json")
    run_id = uuid4()

    facts = DatasetProfiler().profile(connector, writer, run_id)

    assert len(facts) == 4
    assert [fact.value["metric"] for fact in facts] == [
        "profiler.source_metadata",
        "profiler.capabilities",
        "profiler.schema",
        "profiler.record_count.indexed",
    ]
    assert all(fact.status is EvidenceStatus.OBSERVED for fact in facts)

    source_fact, capabilities_fact, schema_fact, count_fact = facts
    assert source_fact.value == {
        "metric": "profiler.source_metadata",
        "value": original_metadata,
    }
    assert capabilities_fact.value == {
        "metric": "profiler.capabilities",
        "value": original_capabilities,
    }
    assert set(capabilities_fact.value["value"]) == {
        "can_stream",
        "can_random_access",
        "has_index_metadata",
        "media_is_referenced_not_present",
        "requires_auth",
        "estimated_record_count",
    }
    assert schema_fact.value == {
        "metric": "profiler.schema",
        "value": original_schema,
    }
    assert count_fact.value == {
        "metric": "profiler.record_count.indexed",
        "value": 3,
    }
    assert count_fact.claim == "Connector-reported estimated record count"

    assert connector._source_metadata == original_metadata
    assert connector._schema == original_schema

    for fact in facts:
        assert isinstance(fact.claim, str) and fact.claim.strip()
        _assert_shared_provenance(
            fact, source_metadata=original_metadata, run_id=run_id
        )

    assert "connector-reported schema" in schema_fact.claim.lower()


def test_profile_never_reads_records_for_metadata_checkpoint(
    writer: TrustedWriter,
) -> None:
    connector = _build_connector(FLAT_SCHEMA)
    run_id = uuid4()

    DatasetProfiler().profile(connector, writer, run_id)

    assert connector.source_metadata_calls == 1
    assert connector.capabilities_calls == 1
    assert connector.schema_calls == 1


@pytest.mark.parametrize(
    "schema",
    [FLAT_SCHEMA, WRAPPED_SCHEMA],
    ids=["flat", "wrapped"],
)
def test_profile_preserves_connector_schema_without_normalizing(
    writer: TrustedWriter, schema: dict[str, Any]
) -> None:
    connector = _build_connector(schema)
    supplied = deepcopy(schema)
    run_id = uuid4()

    facts = DatasetProfiler().profile(connector, writer, run_id)

    schema_fact = facts[2]
    assert schema_fact.value == {
        "metric": "profiler.schema",
        "value": supplied,
    }
    assert connector._schema == supplied


def test_profile_uses_caller_run_id_and_returns_written_facts(
    store: EvidenceStore, writer: TrustedWriter
) -> None:
    connector = _build_connector(FLAT_SCHEMA)
    run_id = uuid4()

    facts = DatasetProfiler().profile(connector, writer, run_id)

    assert len(facts) == 4
    assert len({fact.id for fact in facts}) == 4
    assert all(fact.run_id == run_id for fact in facts)
    assert all(fact.confidence is None for fact in facts)
    assert all(fact.parent_fact_ids == [] for fact in facts)

    stored = store.get_by_run(run_id)
    assert len(stored) == 4
    assert {fact.id for fact in stored} == {fact.id for fact in facts}


def test_profile_indexed_record_count_zero_is_observed(
    store: EvidenceStore, writer: TrustedWriter
) -> None:
    connector = _build_connector(
        FLAT_SCHEMA,
        capabilities=_build_capabilities(estimated_record_count=0),
    )
    original_metadata = deepcopy(connector._source_metadata)
    run_id = uuid4()

    facts = DatasetProfiler().profile(connector, writer, run_id)

    count_fact = facts[3]
    assert count_fact.status is EvidenceStatus.OBSERVED
    assert count_fact.claim == "Connector-reported estimated record count"
    assert count_fact.value == {
        "metric": "profiler.record_count.indexed",
        "value": 0,
    }
    assert count_fact.value["value"] == 0
    assert count_fact.value["value"] is not None
    _assert_shared_provenance(
        count_fact, source_metadata=original_metadata, run_id=run_id
    )
    assert connector.source_metadata_calls == 1
    assert connector.capabilities_calls == 1
    assert connector.schema_calls == 1


def test_profile_indexed_record_count_none_is_unknown(
    store: EvidenceStore, writer: TrustedWriter
) -> None:
    connector = _build_connector(
        FLAT_SCHEMA,
        capabilities=_build_capabilities(estimated_record_count=None),
    )
    original_metadata = deepcopy(connector._source_metadata)
    run_id = uuid4()

    facts = DatasetProfiler().profile(connector, writer, run_id)

    count_fact = facts[3]
    assert count_fact.status is EvidenceStatus.UNKNOWN
    assert count_fact.claim == "Connector did not report an estimated record count"
    assert count_fact.value == {
        "metric": "profiler.record_count.indexed",
        "value": None,
        "reason": "metadata_unavailable",
        "explanation": "Connector did not provide an estimated record count.",
    }
    _assert_shared_provenance(
        count_fact, source_metadata=original_metadata, run_id=run_id
    )

    stored = store.get_by_run(run_id)
    assert count_fact.id in {fact.id for fact in stored}
    assert connector.source_metadata_calls == 1
    assert connector.capabilities_calls == 1
    assert connector.schema_calls == 1
