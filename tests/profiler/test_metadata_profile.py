"""C0/C1/C2/C3/C4 DatasetProfiler contract tests."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from auditor.connectors.base import CapabilityDescriptor, Connector
from auditor.evidence.models import EvidenceStatus, Fact
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

WRAPPED_FULL_SCALAR_SCHEMA: dict[str, Any] = {
    "columns": {
        "id": "string",
        "score": "int64",
        "flag": "bool",
        "when": "timestamp",
        "note": "str | null",
        "unsigned": "uint32",
        "ratio": "double",
        "amount": "decimal128",
        "day": "date32",
        "nullable_number": "int64 | null",
    },
    "_inference": "full",
}

REPOSITORY_FILE_SCHEMA: dict[str, Any] = {
    "columns": {
        "path": "str",
        "sha": "str",
        "size": "int",
        "kind": "str",
        "download_url": "str | null",
        "revision": "str",
        "image": "str",
        "audio_path": "str",
    },
    "_inference": "full",
    "record_model": "repository_file",
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

_ESTIMATED_WORK_EMPTY: list[dict[str, Any]] = []

_ESTIMATED_WORK_RECORDS_TO_SCAN_NONE = [
    {"unit": "RECORDS_TO_SCAN", "magnitude": None},
]

_ESTIMATED_WORK_FILES_TO_OPEN_NONE = [
    {"unit": "FILES_TO_OPEN", "magnitude": None},
]


def _unknown_indexed_value() -> dict[str, Any]:
    return {
        "metric": "profiler.record_count.indexed",
        "value": None,
        "reason": "metadata_unavailable",
        "explanation": "Connector did not provide an estimated record count.",
        "estimated_work": _ESTIMATED_WORK_RECORDS_TO_SCAN_NONE,
    }


def _unknown_exact_value(*, estimated_record_count: int | None) -> dict[str, Any]:
    return {
        "metric": "profiler.record_count.exact",
        "value": None,
        "reason": "scan_not_authorized",
        "explanation": (
            "Exact record count was not computed because "
            "a full record scan was not authorized."
        ),
        "estimated_work": [
            {"unit": "RECORDS_TO_SCAN", "magnitude": estimated_record_count},
        ],
    }


_SCHEMA_INCOMPLETE_EXPLANATION = (
    "Connector schema is incomplete or unavailable for modality classification."
)
_MEDIA_REFERENCED_EXPLANATION = (
    "Dataset content is referenced but not present, and the schema does not "
    "declare a content modality."
)
_MODALITY_AMBIGUOUS_EXPLANATION = (
    "Connector schema types do not establish a supported modality."
)


def _build_capabilities(
    *,
    estimated_record_count: int | None = 3,
    has_index_metadata: bool = True,
    media_is_referenced_not_present: bool = False,
    can_stream: bool = True,
    can_random_access: bool = False,
    requires_auth: bool = False,
) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        can_stream=can_stream,
        can_random_access=can_random_access,
        has_index_metadata=has_index_metadata,
        media_is_referenced_not_present=media_is_referenced_not_present,
        requires_auth=requires_auth,
        estimated_record_count=estimated_record_count,
    )


class CountingMetadataConnector(Connector):
    """Minimal Connector ABC double with optional authorized-scan records."""

    def __init__(
        self,
        *,
        source_metadata: dict[str, Any],
        capabilities: CapabilityDescriptor,
        schema: dict[str, Any],
        records: list[dict[str, Any]] | None = None,
    ) -> None:
        self._source_metadata = deepcopy(source_metadata)
        self._capabilities = capabilities
        self._schema = deepcopy(schema)
        self._records = None if records is None else tuple(deepcopy(records))
        self.source_metadata_calls = 0
        self.capabilities_calls = 0
        self.schema_calls = 0
        self.list_records_calls = 0
        self.records_produced = 0
        self.get_record_calls = 0

    def get_capabilities(self) -> CapabilityDescriptor:
        self.capabilities_calls += 1
        return self._capabilities

    def get_schema(self) -> dict[str, Any]:
        self.schema_calls += 1
        return self._schema

    def list_records(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
        self.list_records_calls += 1
        if self._records is None:
            raise AssertionError(
                "list_records must not be called during metadata profiling "
                "without an authorized scan"
            )
        count = 0
        for record in self._records:
            if limit is not None and count >= limit:
                return
            self.records_produced += 1
            yield deepcopy(record)
            count += 1

    def get_record(self, record_id: str) -> dict[str, Any]:
        self.get_record_calls += 1
        raise AssertionError("get_record must not be called during metadata profiling")

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
    records: list[dict[str, Any]] | None = None,
) -> CountingMetadataConnector:
    return CountingMetadataConnector(
        source_metadata=deepcopy(SOURCE_METADATA),
        capabilities=capabilities if capabilities is not None else CAPABILITIES,
        schema=deepcopy(schema),
        records=records,
    )


def _assert_metadata_method(method: dict[str, Any]) -> None:
    assert set(method) == _METHOD_KEYS
    assert method["analyzer_name"] == "dataset_profiler"
    assert isinstance(method["analyzer_version"], str)
    assert method["analyzer_version"].strip()
    assert method["model_name"] is None
    assert method["model_version"] is None
    assert method["parameters"] == {"operation": "metadata_profiling"}
    assert method["sampling_strategy"] is None
    assert method["sampling_seed"] is None


def _assert_exact_count_method(
    method: dict[str, Any], *, scan_max_records: int
) -> None:
    assert set(method) == _METHOD_KEYS
    assert method["analyzer_name"] == "dataset_profiler"
    assert isinstance(method["analyzer_version"], str)
    assert method["analyzer_version"].strip()
    assert method["model_name"] is None
    assert method["model_version"] is None
    assert method["parameters"] == {
        "operation": "exact_record_count",
        "scan_max_records": scan_max_records,
    }
    assert method["sampling_strategy"] is None
    assert method["sampling_seed"] is None


def _assert_modality_method(method: dict[str, Any]) -> None:
    assert set(method) == _METHOD_KEYS
    assert method["analyzer_name"] == "dataset_profiler"
    assert isinstance(method["analyzer_version"], str)
    assert method["analyzer_version"].strip()
    assert method["model_name"] is None
    assert method["model_version"] is None
    assert method["parameters"] == {"operation": "modality_classification"}
    assert method["sampling_strategy"] is None
    assert method["sampling_seed"] is None


def _assert_shared_provenance(
    fact: Fact, *, source_metadata: dict[str, Any], run_id: Any
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
    _assert_metadata_method(fact.method)


def _assert_exact_fact_common(
    fact: Fact, *, source_metadata: dict[str, Any], run_id: Any, scan_max_records: int
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
    _assert_exact_count_method(fact.method, scan_max_records=scan_max_records)


def _assert_modality_fact_common(
    fact: Fact,
    *,
    source_metadata: dict[str, Any],
    run_id: Any,
    parent_fact_ids: list[Any],
) -> None:
    assert fact.run_id == run_id
    assert fact.confidence is None
    assert fact.parent_fact_ids == parent_fact_ids
    assert fact.source == {
        "source_uri": source_metadata["source_uri"],
        "revision": source_metadata["revision"],
        "file_path": None,
        "field": None,
        "record_id": None,
    }
    _assert_modality_method(fact.method)


def _assert_no_estimated_work(fact: Fact) -> None:
    assert "estimated_work" not in fact.value


def _assert_metric_order(facts: list[Fact]) -> None:
    assert [fact.value["metric"] for fact in facts] == [
        "profiler.source_metadata",
        "profiler.capabilities",
        "profiler.schema",
        "profiler.record_count.indexed",
        "profiler.record_count.exact",
        "profiler.modality",
    ]


def test_profile_writes_source_capabilities_and_schema_facts(
    store: EvidenceStore, writer: TrustedWriter
) -> None:
    connector = _build_connector(FLAT_SCHEMA)
    original_metadata = deepcopy(connector._source_metadata)
    original_schema = deepcopy(connector._schema)
    original_capabilities = CAPABILITIES.model_dump(mode="json")
    run_id = uuid4()

    facts = DatasetProfiler().profile(connector, writer, run_id)

    assert len(facts) == 6
    _assert_metric_order(facts)
    assert all(fact.status is EvidenceStatus.OBSERVED for fact in facts[:4])
    assert facts[4].status is EvidenceStatus.UNKNOWN
    assert facts[5].status is EvidenceStatus.COMPUTED

    (
        source_fact,
        capabilities_fact,
        schema_fact,
        count_fact,
        exact_fact,
        modality_fact,
    ) = facts
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
    assert exact_fact.claim == "Exact record count was not computed"
    assert exact_fact.value == _unknown_exact_value(estimated_record_count=3)
    assert modality_fact.claim == (
        "Dataset modality classified as tabular from connector schema types"
    )
    assert modality_fact.value == {
        "metric": "profiler.modality",
        "modalities": ["tabular"],
        "basis": "schema_types",
        "schema_inference": None,
    }

    assert connector._source_metadata == original_metadata
    assert connector._schema == original_schema
    assert connector.list_records_calls == 0
    assert connector.get_record_calls == 0

    for fact in facts[:4]:
        assert isinstance(fact.claim, str) and fact.claim.strip()
        _assert_no_estimated_work(fact)
        _assert_shared_provenance(
            fact, source_metadata=original_metadata, run_id=run_id
        )
    _assert_no_estimated_work(modality_fact)
    _assert_exact_fact_common(
        exact_fact,
        source_metadata=original_metadata,
        run_id=run_id,
        scan_max_records=0,
    )
    _assert_modality_fact_common(
        modality_fact,
        source_metadata=original_metadata,
        run_id=run_id,
        parent_fact_ids=[schema_fact.id],
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
    assert connector.list_records_calls == 0
    assert connector.get_record_calls == 0


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

    assert len(facts) == 6
    assert len({fact.id for fact in facts}) == 6
    assert all(fact.run_id == run_id for fact in facts)
    assert all(fact.confidence is None for fact in facts)
    assert all(fact.parent_fact_ids == [] for fact in facts[:5])
    assert facts[5].parent_fact_ids == [facts[2].id]

    stored = store.get_by_run(run_id)
    assert len(stored) == 6
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
    _assert_no_estimated_work(count_fact)
    _assert_shared_provenance(
        count_fact, source_metadata=original_metadata, run_id=run_id
    )
    assert connector.source_metadata_calls == 1
    assert connector.capabilities_calls == 1
    assert connector.schema_calls == 1
    assert connector.list_records_calls == 0


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
    assert count_fact.value == _unknown_indexed_value()
    _assert_shared_provenance(
        count_fact, source_metadata=original_metadata, run_id=run_id
    )

    stored = store.get_by_run(run_id)
    assert count_fact.id in {fact.id for fact in stored}
    assert connector.source_metadata_calls == 1
    assert connector.capabilities_calls == 1
    assert connector.schema_calls == 1
    assert connector.list_records_calls == 0


def test_profile_exact_count_uses_authorized_scan_yield(
    store: EvidenceStore, writer: TrustedWriter
) -> None:
    stored_records = [
        {"id": "r1", "label": "a"},
        {"id": "r2", "label": "b"},
        {"id": "r3", "label": "c"},
    ]
    capabilities = _build_capabilities(estimated_record_count=2)
    connector = _build_connector(
        FLAT_SCHEMA,
        capabilities=capabilities,
        records=stored_records,
    )
    original_metadata = deepcopy(connector._source_metadata)
    run_id = uuid4()
    scan_max_records = 10

    facts = DatasetProfiler().profile(
        connector, writer, run_id, scan_max_records=scan_max_records
    )

    exact_fact = facts[4]
    assert exact_fact.status is EvidenceStatus.COMPUTED
    assert exact_fact.claim == (
        "Exact record count computed from an authorized record scan"
    )
    assert exact_fact.value == {
        "metric": "profiler.record_count.exact",
        "value": 3,
    }
    _assert_no_estimated_work(exact_fact)
    _assert_exact_fact_common(
        exact_fact,
        source_metadata=original_metadata,
        run_id=run_id,
        scan_max_records=scan_max_records,
    )
    assert connector.list_records_calls == 1
    assert connector.records_produced == len(stored_records)
    assert connector.get_record_calls == 0
    assert connector.capabilities_calls == 1

    stored = store.get_by_run(run_id)
    assert exact_fact.id in {fact.id for fact in stored}
    assert {fact.id for fact in stored} == {fact.id for fact in facts}


def test_profile_exact_count_disabled_when_scan_max_records_zero(
    store: EvidenceStore, writer: TrustedWriter
) -> None:
    connector = _build_connector(
        FLAT_SCHEMA,
        capabilities=_build_capabilities(estimated_record_count=0),
        records=[{"id": "r1"}],
    )
    original_metadata = deepcopy(connector._source_metadata)
    run_id = uuid4()

    facts = DatasetProfiler().profile(connector, writer, run_id, scan_max_records=0)

    exact_fact = facts[4]
    assert exact_fact.status is EvidenceStatus.UNKNOWN
    assert exact_fact.claim == "Exact record count was not computed"
    assert exact_fact.value == _unknown_exact_value(estimated_record_count=0)
    _assert_exact_fact_common(
        exact_fact,
        source_metadata=original_metadata,
        run_id=run_id,
        scan_max_records=0,
    )
    assert connector.list_records_calls == 0
    assert connector.records_produced == 0
    assert connector.get_record_calls == 0


@pytest.mark.parametrize(
    ("capabilities", "scan_max_records"),
    [
        (
            _build_capabilities(estimated_record_count=3, has_index_metadata=False),
            10,
        ),
        (
            _build_capabilities(
                estimated_record_count=3,
                media_is_referenced_not_present=True,
            ),
            10,
        ),
        (
            _build_capabilities(estimated_record_count=None),
            10,
        ),
        (
            _build_capabilities(estimated_record_count=100),
            10,
        ),
    ],
    ids=[
        "no_index_metadata",
        "media_referenced",
        "estimate_none",
        "estimate_above_limit",
    ],
)
def test_profile_exact_count_rejected_when_scan_gate_fails(
    store: EvidenceStore,
    writer: TrustedWriter,
    capabilities: CapabilityDescriptor,
    scan_max_records: int,
) -> None:
    connector = _build_connector(
        FLAT_SCHEMA,
        capabilities=capabilities,
        records=[{"id": "r1"}, {"id": "r2"}],
    )
    original_metadata = deepcopy(connector._source_metadata)
    run_id = uuid4()

    facts = DatasetProfiler().profile(
        connector, writer, run_id, scan_max_records=scan_max_records
    )

    exact_fact = facts[4]
    assert exact_fact.status is EvidenceStatus.UNKNOWN
    assert exact_fact.claim == "Exact record count was not computed"
    assert exact_fact.value == _unknown_exact_value(
        estimated_record_count=capabilities.estimated_record_count
    )
    _assert_exact_fact_common(
        exact_fact,
        source_metadata=original_metadata,
        run_id=run_id,
        scan_max_records=scan_max_records,
    )
    assert connector.list_records_calls == 0
    assert connector.records_produced == 0
    assert connector.get_record_calls == 0


def test_profile_rejects_negative_scan_max_records_before_connector_use(
    writer: TrustedWriter,
) -> None:
    connector = _build_connector(FLAT_SCHEMA)
    run_id = uuid4()

    with pytest.raises(ValueError, match="scan_max_records must be non-negative"):
        DatasetProfiler().profile(connector, writer, run_id, scan_max_records=-1)

    assert connector.source_metadata_calls == 0
    assert connector.capabilities_calls == 0
    assert connector.schema_calls == 0
    assert connector.list_records_calls == 0
    assert connector.get_record_calls == 0


def test_profile_modality_flat_scalar_schema_is_computed_tabular(
    store: EvidenceStore, writer: TrustedWriter
) -> None:
    schema = {
        "id": "str",
        "label": "str",
        "duration_s": "float",
        "_inference": "full",
        "_sample_rows": 3,
    }
    connector = _build_connector(schema)
    original_metadata = deepcopy(connector._source_metadata)
    run_id = uuid4()

    facts = DatasetProfiler().profile(connector, writer, run_id)

    assert len(facts) == 6
    _assert_metric_order(facts)
    schema_fact = facts[2]
    modality_fact = facts[5]
    assert modality_fact.status is EvidenceStatus.COMPUTED
    assert modality_fact.claim == (
        "Dataset modality classified as tabular from connector schema types"
    )
    assert modality_fact.value == {
        "metric": "profiler.modality",
        "modalities": ["tabular"],
        "basis": "schema_types",
        "schema_inference": "full",
    }
    _assert_modality_fact_common(
        modality_fact,
        source_metadata=original_metadata,
        run_id=run_id,
        parent_fact_ids=[schema_fact.id],
    )
    assert connector.list_records_calls == 0
    assert connector.get_record_calls == 0

    stored = store.get_by_run(run_id)
    assert modality_fact.id in {fact.id for fact in stored}
    assert {fact.id for fact in stored} == {fact.id for fact in facts}


def test_profile_modality_wrapped_full_scalar_schema_is_computed_tabular(
    writer: TrustedWriter,
) -> None:
    connector = _build_connector(WRAPPED_FULL_SCALAR_SCHEMA)
    original_metadata = deepcopy(connector._source_metadata)
    run_id = uuid4()

    facts = DatasetProfiler().profile(connector, writer, run_id)

    modality_fact = facts[5]
    assert modality_fact.status is EvidenceStatus.COMPUTED
    assert modality_fact.claim == (
        "Dataset modality classified as tabular from connector schema types"
    )
    assert modality_fact.value == {
        "metric": "profiler.modality",
        "modalities": ["tabular"],
        "basis": "schema_types",
        "schema_inference": "full",
    }
    _assert_modality_fact_common(
        modality_fact,
        source_metadata=original_metadata,
        run_id=run_id,
        parent_fact_ids=[facts[2].id],
    )
    assert connector.list_records_calls == 0
    assert connector.get_record_calls == 0


def test_profile_modality_sample_head_scalar_preserves_inference(
    writer: TrustedWriter,
) -> None:
    connector = _build_connector(WRAPPED_SCHEMA)
    original_metadata = deepcopy(connector._source_metadata)
    run_id = uuid4()

    facts = DatasetProfiler().profile(connector, writer, run_id)

    modality_fact = facts[5]
    assert modality_fact.status is EvidenceStatus.COMPUTED
    assert modality_fact.claim == (
        "Dataset modality classified as tabular from connector schema types"
    )
    assert modality_fact.value == {
        "metric": "profiler.modality",
        "modalities": ["tabular"],
        "basis": "schema_types",
        "schema_inference": "sample_head",
    }
    _assert_modality_fact_common(
        modality_fact,
        source_metadata=original_metadata,
        run_id=run_id,
        parent_fact_ids=[facts[2].id],
    )
    assert connector.list_records_calls == 0
    assert connector.get_record_calls == 0


def test_profile_modality_repository_file_takes_precedence(
    writer: TrustedWriter,
) -> None:
    schema = {
        **deepcopy(REPOSITORY_FILE_SCHEMA),
        "columns": {
            **REPOSITORY_FILE_SCHEMA["columns"],
            "thumbnail": "Image",
            "clip": "Audio",
        },
    }
    connector = _build_connector(schema)
    original_metadata = deepcopy(connector._source_metadata)
    run_id = uuid4()

    facts = DatasetProfiler().profile(connector, writer, run_id)

    modality_fact = facts[5]
    assert modality_fact.status is EvidenceStatus.OBSERVED
    assert modality_fact.claim == "Connector schema declares repository-file records"
    assert modality_fact.value == {
        "metric": "profiler.modality",
        "modalities": ["repository_file"],
        "basis": "schema_record_model",
        "schema_inference": "full",
    }
    _assert_modality_fact_common(
        modality_fact,
        source_metadata=original_metadata,
        run_id=run_id,
        parent_fact_ids=[facts[2].id],
    )
    assert connector.list_records_calls == 0
    assert connector.get_record_calls == 0


@pytest.mark.parametrize(
    ("type_token", "modality"),
    [
        ("Text", "text"),
        ("Image", "image"),
        ("Audio", "audio"),
        ("Video", "video"),
        ("text", "text"),
        ("IMAGE", "image"),
        ("Image(mode=None, decode=True)", "image"),
        ("Audio(sampling_rate=16000, decode=True)", "audio"),
        ("Video(decode=True)", "video"),
        ("Sequence(feature=Text)", "text"),
    ],
    ids=[
        "Text",
        "Image",
        "Audio",
        "Video",
        "text_case",
        "IMAGE_case",
        "Image_structured",
        "Audio_structured",
        "Video_structured",
        "Sequence_Text",
    ],
)
def test_profile_modality_explicit_content_type_is_observed(
    writer: TrustedWriter, type_token: str, modality: str
) -> None:
    schema = {
        "columns": {"payload": type_token},
        "_inference": "full",
    }
    connector = _build_connector(schema)
    original_metadata = deepcopy(connector._source_metadata)
    run_id = uuid4()

    facts = DatasetProfiler().profile(connector, writer, run_id)

    modality_fact = facts[5]
    assert modality_fact.status is EvidenceStatus.OBSERVED
    assert modality_fact.claim == "Connector schema declares dataset modality"
    assert modality_fact.value == {
        "metric": "profiler.modality",
        "modalities": [modality],
        "basis": "schema_types",
        "schema_inference": "full",
    }
    _assert_modality_fact_common(
        modality_fact,
        source_metadata=original_metadata,
        run_id=run_id,
        parent_fact_ids=[facts[2].id],
    )
    assert connector.list_records_calls == 0
    assert connector.get_record_calls == 0


def test_profile_modality_nested_explicit_type_descriptions_are_recognized(
    writer: TrustedWriter,
) -> None:
    schema = {
        "columns": {
            "media": {
                "dtype": "Image",
                "nested": [{"inner": "Audio"}],
            }
        },
        "_inference": "full",
    }
    connector = _build_connector(schema)
    original_metadata = deepcopy(connector._source_metadata)
    run_id = uuid4()

    facts = DatasetProfiler().profile(connector, writer, run_id)

    modality_fact = facts[5]
    assert modality_fact.status is EvidenceStatus.OBSERVED
    assert modality_fact.claim == "Connector schema declares dataset modality"
    assert modality_fact.value == {
        "metric": "profiler.modality",
        "modalities": ["multimodal", "image", "audio"],
        "basis": "schema_types",
        "schema_inference": "full",
    }
    _assert_modality_fact_common(
        modality_fact,
        source_metadata=original_metadata,
        run_id=run_id,
        parent_fact_ids=[facts[2].id],
    )
    assert connector.list_records_calls == 0
    assert connector.get_record_calls == 0


def test_profile_modality_multiple_explicit_types_are_ordered_multimodal(
    writer: TrustedWriter,
) -> None:
    schema = {
        "columns": {
            "clip": "Video",
            "caption": "Text",
            "frame": "Image",
            "track": "Audio",
        },
        "_inference": "full",
    }
    connector = _build_connector(schema)
    original_metadata = deepcopy(connector._source_metadata)
    run_id = uuid4()

    facts = DatasetProfiler().profile(connector, writer, run_id)

    modality_fact = facts[5]
    assert modality_fact.status is EvidenceStatus.OBSERVED
    assert modality_fact.claim == "Connector schema declares dataset modality"
    assert modality_fact.value == {
        "metric": "profiler.modality",
        "modalities": ["multimodal", "text", "image", "audio", "video"],
        "basis": "schema_types",
        "schema_inference": "full",
    }
    _assert_modality_fact_common(
        modality_fact,
        source_metadata=original_metadata,
        run_id=run_id,
        parent_fact_ids=[facts[2].id],
    )
    assert connector.list_records_calls == 0
    assert connector.get_record_calls == 0


def test_profile_modality_media_like_field_names_with_scalar_types_stay_tabular(
    writer: TrustedWriter,
) -> None:
    schema = {
        "columns": {
            "image": "str",
            "audio_path": "string",
            "video_url": "str",
            "text": "str",
        },
        "_inference": "full",
    }
    connector = _build_connector(schema)
    original_metadata = deepcopy(connector._source_metadata)
    run_id = uuid4()

    facts = DatasetProfiler().profile(connector, writer, run_id)

    modality_fact = facts[5]
    assert modality_fact.status is EvidenceStatus.COMPUTED
    assert modality_fact.claim == (
        "Dataset modality classified as tabular from connector schema types"
    )
    assert modality_fact.value == {
        "metric": "profiler.modality",
        "modalities": ["tabular"],
        "basis": "schema_types",
        "schema_inference": "full",
    }
    _assert_modality_fact_common(
        modality_fact,
        source_metadata=original_metadata,
        run_id=run_id,
        parent_fact_ids=[facts[2].id],
    )
    assert connector.list_records_calls == 0
    assert connector.get_record_calls == 0


@pytest.mark.parametrize(
    "schema",
    [
        {"columns": {}, "_inference": "unavailable"},
        {"columns": {}, "_inference": "full"},
        {"_inference": "unavailable", "columns": {"id": "str"}},
        {},
    ],
    ids=[
        "unavailable_empty_columns",
        "empty_columns",
        "unavailable_with_columns",
        "empty_flat_schema",
    ],
)
def test_profile_modality_incomplete_schema_is_unknown(
    writer: TrustedWriter, schema: dict[str, Any]
) -> None:
    connector = _build_connector(schema)
    original_metadata = deepcopy(connector._source_metadata)
    run_id = uuid4()

    facts = DatasetProfiler().profile(connector, writer, run_id)

    modality_fact = facts[5]
    assert modality_fact.status is EvidenceStatus.UNKNOWN
    assert modality_fact.claim == "Dataset modality could not be determined"
    assert modality_fact.value == {
        "metric": "profiler.modality",
        "modalities": ["unknown"],
        "basis": "none",
        "schema_inference": schema.get("_inference"),
        "reason": "schema_incomplete",
        "explanation": _SCHEMA_INCOMPLETE_EXPLANATION,
        "estimated_work": _ESTIMATED_WORK_EMPTY,
    }
    _assert_modality_fact_common(
        modality_fact,
        source_metadata=original_metadata,
        run_id=run_id,
        parent_fact_ids=[facts[2].id],
    )
    assert connector.list_records_calls == 0
    assert connector.get_record_calls == 0


def test_profile_modality_unsupported_types_are_ambiguous(
    writer: TrustedWriter,
) -> None:
    schema = {
        "columns": {
            "embedding": "float32[768]",
            "payload": "custom_struct",
        },
        "_inference": "full",
    }
    connector = _build_connector(schema)
    original_metadata = deepcopy(connector._source_metadata)
    run_id = uuid4()

    facts = DatasetProfiler().profile(connector, writer, run_id)

    modality_fact = facts[5]
    assert modality_fact.status is EvidenceStatus.UNKNOWN
    assert modality_fact.claim == "Dataset modality could not be determined"
    assert modality_fact.value == {
        "metric": "profiler.modality",
        "modalities": ["unknown"],
        "basis": "none",
        "schema_inference": "full",
        "reason": "modality_ambiguous",
        "explanation": _MODALITY_AMBIGUOUS_EXPLANATION,
        "estimated_work": _ESTIMATED_WORK_EMPTY,
    }
    _assert_modality_fact_common(
        modality_fact,
        source_metadata=original_metadata,
        run_id=run_id,
        parent_fact_ids=[facts[2].id],
    )
    assert connector.list_records_calls == 0
    assert connector.get_record_calls == 0


def test_profile_modality_does_not_match_partial_type_tokens(
    writer: TrustedWriter,
) -> None:
    schema = {
        "columns": {
            "label": "ImageNetLabel",
            "clip": "audiovisual",
            "shot": "videography",
            "body": "textual",
        },
        "_inference": "full",
    }
    connector = _build_connector(schema)
    original_metadata = deepcopy(connector._source_metadata)
    run_id = uuid4()

    facts = DatasetProfiler().profile(connector, writer, run_id)

    modality_fact = facts[5]
    assert modality_fact.status is EvidenceStatus.UNKNOWN
    assert modality_fact.claim == "Dataset modality could not be determined"
    assert modality_fact.value == {
        "metric": "profiler.modality",
        "modalities": ["unknown"],
        "basis": "none",
        "schema_inference": "full",
        "reason": "modality_ambiguous",
        "explanation": _MODALITY_AMBIGUOUS_EXPLANATION,
        "estimated_work": _ESTIMATED_WORK_EMPTY,
    }
    _assert_modality_fact_common(
        modality_fact,
        source_metadata=original_metadata,
        run_id=run_id,
        parent_fact_ids=[facts[2].id],
    )
    assert connector.list_records_calls == 0
    assert connector.get_record_calls == 0


def test_profile_modality_media_referenced_without_declared_type_is_unknown(
    writer: TrustedWriter,
) -> None:
    connector = _build_connector(
        FLAT_SCHEMA,
        capabilities=_build_capabilities(media_is_referenced_not_present=True),
    )
    original_metadata = deepcopy(connector._source_metadata)
    run_id = uuid4()

    facts = DatasetProfiler().profile(connector, writer, run_id)

    capabilities_fact = facts[1]
    schema_fact = facts[2]
    modality_fact = facts[5]
    assert modality_fact.status is EvidenceStatus.UNKNOWN
    assert modality_fact.claim == "Dataset modality could not be determined"
    assert modality_fact.value == {
        "metric": "profiler.modality",
        "modalities": ["unknown"],
        "basis": "none",
        "schema_inference": None,
        "reason": "media_referenced_not_present",
        "explanation": _MEDIA_REFERENCED_EXPLANATION,
        "estimated_work": _ESTIMATED_WORK_FILES_TO_OPEN_NONE,
    }
    _assert_modality_fact_common(
        modality_fact,
        source_metadata=original_metadata,
        run_id=run_id,
        parent_fact_ids=[capabilities_fact.id, schema_fact.id],
    )
    assert connector.list_records_calls == 0
    assert connector.get_record_calls == 0


def test_profile_success_facts_do_not_include_estimated_work(
    writer: TrustedWriter,
) -> None:
    connector = _build_connector(
        {
            "columns": {"frame": "Image"},
            "_inference": "full",
        },
        capabilities=_build_capabilities(estimated_record_count=2),
        records=[{"id": "r1"}, {"id": "r2"}, {"id": "r3"}],
    )
    run_id = uuid4()
    scan_max_records = 10

    facts = DatasetProfiler().profile(
        connector, writer, run_id, scan_max_records=scan_max_records
    )

    for fact in facts:
        if fact.status is not EvidenceStatus.UNKNOWN:
            _assert_no_estimated_work(fact)

    assert connector.source_metadata_calls == 1
    assert connector.capabilities_calls == 1
    assert connector.schema_calls == 1
    assert connector.list_records_calls == 1
    assert connector.get_record_calls == 0


def test_profile_modality_explicit_media_observed_when_bytes_referenced(
    writer: TrustedWriter,
) -> None:
    schema = {
        "columns": {"frame": "Image"},
        "_inference": "full",
    }
    connector = _build_connector(
        schema,
        capabilities=_build_capabilities(media_is_referenced_not_present=True),
    )
    original_metadata = deepcopy(connector._source_metadata)
    run_id = uuid4()

    facts = DatasetProfiler().profile(connector, writer, run_id)

    modality_fact = facts[5]
    assert modality_fact.status is EvidenceStatus.OBSERVED
    assert modality_fact.claim == "Connector schema declares dataset modality"
    assert modality_fact.value == {
        "metric": "profiler.modality",
        "modalities": ["image"],
        "basis": "schema_types",
        "schema_inference": "full",
    }
    _assert_modality_fact_common(
        modality_fact,
        source_metadata=original_metadata,
        run_id=run_id,
        parent_fact_ids=[facts[2].id],
    )
    assert connector.list_records_calls == 0
    assert connector.get_record_calls == 0
