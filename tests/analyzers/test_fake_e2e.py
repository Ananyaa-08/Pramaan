"""End-to-end proof: connector -> analyzer -> TrustedWriter -> store -> query."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from auditor.analyzers.fake import CountRecordsAnalyzer
from auditor.connectors.base import CapabilityDescriptor
from auditor.connectors.fake import InMemoryConnector
from auditor.evidence.models import EvidenceStatus
from auditor.evidence.store import EvidenceStore
from auditor.evidence.writers import TrustedWriter


def _fixture_records() -> list[dict[str, Any]]:
    return [
        {"id": "r1", "label": "a"},
        {"id": "r2", "label": "b"},
        {"id": "r3", "label": "c"},
        {"id": "r4", "label": "d"},
        {"id": "r5", "label": "e"},
    ]


def test_count_records_analyzer_full_chain(tmp_path: Path) -> None:
    records = _fixture_records()
    connector = InMemoryConnector(
        records=records,
        schema={"id": "str", "label": "str"},
        source_metadata={
            "source_uri": "memory://fixture-5",
            "revision": "fixture-rev",
            "provider": "in_memory",
        },
        capabilities=CapabilityDescriptor(
            can_stream=True,
            can_random_access=True,
            has_index_metadata=True,
            media_is_referenced_not_present=False,
            requires_auth=False,
            estimated_record_count=5,
        ),
        id_field="id",
    )
    store = EvidenceStore(db_path=tmp_path / "evidence.db")
    writer = TrustedWriter(store)
    analyzer = CountRecordsAnalyzer()

    produced = analyzer.run(connector, writer)

    assert len(produced) == 1
    fact = produced[0]
    assert fact.status is EvidenceStatus.COMPUTED
    assert fact.value["metric"] == "record_count"
    assert fact.value["value"] == 5

    stored = store.get_by_status(EvidenceStatus.COMPUTED)
    assert len(stored) == 1
    assert stored[0].id == fact.id
    assert stored[0].value == fact.value
    assert stored[0].claim == fact.claim


def test_count_records_analyzer_respects_record_ids(tmp_path: Path) -> None:
    records = _fixture_records()
    connector = InMemoryConnector(
        records=records,
        schema={"id": "str", "label": "str"},
        source_metadata={
            "source_uri": "memory://fixture-5",
            "revision": "fixture-rev",
            "provider": "in_memory",
        },
        capabilities=CapabilityDescriptor(
            can_stream=True,
            can_random_access=True,
            has_index_metadata=True,
            media_is_referenced_not_present=False,
            requires_auth=False,
            estimated_record_count=5,
        ),
        id_field="id",
    )
    store = EvidenceStore(db_path=tmp_path / "evidence.db")
    writer = TrustedWriter(store)
    analyzer = CountRecordsAnalyzer()

    produced = analyzer.run(connector, writer, record_ids=["r1", "r3", "r5"])

    assert len(produced) == 1
    assert produced[0].value["value"] == 3
    stored = store.get_by_status(EvidenceStatus.COMPUTED)
    assert len(stored) == 1
    assert stored[0].value["value"] == 3
