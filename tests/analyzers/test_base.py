"""Tests for the analyzer interface (design §4.5)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import get_type_hints

import pytest

from auditor.analyzers.base import Analyzer, AnalyzerCapability, CostTier
from auditor.analyzers.fake import CountRecordsAnalyzer
from auditor.connectors.base import CapabilityDescriptor, Connector
from auditor.connectors.fake import InMemoryConnector
from auditor.evidence.models import EvidenceStatus, Fact
from auditor.evidence.store import EvidenceStore
from auditor.evidence.writers import InferredWriter, TrustedWriter


def _capability(
    *,
    emits_status: EvidenceStatus = EvidenceStatus.COMPUTED,
) -> AnalyzerCapability:
    return AnalyzerCapability(
        analyzer_name="test",
        version="0.0.1",
        modality="tabular",
        cost_tier=CostTier.FREE,
        emits_status=emits_status,
        questions_answered=["record_count"],
        required_fields=[],
    )


def _source() -> dict:
    return {
        "source_uri": None,
        "revision": None,
        "file_path": None,
        "field": None,
        "record_id": None,
    }


def _method() -> dict:
    return {
        "analyzer_name": "test_analyzer",
        "analyzer_version": "0.0.1",
        "model_name": None,
        "model_version": None,
        "parameters": {},
        "sampling_strategy": None,
        "sampling_seed": None,
    }


def make_fact(
    *,
    status: EvidenceStatus,
    confidence: float | None = None,
    claim: str = "claim",
) -> Fact:
    return Fact(
        status=status,
        claim=claim,
        value={"metric": "n", "value": 1},
        source=_source(),
        method=_method(),
        run_id=uuid.uuid4(),
        parent_fact_ids=[],
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# AnalyzerCapability: emits_status constraint (§3.1 writer table)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [EvidenceStatus.OBSERVED, EvidenceStatus.COMPUTED, EvidenceStatus.DERIVED],
)
def test_analyzer_capability_accepts_allowed_emit_statuses(
    status: EvidenceStatus,
) -> None:
    cap = _capability(emits_status=status)
    assert cap.emits_status is status


@pytest.mark.parametrize(
    "status",
    [
        EvidenceStatus.INFERRED,
        EvidenceStatus.DISCREPANCY,
        EvidenceStatus.DOCUMENTED,
        EvidenceStatus.UNKNOWN,
    ],
)
def test_analyzer_capability_rejects_disallowed_emit_statuses(
    status: EvidenceStatus,
) -> None:
    with pytest.raises(ValueError):
        _capability(emits_status=status)


def test_analyzer_capability_is_frozen() -> None:
    cap = _capability()
    with pytest.raises(Exception):
        cap.analyzer_name = "mutated"  # type: ignore[misc]


def test_cost_tier_has_expected_values() -> None:
    assert {t.name for t in CostTier} == {
        "FREE",
        "CHEAP",
        "MODERATE",
        "EXPENSIVE",
    }


# ---------------------------------------------------------------------------
# Analyzer.run signature requires TrustedWriter
# ---------------------------------------------------------------------------


def test_analyzer_run_type_hints_require_trusted_writer() -> None:
    """Static contract: writer is typed TrustedWriter, not EvidenceStore."""
    hints = get_type_hints(Analyzer.run)
    assert hints["writer"] is TrustedWriter
    assert hints["connector"] is Connector
    assert hints["writer"] is not EvidenceStore


def test_count_records_rejects_inferred_writer_at_runtime(
    tmp_path: Path,
) -> None:
    """
    CountRecordsAnalyzer's own isinstance(writer, TrustedWriter) check must fire.

    Matches TypeError specifically so InferredWriter's PermissionError cannot
    silently cover for a missing analyzer-layer guard.
    """
    records = [{"id": f"r{i}"} for i in range(3)]
    connector = InMemoryConnector(
        records=records,
        schema={"id": "str"},
        source_metadata={
            "source_uri": "memory://test",
            "revision": "r1",
            "provider": "in_memory",
        },
        capabilities=CapabilityDescriptor(
            can_stream=True,
            can_random_access=True,
            has_index_metadata=True,
            media_is_referenced_not_present=False,
            requires_auth=False,
            estimated_record_count=3,
        ),
    )
    store = EvidenceStore(db_path=tmp_path / "evidence.db")
    analyzer = CountRecordsAnalyzer()

    with pytest.raises(TypeError, match="requires a TrustedWriter"):
        analyzer.run(connector, InferredWriter(store))  # type: ignore[arg-type]


def test_inferred_writer_independently_rejects_computed_fact(
    tmp_path: Path,
) -> None:
    """Writer-layer defense: InferredWriter rejects COMPUTED without the analyzer."""
    store = EvidenceStore(db_path=tmp_path / "evidence.db")
    writer = InferredWriter(store)
    fact = make_fact(status=EvidenceStatus.COMPUTED)

    with pytest.raises(PermissionError):
        writer.write(fact)
