"""Tests for the evidence Fact model and append-only EvidenceStore (design §3).

Inserts go through TrustedWriter / InferredWriter so that `_insert` is never
called directly from tests (see test_writers.py AST enforcement).
"""

from __future__ import annotations

import ast
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from auditor.evidence.errors import DuplicateFactError
from auditor.evidence.models import EvidenceStatus, Fact
from auditor.evidence.store import EvidenceStore
from auditor.evidence.writers import InferredWriter, TrustedWriter


def _source(**overrides: object) -> dict:
    base: dict = {
        "source_uri": None,
        "revision": None,
        "file_path": None,
        "field": None,
        "record_id": None,
    }
    base.update(overrides)
    return base


def _method(**overrides: object) -> dict:
    base: dict = {
        "analyzer_name": "test_analyzer",
        "analyzer_version": "0.0.1",
        "model_name": None,
        "model_version": None,
        "parameters": {},
        "sampling_strategy": None,
        "sampling_seed": None,
    }
    base.update(overrides)
    return base


def make_fact(
    *,
    status: EvidenceStatus = EvidenceStatus.OBSERVED,
    claim: str = "test claim",
    value: dict | None = None,
    run_id: uuid.UUID | None = None,
    parent_fact_ids: list[uuid.UUID] | None = None,
    confidence: float | None = None,
    fact_id: uuid.UUID | None = None,
    source: dict | None = None,
    method: dict | None = None,
) -> Fact:
    """Build a Fact with sensible defaults for store tests."""
    kwargs: dict = {
        "status": status,
        "claim": claim,
        "value": value if value is not None else {"metric": "n", "value": 1},
        "source": source if source is not None else _source(),
        "method": method if method is not None else _method(),
        "run_id": run_id if run_id is not None else uuid.uuid4(),
        "parent_fact_ids": (
            parent_fact_ids if parent_fact_ids is not None else []
        ),
        "confidence": confidence,
    }
    if fact_id is not None:
        kwargs["id"] = fact_id
    return Fact(**kwargs)


@pytest.fixture
def store(tmp_path: Path) -> EvidenceStore:
    return EvidenceStore(db_path=tmp_path / "evidence.db")


@pytest.fixture
def trusted(store: EvidenceStore) -> TrustedWriter:
    return TrustedWriter(store)


@pytest.fixture
def inferred(store: EvidenceStore) -> InferredWriter:
    return InferredWriter(store)


# ---------------------------------------------------------------------------
# EvidenceStatus & Fact construction / confidence rule (§3.1)
# ---------------------------------------------------------------------------


def test_evidence_status_has_exactly_seven_values() -> None:
    assert {s.name for s in EvidenceStatus} == {
        "OBSERVED",
        "COMPUTED",
        "DERIVED",
        "DOCUMENTED",
        "INFERRED",
        "UNKNOWN",
        "DISCREPANCY",
    }
    assert len(EvidenceStatus) == 7


def test_fact_is_frozen_immutable() -> None:
    fact = make_fact()
    with pytest.raises(ValidationError):
        fact.claim = "mutated"  # type: ignore[misc]


def test_fact_auto_generates_id_and_created_at() -> None:
    fact = make_fact()
    assert isinstance(fact.id, uuid.UUID)
    assert isinstance(fact.created_at, datetime)
    assert fact.created_at.tzinfo is not None
    assert fact.created_at.utcoffset() == datetime.now(UTC).utcoffset()


def test_inferred_without_confidence_raises() -> None:
    with pytest.raises(ValueError, match="confidence"):
        make_fact(status=EvidenceStatus.INFERRED, confidence=None)


def test_inferred_with_confidence_succeeds() -> None:
    fact = make_fact(status=EvidenceStatus.INFERRED, confidence=0.85)
    assert fact.confidence == 0.85


def test_non_inferred_with_confidence_raises() -> None:
    for status in (
        EvidenceStatus.OBSERVED,
        EvidenceStatus.COMPUTED,
        EvidenceStatus.DERIVED,
        EvidenceStatus.DOCUMENTED,
        EvidenceStatus.UNKNOWN,
        EvidenceStatus.DISCREPANCY,
    ):
        with pytest.raises(ValueError, match="confidence"):
            make_fact(status=status, confidence=0.5)


def test_confidence_out_of_range_raises_for_inferred() -> None:
    with pytest.raises(ValueError):
        make_fact(status=EvidenceStatus.INFERRED, confidence=1.5)
    with pytest.raises(ValueError):
        make_fact(status=EvidenceStatus.INFERRED, confidence=-0.1)


# ---------------------------------------------------------------------------
# EvidenceStore reads (writes via partitioned writers only)
# ---------------------------------------------------------------------------


def test_write_and_get_by_id(store: EvidenceStore, trusted: TrustedWriter) -> None:
    fact = make_fact(claim="71 signer IDs present")
    trusted.write(fact)

    loaded = store.get_by_id(fact.id)
    assert loaded is not None
    assert loaded.id == fact.id
    assert loaded.claim == "71 signer IDs present"
    assert loaded.status == EvidenceStatus.OBSERVED


def test_get_by_id_missing_returns_none(store: EvidenceStore) -> None:
    assert store.get_by_id(uuid.uuid4()) is None


def test_get_by_status(store: EvidenceStore, trusted: TrustedWriter) -> None:
    observed = make_fact(status=EvidenceStatus.OBSERVED, claim="observed")
    computed = make_fact(status=EvidenceStatus.COMPUTED, claim="computed")
    trusted.write(observed)
    trusted.write(computed)

    results = store.get_by_status(EvidenceStatus.OBSERVED)
    assert len(results) == 1
    assert results[0].id == observed.id


def test_get_by_run(store: EvidenceStore, trusted: TrustedWriter) -> None:
    run_a = uuid.uuid4()
    run_b = uuid.uuid4()
    a1 = make_fact(run_id=run_a, claim="a1")
    a2 = make_fact(run_id=run_a, claim="a2")
    b1 = make_fact(run_id=run_b, claim="b1")
    for f in (a1, a2, b1):
        trusted.write(f)

    results = store.get_by_run(run_a)
    assert {f.id for f in results} == {a1.id, a2.id}


def test_store_rejects_fact_violating_confidence_rule(
    trusted: TrustedWriter, inferred: InferredWriter
) -> None:
    """Store rejects confidence-rule violations even if validators were bypassed."""
    bad_inferred = Fact.model_construct(
        id=uuid.uuid4(),
        status=EvidenceStatus.INFERRED,
        claim="unguarded inferred",
        value={"x": 1},
        source=_source(),
        method=_method(),
        run_id=uuid.uuid4(),
        parent_fact_ids=[],
        confidence=None,
        created_at=datetime.now(UTC),
    )
    with pytest.raises(ValueError, match="confidence"):
        inferred.write(bad_inferred)

    bad_observed = Fact.model_construct(
        id=uuid.uuid4(),
        status=EvidenceStatus.OBSERVED,
        claim="observed with confidence",
        value={"x": 1},
        source=_source(),
        method=_method(),
        run_id=uuid.uuid4(),
        parent_fact_ids=[],
        confidence=0.9,
        created_at=datetime.now(UTC),
    )
    with pytest.raises(ValueError, match="confidence"):
        trusted.write(bad_observed)


def test_get_lineage_three_level_root_first(
    store: EvidenceStore, trusted: TrustedWriter, inferred: InferredWriter
) -> None:
    run_id = uuid.uuid4()
    grandparent = make_fact(
        status=EvidenceStatus.OBSERVED,
        claim="grandparent",
        run_id=run_id,
    )
    parent = make_fact(
        status=EvidenceStatus.COMPUTED,
        claim="parent",
        run_id=run_id,
        parent_fact_ids=[grandparent.id],
    )
    child = make_fact(
        status=EvidenceStatus.INFERRED,
        claim="child",
        run_id=run_id,
        parent_fact_ids=[parent.id],
        confidence=0.7,
    )
    trusted.write(grandparent)
    trusted.write(parent)
    inferred.write(child)

    lineage = store.get_lineage(child.id)
    assert [f.id for f in lineage] == [grandparent.id, parent.id, child.id]
    assert [f.claim for f in lineage] == ["grandparent", "parent", "child"]
    assert lineage[-1].id == child.id


def test_get_lineage_missing_fact_raises(store: EvidenceStore) -> None:
    with pytest.raises(ValueError):
        store.get_lineage(uuid.uuid4())


def test_duplicate_id_insert_does_not_corrupt_original(
    store: EvidenceStore, trusted: TrustedWriter
) -> None:
    """Append-only: a second write with the same id must not overwrite the first."""
    fact_id = uuid.uuid4()
    original = make_fact(fact_id=fact_id, claim="original claim", value={"v": 1})
    trusted.write(original)

    duplicate = make_fact(
        fact_id=fact_id, claim="corrupted claim", value={"v": 999}
    )
    with pytest.raises(DuplicateFactError):
        trusted.write(duplicate)

    loaded = store.get_by_id(fact_id)
    assert loaded is not None
    assert loaded.claim == "original claim"
    assert loaded.value == {"v": 1}


def test_store_sql_strings_are_append_only() -> None:
    """EvidenceStore must not embed UPDATE/DELETE SQL (append-only)."""
    store_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "auditor"
        / "evidence"
        / "store.py"
    )
    tree = ast.parse(store_path.read_text(encoding="utf-8"))
    sql_chunks: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            sql_chunks.append(node.value.upper())
    for chunk in sql_chunks:
        padded = f" {chunk} "
        assert " UPDATE " not in padded
        assert " DELETE " not in padded
        assert not chunk.strip().startswith("UPDATE")
        assert not chunk.strip().startswith("DELETE")
