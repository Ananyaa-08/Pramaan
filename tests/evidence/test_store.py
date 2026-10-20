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
        "parent_fact_ids": (parent_fact_ids if parent_fact_ids is not None else []),
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

    duplicate = make_fact(fact_id=fact_id, claim="corrupted claim", value={"v": 999})
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


# ---------------------------------------------------------------------------
# Owner D-4: parent existence / repeated-parent checks (tests only)
# ---------------------------------------------------------------------------

_DUPLICATE_FACT_MESSAGE = "already exists in the evidence store"


def _import_missing_parent_fact_error() -> type[Exception]:
    """Import MissingParentFactError lazily so collection survives its absence."""
    try:
        from auditor.evidence.errors import MissingParentFactError
    except ImportError as exc:
        pytest.fail(
            "MissingParentFactError is not implemented yet; "
            f"D-4 parent checks are unavailable ({exc})"
        )
    return MissingParentFactError


def _assert_repeated_parent_message(message: str, repeated_ids: set[uuid.UUID]) -> None:
    """Message must identify every repeated ID without requiring ID order."""
    lowered = message.lower()
    assert "repeated" in lowered
    assert "parent" in lowered
    for repeated_id in repeated_ids:
        assert str(repeated_id) in message


def _assert_validation_error_names_repeated_parents(
    exc: ValidationError, repeated_ids: set[uuid.UUID]
) -> None:
    errors = exc.errors()
    assert errors
    combined = " ".join(str(error.get("msg", "")) for error in errors)
    loc_blob = " ".join(str(error.get("loc", ())) for error in errors)
    assert "parent" in combined.lower() or "parent_fact_ids" in loc_blob
    _assert_repeated_parent_message(combined, repeated_ids)


def _assert_facts_equal(left: Fact, right: Fact) -> None:
    assert left.id == right.id
    assert left.status == right.status
    assert left.claim == right.claim
    assert left.value == right.value
    assert left.source == right.source
    assert left.method == right.method
    assert left.run_id == right.run_id
    assert left.parent_fact_ids == right.parent_fact_ids
    assert left.confidence == right.confidence
    assert left.created_at == right.created_at


def _snapshot_facts(
    store: EvidenceStore, fact_ids: set[uuid.UUID]
) -> dict[uuid.UUID, Fact]:
    snaps: dict[uuid.UUID, Fact] = {}
    for fact_id in fact_ids:
        loaded = store.get_by_id(fact_id)
        assert loaded is not None
        snaps[fact_id] = loaded
    return snaps


def _assert_snapshots_unchanged(
    store: EvidenceStore, before: dict[uuid.UUID, Fact]
) -> None:
    for fact_id, original in before.items():
        loaded = store.get_by_id(fact_id)
        assert loaded is not None
        _assert_facts_equal(loaded, original)


def _one_repeated_parents(
    a: uuid.UUID, b: uuid.UUID, c: uuid.UUID
) -> tuple[list[uuid.UUID], set[uuid.UUID]]:
    del c
    return [a, b, a], {a}


def _two_repeated_parents(
    a: uuid.UUID, b: uuid.UUID, c: uuid.UUID
) -> tuple[list[uuid.UUID], set[uuid.UUID]]:
    return [a, b, a, c, b], {a, b}


def _writer_for(
    name: str, trusted: TrustedWriter, inferred: InferredWriter
) -> TrustedWriter | InferredWriter:
    return trusted if name == "trusted" else inferred


def _child_status_confidence(
    writer_name: str,
) -> tuple[EvidenceStatus, float | None]:
    if writer_name == "trusted":
        return EvidenceStatus.OBSERVED, None
    return EvidenceStatus.INFERRED, 0.8


# --- Amendment: repeated parent IDs -----------------------------------------


@pytest.mark.parametrize(
    "build_parents",
    [
        pytest.param(_one_repeated_parents, id="one_repeated_id"),
        pytest.param(_two_repeated_parents, id="two_repeated_ids_one_unique"),
    ],
)
def test_owner_d4_fact_rejects_repeated_parent_ids(build_parents: object) -> None:
    """Normal Fact construction rejects repeated parent IDs (D-4 amendment)."""
    parent_a = uuid.uuid4()
    parent_b = uuid.uuid4()
    parent_c = uuid.uuid4()
    assert callable(build_parents)
    parent_ids, repeated_ids = build_parents(parent_a, parent_b, parent_c)

    with pytest.raises(ValidationError) as exc_info:
        make_fact(parent_fact_ids=parent_ids)

    _assert_validation_error_names_repeated_parents(exc_info.value, repeated_ids)


@pytest.mark.parametrize(
    "writer_name",
    [
        pytest.param("trusted", id="trusted_observed"),
        pytest.param("inferred", id="inferred"),
    ],
)
@pytest.mark.parametrize(
    "build_parents",
    [
        pytest.param(_one_repeated_parents, id="one_repeated_id"),
        pytest.param(_two_repeated_parents, id="two_repeated_ids_one_unique"),
    ],
)
def test_owner_d4_store_rejects_repeated_parent_ids_bypassing_fact_validation(
    store: EvidenceStore,
    trusted: TrustedWriter,
    inferred: InferredWriter,
    writer_name: str,
    build_parents: object,
) -> None:
    """Store revalidates repeated parents when Fact validators are bypassed."""
    assert callable(build_parents)
    writer = _writer_for(writer_name, trusted, inferred)
    status, confidence = _child_status_confidence(writer_name)
    run_id = uuid.uuid4()

    parent_a = make_fact(claim="parent-a", run_id=run_id)
    parent_b = make_fact(claim="parent-b", run_id=run_id)
    parent_c = make_fact(claim="parent-c", run_id=run_id)
    for parent in (parent_a, parent_b, parent_c):
        trusted.write(parent)

    parent_ids, repeated_ids = build_parents(parent_a.id, parent_b.id, parent_c.id)
    distinct_ids = set(parent_ids)
    before = _snapshot_facts(store, distinct_ids)

    child_id = uuid.uuid4()
    child = Fact.model_construct(
        id=child_id,
        status=status,
        claim="child-with-repeated-parents",
        value={"metric": "n", "value": 1},
        source=_source(),
        method=_method(),
        run_id=run_id,
        parent_fact_ids=list(parent_ids),
        confidence=confidence,
        created_at=datetime.now(UTC),
    )
    assert child.parent_fact_ids == parent_ids

    insert_statements: list[str] = []

    def _trace(statement: str) -> None:
        if "INSERT" in statement.upper():
            insert_statements.append(statement)

    store._conn.set_trace_callback(_trace)
    try:
        with pytest.raises(ValueError) as exc_info:
            writer.write(child)
    finally:
        store._conn.set_trace_callback(None)

    _assert_repeated_parent_message(str(exc_info.value), repeated_ids)
    assert store.get_by_id(child_id) is None
    _assert_snapshots_unchanged(store, before)
    assert insert_statements == []


def test_owner_d4_preserves_distinct_parent_order(
    store: EvidenceStore, trusted: TrustedWriter
) -> None:
    """Distinct parent IDs keep exact supplied order through write and read."""
    run_id = uuid.uuid4()
    # Fixed UUIDs: supplied order is deterministically not UUID-sorted order.
    id_low = uuid.UUID("00000000-0000-4000-8000-000000000001")
    id_mid = uuid.UUID("00000000-0000-4000-8000-000000000002")
    id_high = uuid.UUID("00000000-0000-4000-8000-000000000003")
    first = make_fact(fact_id=id_low, claim="order-low", run_id=run_id)
    second = make_fact(fact_id=id_mid, claim="order-mid", run_id=run_id)
    third = make_fact(fact_id=id_high, claim="order-high", run_id=run_id)
    for parent in (first, second, third):
        trusted.write(parent)

    expected_parents = (id_high, id_low, id_mid)
    assert list(expected_parents) != sorted(expected_parents)

    supplied_parents = list(expected_parents)
    assert supplied_parents != sorted(supplied_parents)

    child = make_fact(
        claim="ordered-child",
        run_id=run_id,
        parent_fact_ids=supplied_parents,
    )
    assert supplied_parents == list(expected_parents)
    assert child.parent_fact_ids == list(expected_parents)

    trusted.write(child)
    assert supplied_parents == list(expected_parents)
    assert child.parent_fact_ids == list(expected_parents)

    loaded = store.get_by_id(child.id)
    assert loaded is not None
    assert loaded.parent_fact_ids == list(expected_parents)


# --- Original D-4: missing-parent contract ----------------------------------


def test_owner_d4_missing_parent_error_is_dedicated() -> None:
    """MissingParentFactError is dedicated and mirrors DuplicateFactError bases."""
    MissingParentFactError = _import_missing_parent_fact_error()
    assert MissingParentFactError is not DuplicateFactError
    assert issubclass(MissingParentFactError, Exception)
    assert MissingParentFactError.__bases__ == DuplicateFactError.__bases__
    assert MissingParentFactError.__module__ == "auditor.evidence.errors"


def test_owner_d4_missing_parent_error_is_publicly_exported() -> None:
    import auditor.evidence as evidence
    from auditor.evidence import MissingParentFactError

    assert MissingParentFactError is evidence.MissingParentFactError
    assert "MissingParentFactError" in evidence.__all__


def test_owner_d4_root_fact_remains_valid(
    store: EvidenceStore, trusted: TrustedWriter
) -> None:
    """Empty parent list remains valid and persists."""
    root = make_fact(claim="root-empty-parents", parent_fact_ids=[])
    assert root.parent_fact_ids == []
    trusted.write(root)
    loaded = store.get_by_id(root.id)
    assert loaded is not None
    assert loaded.parent_fact_ids == []
    _assert_facts_equal(loaded, root)


@pytest.mark.parametrize(
    "writer_name",
    [
        pytest.param("trusted", id="trusted_observed"),
        pytest.param("inferred", id="inferred"),
    ],
)
def test_owner_d4_accepts_all_existing_parents(
    store: EvidenceStore,
    trusted: TrustedWriter,
    inferred: InferredWriter,
    writer_name: str,
) -> None:
    """Child with multiple distinct existing parents persists successfully."""
    writer = _writer_for(writer_name, trusted, inferred)
    status, confidence = _child_status_confidence(writer_name)
    run_id = uuid.uuid4()
    parents = [make_fact(claim=f"ok-parent-{i}", run_id=run_id) for i in range(3)]
    for parent in parents:
        trusted.write(parent)
    parent_ids = [parent.id for parent in parents]

    child = make_fact(
        status=status,
        confidence=confidence,
        claim="child-all-parents-present",
        run_id=run_id,
        parent_fact_ids=parent_ids,
    )
    writer.write(child)

    loaded = store.get_by_id(child.id)
    assert loaded is not None
    assert loaded.parent_fact_ids == parent_ids
    _assert_facts_equal(loaded, child)


@pytest.mark.parametrize(
    "writer_name",
    [
        pytest.param("trusted", id="trusted_observed"),
        pytest.param("inferred", id="inferred"),
    ],
)
def test_owner_d4_rejects_missing_parent(
    store: EvidenceStore,
    trusted: TrustedWriter,
    inferred: InferredWriter,
    writer_name: str,
) -> None:
    """MissingParentFactError names the missing UUID; existing rows unchanged."""
    MissingParentFactError = _import_missing_parent_fact_error()
    writer = _writer_for(writer_name, trusted, inferred)
    status, confidence = _child_status_confidence(writer_name)
    run_id = uuid.uuid4()
    existing = make_fact(claim="existing-parent", run_id=run_id)
    trusted.write(existing)
    before = _snapshot_facts(store, {existing.id})

    missing_id = uuid.uuid4()
    child = make_fact(
        status=status,
        confidence=confidence,
        claim="child-missing-parent",
        run_id=run_id,
        parent_fact_ids=[existing.id, missing_id],
    )

    with pytest.raises(MissingParentFactError) as exc_info:
        writer.write(child)

    assert str(missing_id) in str(exc_info.value)
    assert store.get_by_id(child.id) is None
    _assert_snapshots_unchanged(store, before)


def test_owner_d4_reports_all_and_only_missing_parent_ids(
    store: EvidenceStore, trusted: TrustedWriter
) -> None:
    """Message includes every missing parent ID and excludes stored IDs."""
    MissingParentFactError = _import_missing_parent_fact_error()
    run_id = uuid.uuid4()
    stored = make_fact(claim="stored-parent", run_id=run_id)
    trusted.write(stored)
    missing_a = uuid.uuid4()
    missing_b = uuid.uuid4()

    child = make_fact(
        claim="child-two-missing",
        run_id=run_id,
        parent_fact_ids=[stored.id, missing_a, missing_b],
    )
    with pytest.raises(MissingParentFactError) as exc_info:
        trusted.write(child)

    message = str(exc_info.value)
    assert str(missing_a) in message
    assert str(missing_b) in message
    assert str(stored.id) not in message
    assert store.get_by_id(child.id) is None


def test_owner_d4_rejects_self_parent(
    store: EvidenceStore, trusted: TrustedWriter
) -> None:
    """A candidate that parents itself is rejected as a missing parent."""
    MissingParentFactError = _import_missing_parent_fact_error()
    child_id = uuid.uuid4()
    child = make_fact(
        fact_id=child_id,
        claim="self-parent",
        parent_fact_ids=[child_id],
    )

    with pytest.raises(MissingParentFactError) as exc_info:
        trusted.write(child)

    assert str(child_id) in str(exc_info.value)
    assert store.get_by_id(child_id) is None


def test_owner_d4_duplicate_id_check_precedes_parent_check(
    store: EvidenceStore, trusted: TrustedWriter
) -> None:
    """DuplicateFactError wins when the same id also references a missing parent."""
    fact_id = uuid.uuid4()
    original = make_fact(
        fact_id=fact_id,
        claim="original-before-dup",
        value={"v": 1},
        source=_source(file_path="orig.txt"),
        method=_method(analyzer_name="orig"),
    )
    trusted.write(original)
    before = _snapshot_facts(store, {fact_id})

    missing_parent = uuid.uuid4()
    duplicate = make_fact(
        fact_id=fact_id,
        claim="dup-with-missing-parent",
        value={"v": 999},
        parent_fact_ids=[missing_parent],
    )
    with pytest.raises(DuplicateFactError) as exc_info:
        trusted.write(duplicate)

    assert exc_info.value.fact_id == fact_id
    assert str(fact_id) in str(exc_info.value)
    assert _DUPLICATE_FACT_MESSAGE in str(exc_info.value)
    _assert_snapshots_unchanged(store, before)


def test_owner_d4_missing_parent_failure_never_attempts_insert(
    store: EvidenceStore, trusted: TrustedWriter
) -> None:
    """Missing-parent rejection must not attempt INSERT."""
    MissingParentFactError = _import_missing_parent_fact_error()
    missing_id = uuid.uuid4()
    child = make_fact(
        claim="trace-missing-parent",
        parent_fact_ids=[missing_id],
    )

    insert_statements: list[str] = []

    def _trace(statement: str) -> None:
        if "INSERT" in statement.upper():
            insert_statements.append(statement)

    store._conn.set_trace_callback(_trace)
    try:
        with pytest.raises(MissingParentFactError) as exc_info:
            trusted.write(child)
    finally:
        store._conn.set_trace_callback(None)

    assert str(missing_id) in str(exc_info.value)
    assert insert_statements == []
    assert store.get_by_id(child.id) is None


def test_owner_d4_duplicate_failure_never_attempts_insert(
    store: EvidenceStore, trusted: TrustedWriter
) -> None:
    """Duplicate-id rejection must not attempt INSERT; original unchanged."""
    fact_id = uuid.uuid4()
    original = make_fact(fact_id=fact_id, claim="dup-trace-original", value={"v": 1})
    trusted.write(original)
    before = _snapshot_facts(store, {fact_id})

    duplicate = make_fact(fact_id=fact_id, claim="dup-trace-second", value={"v": 2})
    insert_statements: list[str] = []

    def _trace(statement: str) -> None:
        if "INSERT" in statement.upper():
            insert_statements.append(statement)

    store._conn.set_trace_callback(_trace)
    try:
        with pytest.raises(DuplicateFactError) as exc_info:
            trusted.write(duplicate)
    finally:
        store._conn.set_trace_callback(None)

    assert exc_info.value.fact_id == fact_id
    assert _DUPLICATE_FACT_MESSAGE in str(exc_info.value)
    assert insert_statements == []
    _assert_snapshots_unchanged(store, before)


def test_owner_d4_checks_and_insert_share_one_transaction(
    store: EvidenceStore, trusted: TrustedWriter
) -> None:
    """Candidate lookup, parent coverage lookups, and INSERT share one transaction."""
    import sqlite3

    run_id = uuid.uuid4()
    parents = [make_fact(claim=f"tx-parent-{i}", run_id=run_id) for i in range(2)]
    for parent in parents:
        trusted.write(parent)
    parent_ids = [parent.id for parent in parents]
    child = make_fact(
        claim="tx-child",
        run_id=run_id,
        parent_fact_ids=parent_ids,
    )
    child_id_text = str(child.id)
    parent_id_texts = [str(pid) for pid in parent_ids]

    events: list[tuple[str, bool]] = []

    def _trace(statement: str) -> None:
        events.append((statement, bool(store._conn.in_transaction)))

    store._conn.set_trace_callback(_trace)
    try:
        trusted.write(child)
    finally:
        store._conn.set_trace_callback(None)

    assert store.get_by_id(child.id) is not None
    assert store._conn.in_transaction is False

    def _is_select(sql: str) -> bool:
        return " ".join(sql.upper().split()).startswith("SELECT")

    def _is_insert(sql: str) -> bool:
        upper = " ".join(sql.upper().split())
        return upper.startswith("INSERT") and "FACTS" in upper

    def _is_txn_boundary(sql: str) -> bool:
        upper = " ".join(sql.upper().split())
        return upper.startswith("COMMIT") or upper.startswith("ROLLBACK")

    insert_idxs = [i for i, (sql, _) in enumerate(events) if _is_insert(sql)]
    assert insert_idxs, f"expected INSERT, got {events!r}"
    insert_i = insert_idxs[0]

    candidate_idxs = [
        i
        for i, (sql, _) in enumerate(events)
        if i < insert_i and _is_select(sql) and child_id_text in sql
    ]
    assert (
        candidate_idxs
    ), f"expected candidate-id SELECT before INSERT; events={events!r}"
    candidate_i = candidate_idxs[0]

    parent_lookup_idxs = [
        i
        for i, (sql, _) in enumerate(events)
        if candidate_i < i < insert_i
        and _is_select(sql)
        and any(pid_text in sql for pid_text in parent_id_texts)
    ]
    assert (
        parent_lookup_idxs
    ), f"expected parent-coverage SELECT(s) after candidate lookup; events={events!r}"
    assert candidate_i < parent_lookup_idxs[0] < insert_i

    coverage_sql = " ".join(
        sql
        for i, (sql, _) in enumerate(events)
        if candidate_i < i < insert_i and _is_select(sql)
    )
    for pid_text in parent_id_texts:
        assert pid_text in coverage_sql, (
            f"parent {pid_text} not covered by SELECT(s) before INSERT; "
            f"events={events!r}"
        )

    critical_idxs = [candidate_i, *parent_lookup_idxs, insert_i]
    assert all(events[i][1] for i in critical_idxs), (
        f"candidate/parent lookups and INSERT must share one active transaction; "
        f"events={events!r}"
    )
    between = events[candidate_i : insert_i + 1]
    assert not any(
        _is_txn_boundary(sql) for sql, _ in between
    ), f"no COMMIT/ROLLBACK may separate checks from INSERT; events={events!r}"

    independent = sqlite3.connect(str(store._db_path))
    try:
        row = independent.execute(
            "SELECT 1 FROM facts WHERE id = ?",
            (child_id_text,),
        ).fetchone()
        assert row is not None
    finally:
        independent.close()


def test_owner_d4_duplicate_id_check_precedes_repeated_parent_check(
    store: EvidenceStore, trusted: TrustedWriter
) -> None:
    """DuplicateFactError precedes repeated-parent rejection for the same id."""
    run_id = uuid.uuid4()
    parent_a = make_fact(claim="rep-parent-a", run_id=run_id)
    parent_b = make_fact(claim="rep-parent-b", run_id=run_id)
    trusted.write(parent_a)
    trusted.write(parent_b)

    fact_id = uuid.uuid4()
    original = make_fact(
        fact_id=fact_id,
        claim="original-before-repeated-dup",
        value={"v": 1},
        run_id=run_id,
    )
    trusted.write(original)
    before = _snapshot_facts(store, {fact_id, parent_a.id, parent_b.id})

    duplicate = Fact.model_construct(
        id=fact_id,
        status=EvidenceStatus.OBSERVED,
        claim="dup-with-repeated-parents",
        value={"v": 2},
        source=_source(),
        method=_method(),
        run_id=run_id,
        parent_fact_ids=[parent_a.id, parent_b.id, parent_a.id],
        confidence=None,
        created_at=datetime.now(UTC),
    )
    assert duplicate.parent_fact_ids == [parent_a.id, parent_b.id, parent_a.id]

    with pytest.raises(DuplicateFactError) as exc_info:
        trusted.write(duplicate)

    assert exc_info.value.fact_id == fact_id
    assert _DUPLICATE_FACT_MESSAGE in str(exc_info.value)
    _assert_snapshots_unchanged(store, before)
