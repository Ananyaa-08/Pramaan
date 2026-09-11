"""Tests for status-partitioned writers (design §3.1 structural enforcement)."""

from __future__ import annotations

import ast
import uuid
from pathlib import Path

import pytest

from auditor.evidence.models import EvidenceStatus, Fact
from auditor.evidence.store import EvidenceStore
from auditor.evidence.writers import InferredWriter, TrustedWriter


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


TRUSTED_STATUSES = (
    EvidenceStatus.OBSERVED,
    EvidenceStatus.COMPUTED,
    EvidenceStatus.DERIVED,
    EvidenceStatus.DOCUMENTED,
    EvidenceStatus.DISCREPANCY,
    EvidenceStatus.UNKNOWN,
)

INFERRED_STATUSES = (EvidenceStatus.INFERRED,)


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
# TrustedWriter partition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", TRUSTED_STATUSES)
def test_trusted_writer_accepts_allowed_statuses(
    trusted: TrustedWriter, store: EvidenceStore, status: EvidenceStatus
) -> None:
    fact = make_fact(status=status)
    trusted.write(fact)
    loaded = store.get_by_id(fact.id)
    assert loaded is not None
    assert loaded.status == status


@pytest.mark.parametrize("status", INFERRED_STATUSES)
def test_trusted_writer_rejects_inferred_statuses(
    trusted: TrustedWriter, status: EvidenceStatus
) -> None:
    fact = make_fact(status=status, confidence=0.5)
    with pytest.raises(
        PermissionError, match=f"TrustedWriter cannot write status {status.value}"
    ):
        trusted.write(fact)


# ---------------------------------------------------------------------------
# InferredWriter partition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", INFERRED_STATUSES)
def test_inferred_writer_accepts_allowed_statuses(
    inferred: InferredWriter, store: EvidenceStore, status: EvidenceStatus
) -> None:
    fact = make_fact(status=status, confidence=0.9, claim=f"{status.name}")
    inferred.write(fact)
    loaded = store.get_by_id(fact.id)
    assert loaded is not None
    assert loaded.status == status


@pytest.mark.parametrize("status", TRUSTED_STATUSES)
def test_inferred_writer_rejects_trusted_statuses(
    inferred: InferredWriter, status: EvidenceStatus
) -> None:
    fact = make_fact(status=status)
    with pytest.raises(
        PermissionError, match=f"InferredWriter cannot write status {status.value}"
    ):
        inferred.write(fact)


# ---------------------------------------------------------------------------
# Confidence rule at Fact construction (both directions)
# ---------------------------------------------------------------------------


def test_inferred_fact_without_confidence_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="confidence"):
        make_fact(status=EvidenceStatus.INFERRED, confidence=None)


def test_non_inferred_fact_with_confidence_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="confidence"):
        make_fact(status=EvidenceStatus.COMPUTED, confidence=0.8)


# ---------------------------------------------------------------------------
# Structural enforcement: _insert only called from writers.py / store.py
# ---------------------------------------------------------------------------


def _project_roots() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2]
    return root / "src", root / "tests"


def _python_files(base: Path) -> list[Path]:
    return sorted(p for p in base.rglob("*.py") if p.is_file())


def _calls_named_insert(path: Path) -> list[tuple[int, str]]:
    """Return (lineno, snippet) for calls to _insert."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "_insert":
            hits.append(
                (node.lineno, ast.get_source_segment(source, node) or "_insert")
            )
        if isinstance(func, ast.Name) and func.id == "_insert":
            hits.append(
                (node.lineno, ast.get_source_segment(source, node) or "_insert")
            )
    return hits


def test_insert_only_called_from_writers_or_store() -> None:
    """
    `_insert` must only be invoked from writers.py or store.py.

    Encodes the doc's partitioned write API without name-mangling: callers
    outside those modules must use TrustedWriter / InferredWriter.
    """
    src_root, tests_root = _project_roots()
    allowed = {
        (src_root / "auditor" / "evidence" / "writers.py").resolve(),
        (src_root / "auditor" / "evidence" / "store.py").resolve(),
    }

    offenders: list[str] = []
    for base in (src_root, tests_root):
        for path in _python_files(base):
            if path.resolve() in allowed:
                continue
            for lineno, snippet in _calls_named_insert(path):
                offenders.append(f"{path}:{lineno}: {snippet}")

    assert offenders == [], (
        "_insert() must only be called from writers.py or store.py; found:\n"
        + "\n".join(offenders)
    )


def test_insert_not_reexported_from_evidence_init() -> None:
    init_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "auditor"
        / "evidence"
        / "__init__.py"
    )
    if not init_path.exists():
        return
    source = init_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assert alias.name != "_insert"
                assert alias.asname != "_insert"
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__all__":
                    names = ast.literal_eval(node.value)
                    assert "_insert" not in names


def test_insert_absent_from_evidencestore_public_docstring() -> None:
    doc = EvidenceStore.__doc__ or ""
    assert "_insert" not in doc
