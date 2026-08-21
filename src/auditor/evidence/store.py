"""Append-only SQLite evidence store — provenance DAG hub (design §3)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import UUID

from auditor.evidence.errors import DuplicateFactError
from auditor.evidence.models import EvidenceStatus, Fact


def _validate_confidence_rule(fact: Fact) -> None:
    """Enforce the §3.1 confidence rule (also catches model_construct bypasses)."""
    if fact.status is EvidenceStatus.INFERRED:
        if fact.confidence is None:
            raise ValueError(
                "confidence is required when status is INFERRED (0.0–1.0)"
            )
        if not 0.0 <= fact.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0 inclusive")
    elif fact.confidence is not None:
        raise ValueError("confidence must be None unless status is INFERRED")


class EvidenceStore:
    """
    Append-only evidence store backed by SQLite.

    Public surface is read-oriented: get_by_id, get_by_status, get_by_run,
    and get_lineage. New facts enter only through status-partitioned writers
    (TrustedWriter / InferredWriter).
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                claim TEXT NOT NULL,
                value TEXT NOT NULL,
                source TEXT NOT NULL,
                method TEXT NOT NULL,
                run_id TEXT NOT NULL,
                parent_fact_ids TEXT NOT NULL,
                confidence REAL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def _insert(self, fact: Fact) -> None:
        _validate_confidence_rule(fact)

        # Existence check before any write — original row must stay untouched.
        existing = self._conn.execute(
            "SELECT 1 FROM facts WHERE id = ?",
            (str(fact.id),),
        ).fetchone()
        if existing is not None:
            raise DuplicateFactError(fact.id)

        self._conn.execute(
            """
            INSERT INTO facts (
                id, status, claim, value, source, method,
                run_id, parent_fact_ids, confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(fact.id),
                fact.status.value,
                fact.claim,
                json.dumps(fact.value),
                json.dumps(fact.source),
                json.dumps(fact.method),
                str(fact.run_id),
                json.dumps([str(pid) for pid in fact.parent_fact_ids]),
                fact.confidence,
                fact.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def get_by_id(self, fact_id: UUID) -> Fact | None:
        row = self._conn.execute(
            "SELECT * FROM facts WHERE id = ?",
            (str(fact_id),),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_fact(row)

    def get_by_status(self, status: EvidenceStatus) -> list[Fact]:
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE status = ?",
            (status.value,),
        ).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def get_by_run(self, run_id: UUID) -> list[Fact]:
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE run_id = ?",
            (str(run_id),),
        ).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def get_lineage(self, fact_id: UUID) -> list[Fact]:
        """
        Return the ancestor chain as an ordered list, root-first.

        The requested fact is always the last element. Walks parent_fact_ids
        so a reader can answer "why did you conclude this?" top to bottom.
        """
        fact = self.get_by_id(fact_id)
        if fact is None:
            raise ValueError(f"Fact not found: {fact_id}")

        chain: list[Fact] = []
        visited: set[UUID] = set()

        def walk(current: Fact) -> None:
            if current.id in visited:
                return
            visited.add(current.id)
            for parent_id in current.parent_fact_ids:
                parent = self.get_by_id(parent_id)
                if parent is None:
                    raise ValueError(f"Missing parent fact: {parent_id}")
                walk(parent)
            chain.append(current)

        walk(fact)
        return chain

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> Fact:
        return Fact.model_construct(
            id=UUID(row["id"]),
            status=EvidenceStatus(row["status"]),
            claim=row["claim"],
            value=json.loads(row["value"]),
            source=json.loads(row["source"]),
            method=json.loads(row["method"]),
            run_id=UUID(row["run_id"]),
            parent_fact_ids=[UUID(pid) for pid in json.loads(row["parent_fact_ids"])],
            confidence=row["confidence"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
