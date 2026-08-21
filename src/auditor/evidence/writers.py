"""Status-partitioned write handles (design §3.1 structural enforcement)."""

from __future__ import annotations

from auditor.evidence.models import EvidenceStatus, Fact
from auditor.evidence.store import EvidenceStore

_TRUSTED_STATUSES = frozenset(
    {
        EvidenceStatus.OBSERVED,
        EvidenceStatus.COMPUTED,
        EvidenceStatus.DERIVED,
        EvidenceStatus.DOCUMENTED,
        EvidenceStatus.DISCREPANCY,
    }
)

_INFERRED_STATUSES = frozenset(
    {
        EvidenceStatus.INFERRED,
        EvidenceStatus.UNKNOWN,
    }
)


class TrustedWriter:
    """
    Write handle for deterministic / documentation evidence.

    Accepts OBSERVED, COMPUTED, DERIVED, DOCUMENTED, DISCREPANCY.
    Rejects INFERRED and UNKNOWN with PermissionError.
    """

    def __init__(self, store: EvidenceStore) -> None:
        self._store = store

    def write(self, fact: Fact) -> None:
        if fact.status not in _TRUSTED_STATUSES:
            raise PermissionError(
                f"TrustedWriter cannot write status {fact.status.value}; "
                "use InferredWriter for INFERRED/UNKNOWN"
            )
        self._store._insert(fact)


class InferredWriter:
    """
    Write handle for the AI reasoning layer.

    Accepts only INFERRED and UNKNOWN. Physically cannot write OBSERVED or
    COMPUTED (or other trusted statuses) — raises PermissionError.
    """

    def __init__(self, store: EvidenceStore) -> None:
        self._store = store

    def write(self, fact: Fact) -> None:
        if fact.status not in _INFERRED_STATUSES:
            raise PermissionError(
                f"InferredWriter cannot write status {fact.status.value}; "
                "use TrustedWriter for OBSERVED/COMPUTED/DERIVED/DOCUMENTED/DISCREPANCY"
            )
        self._store._insert(fact)
