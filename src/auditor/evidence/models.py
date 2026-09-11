"""Evidence fact model and epistemic statuses (design §3.1–3.2)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceStatus(StrEnum):
    """Epistemic status of a recorded claim (design §3.1)."""

    OBSERVED = "OBSERVED"
    COMPUTED = "COMPUTED"
    DERIVED = "DERIVED"
    DOCUMENTED = "DOCUMENTED"
    INFERRED = "INFERRED"
    UNKNOWN = "UNKNOWN"
    DISCREPANCY = "DISCREPANCY"


class Fact(BaseModel):
    """
    Immutable evidence fact with full provenance (design §3.2).

    Facts form a provenance DAG via parent_fact_ids. Confidence is required
    for INFERRED facts and must be absent for every other status.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    status: EvidenceStatus
    claim: str
    value: dict[str, Any]
    source: dict[str, Any]
    method: dict[str, Any]
    run_id: UUID
    parent_fact_ids: list[UUID] = Field(default_factory=list)
    confidence: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _validate_confidence_rule(self) -> Self:
        if self.status is EvidenceStatus.INFERRED:
            if self.confidence is None:
                raise ValueError(
                    "confidence is required when status is INFERRED (0.0–1.0)"
                )
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError("confidence must be between 0.0 and 1.0 inclusive")
        elif self.confidence is not None:
            raise ValueError("confidence must be None unless status is INFERRED")
        return self
