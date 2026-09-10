"""Analyzer interface and capability model (design §4.5)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from auditor.connectors.base import Connector
from auditor.evidence.models import EvidenceStatus, Fact
from auditor.evidence.writers import TrustedWriter

_ANALYZER_EMIT_STATUSES = frozenset(
    {
        EvidenceStatus.OBSERVED,
        EvidenceStatus.COMPUTED,
        EvidenceStatus.DERIVED,
    }
)


class CostTier(StrEnum):
    """Cost tier for plan validation and approval gating (design §6)."""

    FREE = "FREE"
    CHEAP = "CHEAP"
    MODERATE = "MODERATE"
    EXPENSIVE = "EXPENSIVE"


class AnalyzerCapability(BaseModel):
    """
    Self-describing analyzer registration payload (design §4.5 / §7).

    emits_status is constrained to OBSERVED / COMPUTED / DERIVED — analyzers
    never emit INFERRED or DISCREPANCY (those come from planner / cross-check).
    """

    model_config = ConfigDict(frozen=True)

    analyzer_name: str
    version: str
    modality: str
    cost_tier: CostTier
    emits_status: EvidenceStatus
    questions_answered: list[str]
    required_fields: list[str]

    @model_validator(mode="after")
    def _validate_emits_status(self) -> Self:
        if self.emits_status not in _ANALYZER_EMIT_STATUSES:
            raise ValueError(
                f"analyzers may only emit {_ANALYZER_EMIT_STATUSES}; "
                f"got {self.emits_status!r}"
            )
        return self


class Analyzer(ABC):
    """
    Pluggable analysis unit.

    Deterministic analyzers emit COMPUTED (or OBSERVED); model-based ones
    emit DERIVED. Writes go only through TrustedWriter.
    """

    @abstractmethod
    def get_capability(self) -> AnalyzerCapability:
        """Declare name, cost tier, modality, and questions answered."""

    @abstractmethod
    def estimate_cost(self, record_count: int) -> float:
        """Return an estimated cost for analyzing record_count records."""

    @abstractmethod
    def run(
        self,
        connector: Connector,
        writer: TrustedWriter,
        record_ids: list[str] | None = None,
    ) -> list[Fact]:
        """
        Execute analysis and write facts via TrustedWriter.

        record_ids=None means a full-dataset run; otherwise only the given
        sample is analyzed.
        """
