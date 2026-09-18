"""Immutable Phase 3 cost-gate models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
)


class WorkUnit(StrEnum):
    """Observable work units currently consumed by Phase 2 payloads."""

    RECORDS_TO_SCAN = "RECORDS_TO_SCAN"
    FILES_TO_OPEN = "FILES_TO_OPEN"


class WorkItem(BaseModel):
    """One unit of estimated work with an optional unknown magnitude."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    unit: WorkUnit
    magnitude: StrictInt | None = Field(ge=0)


class BudgetLimit(BaseModel):
    """Hard non-negative ceiling for one work unit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    unit: WorkUnit
    maximum: StrictInt = Field(ge=0)


class WorkEstimate(BaseModel):
    """Normalized, duplicate-free collection of work items."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: tuple[WorkItem, ...]

    @field_validator("items", mode="after")
    @classmethod
    def _normalize_items(cls, items: tuple[WorkItem, ...]) -> tuple[WorkItem, ...]:
        units = [item.unit for item in items]
        if len(units) != len(set(units)):
            raise ValueError("duplicate WorkUnit values are not allowed")
        return tuple(sorted(items, key=lambda item: item.unit.value))


class Budget(BaseModel):
    """Normalized, duplicate-free collection of per-unit budget limits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    limits: tuple[BudgetLimit, ...]

    @field_validator("limits", mode="after")
    @classmethod
    def _normalize_limits(
        cls, limits: tuple[BudgetLimit, ...]
    ) -> tuple[BudgetLimit, ...]:
        units = [limit.unit for limit in limits]
        if len(units) != len(set(units)):
            raise ValueError("duplicate WorkUnit values are not allowed")
        return tuple(sorted(limits, key=lambda limit: limit.unit.value))
