"""Evidence-layer errors."""

from __future__ import annotations

from uuid import UUID


class DuplicateFactError(Exception):
    """Raised when inserting a Fact whose id already exists in the store."""

    def __init__(self, fact_id: object) -> None:
        self.fact_id = fact_id
        super().__init__(f"Fact with id {fact_id} already exists in the evidence store")


class MissingParentFactError(Exception):
    """Raised when a Fact references parents absent from the store."""

    def __init__(self, missing_parent_ids: list[UUID]) -> None:
        self.missing_parent_ids = missing_parent_ids
        super().__init__(
            "Missing parent facts: " + ", ".join(map(str, missing_parent_ids))
        )
