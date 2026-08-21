"""Evidence-layer errors."""

from __future__ import annotations


class DuplicateFactError(Exception):
    """Raised when inserting a Fact whose id already exists in the store."""

    def __init__(self, fact_id: object) -> None:
        self.fact_id = fact_id
        super().__init__(f"Fact with id {fact_id} already exists in the evidence store")
