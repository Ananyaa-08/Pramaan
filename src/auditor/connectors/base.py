"""Connector interface and capability descriptor (design §4.1)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict


class CapabilityDescriptor(BaseModel):
    """
    What a connector can do without performing analysis.

    Drives profiler/planner honesty: cheap full-dataset stats are only
    attempted when has_index_metadata (or equivalent) makes them genuinely cheap.
    """

    model_config = ConfigDict(frozen=True)

    can_stream: bool
    can_random_access: bool
    has_index_metadata: bool
    media_is_referenced_not_present: bool
    requires_auth: bool
    estimated_record_count: int | None


class Connector(ABC):
    """
    Abstract access adapter for a dataset source.

    Pure access — no intelligence and no analysis. Concrete adapters
    (local files, Hugging Face, GitHub, …) implement this contract later.
    """

    @abstractmethod
    def get_capabilities(self) -> CapabilityDescriptor:
        """Publish what this source can support."""

    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """Field names/types discoverable without a full scan."""

    @abstractmethod
    def list_records(
        self, limit: int | None = None
    ) -> Iterator[dict[str, Any]]:
        """
        Lazily stream records.

        Must return a generator/iterator — never a fully materialized list —
        so large corpora (e.g. 118k videos) can be audited without loading
        everything into memory.
        """

    @abstractmethod
    def get_record(self, record_id: str) -> dict[str, Any]:
        """
        Fetch one record by id.

        Must raise NotImplementedError when capabilities.can_random_access
        is False.
        """

    @abstractmethod
    def get_source_metadata(self) -> dict[str, Any]:
        """
        Source coordinates for reproducibility.

        Expected keys include source_uri, revision (when pinnable), and
        provider.
        """
