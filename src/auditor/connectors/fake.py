"""In-memory Connector test double (design §4.1 interface freeze)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from auditor.connectors.base import CapabilityDescriptor, Connector


class InMemoryConnector(Connector):
    """
    Connector backed by an in-memory iterable of record dicts.

    Used as the test double for profiler/analyzer work before real
    adapters exist. Accepts a list or a generator so laziness of
    list_records can be proven without materializing the source.
    """

    def __init__(
        self,
        records: Iterable[dict[str, Any]],
        schema: dict[str, Any],
        source_metadata: dict[str, Any],
        capabilities: CapabilityDescriptor,
        id_field: str = "id",
    ) -> None:
        # Keep the iterable as-is — do not materialize into a list here.
        self._records = records
        self._schema = schema
        self._source_metadata = source_metadata
        self._capabilities = capabilities
        self._id_field = id_field

    def get_capabilities(self) -> CapabilityDescriptor:
        return self._capabilities

    def get_schema(self) -> dict[str, Any]:
        return self._schema

    def list_records(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
        count = 0
        for record in self._records:
            if limit is not None and count >= limit:
                return
            yield record
            count += 1

    def get_record(self, record_id: str) -> dict[str, Any]:
        if not self._capabilities.can_random_access:
            raise NotImplementedError(
                "This connector does not support random access "
                "(can_random_access=False)"
            )
        for record in self._records:
            if str(record.get(self._id_field)) == record_id:
                return record
        raise KeyError(record_id)

    def get_source_metadata(self) -> dict[str, Any]:
        return self._source_metadata
