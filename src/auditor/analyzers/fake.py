"""Trivial CountRecordsAnalyzer — proof-of-life for the analyzer chain."""

from __future__ import annotations

import uuid
from typing import Any

from auditor.analyzers.base import Analyzer, AnalyzerCapability, CostTier
from auditor.connectors.base import Connector
from auditor.evidence.models import EvidenceStatus, Fact
from auditor.evidence.writers import TrustedWriter


class CountRecordsAnalyzer(Analyzer):
    """
    Count records from a connector and write one COMPUTED fact.

    Smallest end-to-end slice: connector -> analyzer -> TrustedWriter -> store.
    """

    def get_capability(self) -> AnalyzerCapability:
        return AnalyzerCapability(
            analyzer_name="count_records",
            version="0.1.0",
            modality="tabular",
            cost_tier=CostTier.FREE,
            emits_status=EvidenceStatus.COMPUTED,
            questions_answered=["record_count"],
            required_fields=[],
        )

    def estimate_cost(self, record_count: int) -> float:
        return float(record_count) * 0.0

    def run(
        self,
        connector: Connector,
        writer: TrustedWriter,
        record_ids: list[str] | None = None,
    ) -> list[Fact]:
        if not isinstance(writer, TrustedWriter):
            raise TypeError(
              "CountRecordsAnalyzer.run requires a TrustedWriter; "
               f"got {type(writer).__name__}"
            )

        cap = self.get_capability()
        count = self._count(connector, record_ids)
        source_meta = connector.get_source_metadata()

        fact = Fact(
            status=EvidenceStatus.COMPUTED,
            claim=f"dataset contains {count} records",
            value={"metric": "record_count", "value": count},
            source={
                "source_uri": source_meta.get("source_uri"),
                "revision": source_meta.get("revision"),
                "file_path": None,
                "field": None,
                "record_id": None,
            },
            method={
                "analyzer_name": cap.analyzer_name,
                "analyzer_version": cap.version,
                "model_name": None,
                "model_version": None,
                "parameters": {"record_ids": record_ids},
                "sampling_strategy": (
                    "explicit_ids" if record_ids is not None else None
                ),
                "sampling_seed": None,
            },
            run_id=uuid.uuid4(),
            parent_fact_ids=[],
            confidence=None,
        )
        writer.write(fact)
        return [fact]

    @staticmethod
    def _count(
        connector: Connector, record_ids: list[str] | None
    ) -> int:
        if record_ids is None:
            return sum(1 for _ in connector.list_records())

        id_set = set(record_ids)
        total = 0
        for record in connector.list_records():
            record_id = _record_id(record)
            if record_id is not None and record_id in id_set:
                total += 1
        return total


def _record_id(record: dict[str, Any]) -> str | None:
    if "id" in record:
        return str(record["id"])
    return None
