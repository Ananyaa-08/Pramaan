"""Deterministic Dataset Profiler (Phase 2)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import UUID

from auditor.connectors.base import Connector
from auditor.evidence.models import EvidenceStatus, Fact
from auditor.evidence.writers import TrustedWriter

_PROFILER_VERSION = "0.1.0"


def _source_provenance(source_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_uri": source_metadata["source_uri"],
        "revision": source_metadata["revision"],
        "file_path": None,
        "field": None,
        "record_id": None,
    }


def _method_provenance() -> dict[str, Any]:
    return {
        "analyzer_name": "dataset_profiler",
        "analyzer_version": _PROFILER_VERSION,
        "model_name": None,
        "model_version": None,
        "parameters": {"operation": "metadata_profiling"},
        "sampling_strategy": None,
        "sampling_seed": None,
    }


class DatasetProfiler:
    """Metadata-first deterministic profiler over the frozen Connector interface."""

    def profile(
        self,
        connector: Connector,
        writer: TrustedWriter,
        run_id: UUID,
    ) -> list[Fact]:
        source_metadata = deepcopy(connector.get_source_metadata())
        capabilities = connector.get_capabilities().model_dump(mode="json")
        schema = deepcopy(connector.get_schema())

        facts = [
            Fact(
                status=EvidenceStatus.OBSERVED,
                claim="Connector source metadata as reported by the connector",
                value={
                    "metric": "profiler.source_metadata",
                    "value": deepcopy(source_metadata),
                },
                source=_source_provenance(source_metadata),
                method=_method_provenance(),
                run_id=run_id,
                parent_fact_ids=[],
                confidence=None,
            ),
            Fact(
                status=EvidenceStatus.OBSERVED,
                claim="Connector-reported capabilities as published by the connector",
                value={
                    "metric": "profiler.capabilities",
                    "value": deepcopy(capabilities),
                },
                source=_source_provenance(source_metadata),
                method=_method_provenance(),
                run_id=run_id,
                parent_fact_ids=[],
                confidence=None,
            ),
            Fact(
                status=EvidenceStatus.OBSERVED,
                claim="Connector-reported schema as returned by the connector",
                value={
                    "metric": "profiler.schema",
                    "value": deepcopy(schema),
                },
                source=_source_provenance(source_metadata),
                method=_method_provenance(),
                run_id=run_id,
                parent_fact_ids=[],
                confidence=None,
            ),
        ]

        for fact in facts:
            writer.write(fact)
        return facts
