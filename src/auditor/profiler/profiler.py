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


def _exact_count_method_provenance(scan_max_records: int) -> dict[str, Any]:
    method = _method_provenance()
    method["parameters"] = {
        "operation": "exact_record_count",
        "scan_max_records": scan_max_records,
    }
    return method


class DatasetProfiler:
    """Metadata-first deterministic profiler over the frozen Connector interface."""

    def profile(
        self,
        connector: Connector,
        writer: TrustedWriter,
        run_id: UUID,
        *,
        scan_max_records: int = 0,
    ) -> list[Fact]:
        if scan_max_records < 0:
            raise ValueError("scan_max_records must be non-negative")

        source_metadata = deepcopy(connector.get_source_metadata())
        capability_descriptor = connector.get_capabilities()
        capabilities = capability_descriptor.model_dump(mode="json")
        estimated_record_count = capability_descriptor.estimated_record_count
        schema = deepcopy(connector.get_schema())

        if estimated_record_count is not None:
            count_fact = Fact(
                status=EvidenceStatus.OBSERVED,
                claim="Connector-reported estimated record count",
                value={
                    "metric": "profiler.record_count.indexed",
                    "value": estimated_record_count,
                },
                source=_source_provenance(source_metadata),
                method=_method_provenance(),
                run_id=run_id,
                parent_fact_ids=[],
                confidence=None,
            )
        else:
            count_fact = Fact(
                status=EvidenceStatus.UNKNOWN,
                claim="Connector did not report an estimated record count",
                value={
                    "metric": "profiler.record_count.indexed",
                    "value": None,
                    "reason": "metadata_unavailable",
                    "explanation": (
                        "Connector did not provide an estimated record count."
                    ),
                },
                source=_source_provenance(source_metadata),
                method=_method_provenance(),
                run_id=run_id,
                parent_fact_ids=[],
                confidence=None,
            )

        scan_authorized = (
            scan_max_records > 0
            and capability_descriptor.has_index_metadata
            and not capability_descriptor.media_is_referenced_not_present
            and estimated_record_count is not None
            and estimated_record_count <= scan_max_records
        )

        if scan_authorized:
            exact_record_count = sum(1 for _ in connector.list_records())
            exact_fact = Fact(
                status=EvidenceStatus.COMPUTED,
                claim="Exact record count computed from an authorized record scan",
                value={
                    "metric": "profiler.record_count.exact",
                    "value": exact_record_count,
                },
                source=_source_provenance(source_metadata),
                method=_exact_count_method_provenance(scan_max_records),
                run_id=run_id,
                parent_fact_ids=[],
                confidence=None,
            )
        else:
            exact_fact = Fact(
                status=EvidenceStatus.UNKNOWN,
                claim="Exact record count was not computed",
                value={
                    "metric": "profiler.record_count.exact",
                    "value": None,
                    "reason": "scan_not_authorized",
                    "explanation": (
                        "Exact record count was not computed because "
                        "a full record scan was not authorized."
                    ),
                },
                source=_source_provenance(source_metadata),
                method=_exact_count_method_provenance(scan_max_records),
                run_id=run_id,
                parent_fact_ids=[],
                confidence=None,
            )

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
            count_fact,
            exact_fact,
        ]

        for fact in facts:
            writer.write(fact)
        return facts
